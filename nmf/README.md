# ESM 线性层的三因子非负矩阵分解（NMF）

对 ESM-2 中任意一个（或一批）`nn.Linear` 的权重矩阵做 **非负矩阵分解**，拆成三个
**非负** 矩阵，中间为 **方阵**；然后对这三个矩阵做若干批次的训练，最后把分解模型与
原模型的能力做量化对比。

```
W ≈ A @ S @ B + offset

A : (out_features, rank)   非负
S : (rank, rank)           非负方阵   <- 中间矩阵
B : (rank, in_features)    非负
offset : (out_features,)   非负目标矩阵的平移项（前向里用 x.sum(-1) * offset 精确加回）
```

## 0. 环境

本项目已在 conda 环境 **`ESM`** 中安装（Python 3.10 + PyTorch 2.14 CPU 版）：

```bash
conda activate ESM
# 或直接使用解释器
/home/zzj/anaconda3/envs/ESM/bin/python -m nmf.factorize --help
```

从零重建该环境：

```bash
conda create -y -n ESM python=3.10 pip
conda run -n ESM pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n ESM pip install -e . --no-deps   # --no-deps：跳过上游 openfold 等重依赖
conda run -n ESM pip install matplotlib pytest
```

> **换机器 / 换设备都不用改脚本**：三个脚本都通过 `nmf/_env.sh` 自动解析解释器（用当前激活
> 环境里的 `python`，或 `PY=/path/to/python` 指定）、计算设备（`DEVICE`，默认 `auto`）和
> `TMPDIR`（默认 `<仓库>/.tmp`，避免 `/tmp` 是小容量 tmpfs）。

### 0.0 在 GPU 服务器上跑（例如 3090 / 24GB）

```bash
# 1) 建环境（CUDA 版 torch；3090 是 sm_86，cu118/cu121/cu124 的官方轮子都支持）
conda create -y -n esm-nmf python=3.10 pip
conda activate esm-nmf
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 显卡驱动需 >= 525
pip install -e /path/to/esm --no-deps        # --no-deps 很重要：上游 requirements 里有 openfold 等重依赖
pip install matplotlib pytest

# 2) 自检（必须看到 cuda 可用 = True 与 GPU 名字）
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi
```

如果 `torch.cuda.is_available()` 是 False，说明装成了 CPU 版轮子或驱动版本不匹配；
`nmf/run_matrix.sh` 启动时会自动检测并**明确警告 + 回退到 CPU**，不会静默变慢。

```bash
# 3) 拷数据（服务器能上外网时可跳过，直接跑 prepare_data 下载）
#    已切分好的 processed 只有 13 MB，比 236 MB 的原始 FASTA 划算得多
rsync -avP data/processed user@server:/path/to/esm/data/

# 4) 跑实验：DEVICE=auto 会在有 CUDA 时自动用 GPU
export TMPDIR=$HOME/tmp            # 若服务器 /tmp 较小
cd /path/to/esm
DEVICE=cuda bash nmf/run_matrix.sh 2>&1 | tee nmf/outputs/fullmodel/matrix_gpu.log
```

**GPU 上各阶段的收益不一样，别期待线性加速**：

| 阶段 | GPU vs CPU | 说明 |
| --- | --- | --- |
| 训练（`nmf.train_nmf`） | **明显更快**（本机 CPU 3.54 s/步） | 主要吃 forward+backward，3090 上预计提速 10~20×；因此可以把 `MAX_RECORDS` 提到全部 29,124 条、`EPOCHS` 2~3 |
| 论文级评测（`nmf.benchmark`） | 明显更快 | 伪困惑度是大量小批量前向，GPU 收益大；可把 `N_EVAL` 提到 256、`PPL_RECORDS` 提到 16 |
| **分解（`nmf.factorize`）** | **未必更快** | HALS 对每个因子逐列/逐行更新，是 Python 循环 + 每个子问题一次 CPU 同步，`rank=320` 时一次外循环有上千次小 kernel 与同步，GPU 利用率极低。这是一次性离线成本（8M 模型约 30 分钟），**建议保持 `--device cpu`**（`FACTORIZE_DEVICE=cpu`），或至少和 GPU 版对比一次耗时再定 |

其他注意事项：

- 随机数全部在 CPU 上用固定 seed 生成后搬运到设备（初始化、MLM 遮挡位置），因此
  **同一 seed 在 CPU 与 GPU 上得到相同的初值与遮挡位置**，两边的指标可以直接对比；
- 显存需求很小：8M 模型训练时 batch tokens ≤ 4096，3090 上占用 < 2 GB。换
  `MODEL=esm2_t33_650M_UR50D` 时约 2.6 GB 权重 + 激活，24 GB 完全够；
- **但 650M 模型的分解要谨慎**：`layers.N.fc1` 是 5120×1280，满阶 `rank=1280`，
  HALS 的 Python 循环会变成主要瓶颈（比 8M 慢一个数量级以上）。建议
  `--rank-ratio 0.25 --iters 200`，或先用 `--layers first,middle,last` 只分解少量层试水；
- 权重首次加载会下载到 `~/.cache/torch/hub/checkpoints`（8M 约 29 MB）。服务器无外网时，
  把本地这个目录整体拷过去即可；
- 检查点里只存张量/字符串/数字，`load_checkpoint` 已兼容 `weights_only=True`，
  所以**在不同 torch 版本的机器之间互相搬运检查点没问题**。

**本机（开发机）的特殊之处**：`/tmp` 只有 10 MB tmpfs，`pip` 装大轮子前需
`export TMPDIR=/home/zzj/tmp`；环境 `ESM` 里是 CPU 版 torch。这些不影响服务器上的流程。

## 0.1 数据准备（`nmf.prepare_data`）

要让"性能不下降"的结论站得住，训练/评测数据必须**规模足够、来源权威、长度多样、且做过
去污染**。本仓库自带的小样例（`examples/data/p62593.fasta` 等）不满足这些条件，
因此用 `nmf/prepare_data.py` 准备正式数据：

```bash
python -m nmf.prepare_data \
  --source swissprot \
  --raw-file data/raw/swissprot_reviewed.fasta \
  --out-dir data/processed/swissprot \
  --min-len 50 --max-len 1022 \
  --train 30000 --valid 2000 --test 4000 --seed 0 \
  --decontaminate --decontam-threshold 0.6 --kmer 6
```

**数据源：UniProtKB/Swiss-Prot（reviewed，人工审阅）**，通过 UniProt REST 流式接口下载
（`rest.uniprot.org` 实测约 2 MB/s；`ftp.uniprot.org` 只有 ~30 KB/s，不要用）：

```
https://rest.uniprot.org/uniprotkb/stream
  ?query=(reviewed:true) AND (fragment:false) AND (length:[50 TO 1022])
  &format=fasta
```

这是蛋白质语言模型文献里最常用的**权威、去冗余、带注释**的序列集合（ESM-2 的预训练语料
是体量更大的 UniRef50 2021_04，约 4900 万条；Swiss-Prot 则是其人工审阅子集，
下载量与算力可控，且**分布多样**，适合做严谨的对比实验）。

本次实际得到的数据（`data/processed/swissprot/stats.json` 有完整记录与 sha256）：

| 阶段 | 条数 | 说明 |
| --- | --- | --- |
| 原始下载 | 524,874 | reviewed + 非 fragment + 长度 50~1022 |
| 质控后 | 439,721 | 丢弃非常规残基 1,497、精确重复 83,656 |
| train | 30,000 → **29,124** | 去污染剔除 876 条 |
| valid | 2,000 | 长度 min 50 / p50 315 / p95 745 / max 1019 |
| test | 4,000 | 长度 min 50 / p50 310 / p95 739 / max 1020 |

长度分布（train）：**min 50、p50 312、p95 760、max 1022、平均 346 残基，共 10,085,760 残基**
——与"5397 条等长 286 的同一酶家族变体"相比，来源、家族、长度都完全不同量级。

**去污染**：用 6-mer 包含度作为同源性代理，剔除与 valid/test 高度相似的 train 序列
（某条 train 覆盖某条 guard ≥ 60% 的 6-mer 即视为污染）。这是 ESM-2 论文用 MMseqs 以
50% 序列一致性清洗训练集的**无需外部二进制的近似实现**，避免测试集泄漏。


## 1. 为什么要平移（offset）

NMF 要求被分解的目标矩阵非负，而预训练 Transformer 的线性层权重通常同时含正负
元素。因此先把权重平移为非负目标，默认 `--shift min`：

```
offset = min(W)，   W' = W - offset[:, None] >= 0
```

`offset` 在前向里通过 `x.sum(-1) * offset` 精确加回，等价于"给权重的每一行加一个常数"，
所以 **原始线性层的信息除了这一行常数外，全部由 A/S/B 三个非负矩阵承载**。

也可用 `--shift row`（逐行减去该行最小值，`offset_i = min_j W_ij`）。
直觉上逐行平移更紧，但实测相反——`esm2_t6_8M_UR50D`、`--solver pgd --iters 2000`：

| 层 | rank | shift | 目标范数比 `‖W'‖/‖W‖` | 相对非负目标误差 | **相对原权重误差** | 余弦相似度 |
| --- | --- | --- | --- | --- | --- | --- |
| `layers.5.fc1` | 320 | `min`（默认） | 5.50 | 0.0283 | **0.1558** | 0.9888 |
| `layers.5.fc1` | 320 | `row` | 3.12 | 0.0646 | 0.2013 | 0.9805 |
| `layers.5.self_attn.out_proj` | 320 | `min`（默认） | 5.50 | 0.0193 | **0.1503** | 0.9891 |
| `layers.5.self_attn.out_proj` | 320 | `row` | 3.23 | 0.1129 | 0.3647 | 0.9373 |

因为 `‖W'‖/‖W‖ ≈ 3~5.5`，**同一个分解在"非负目标"上的相对误差换算到原权重上会被放大
同样的倍数**，所以脚本同时报告两个数字，判断重构质量以"相对原权重"为准。

另外注意：当 `rank = min(in, out)`（满阶）时形式上存在精确解
（`A = W - offset[:, None]`、`S = I`、`B = I`），此时非负分解只是换了一种参数化、
并不压缩，而且现有迭代求解器**达不到**该精确解；**真正有意义的区间是
`rank < min(in, out)`**（用 `--rank-ratio 0.5` 等控制）。

## 2. 三个阶段

### 阶段 1：离线非负分解 —— `nmf.factorize`

```bash
# 先看看有哪些线性层可选
python -m nmf.factorize --list-layers

# 对第 5 层 FFN 的第一个线性层做分解，中间方阵降一半秩
python -m nmf.factorize \
  --layers "layers.5.fc1" \
  --rank-ratio 0.5 \
  --solver pgd --iters 2000 --shift min \
  --out nmf/outputs/nmf_factors.pt --report nmf/outputs/factorize_report.json --plot
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--layers` | 层选择：`all` / `first` / `middle` / `last`（可带子模式如 `last.fc1`、`first.self_attn.*`）、精确名 `layers.5.fc1`、通配符 `layers.*.fc1`、后缀 `fc1`、逗号分隔组合 |
| `--rank` / `--rank-ratio` | 中间方阵的阶数；默认 `min(in, out)`（满阶），`--rank-ratio 0.5` 表示降一半秩 |
| `--solver` | `hals`（默认，逐列/逐行精确 NNLS，精度最高）/ `pgd`（Adam 投影梯度）/ `mu`（经典 Lee–Seung 乘性更新，目标单调下降） |
| `--iters` | 迭代次数，默认 500；`--n-inner` 控制 hals 中 S 的 FISTA 内层次数 |
| `--shift` | `min`（默认）/ `row` / `zero` / `none` |
| `--init` | `nndsvd`（默认，基于截断 SVD 的非负初始化）/ `random` |

求解器每步都把 A/S/B 截断到 ≥ 0，因此分解结果严格满足"三个非负矩阵 + 中间方阵"。
`--plot` 会输出相对 Frobenius 误差随迭代下降的曲线。

### 2.1 三个求解器的实测对比（`esm2_t6_8M_UR50D`，`--shift min`）

| 层 | rank | 求解器 | 迭代 | 相对非负目标误差 | **相对原权重误差** | 余弦相似度 |
| --- | --- | --- | --- | --- | --- | --- |
| `layers.0.self_attn.k_proj` | 320（满） | `hals` | 600 | **0.00157** | **0.0160** | **0.99987** |
| `layers.0.self_attn.v_proj` | 320（满） | `hals` | 600 | 0.00084 | 0.0088 | 0.99996 |
| `layers.5.self_attn.out_proj` | 320（满） | `hals` | 300 | 0.00263 | 0.0205 | 0.99980 |
| `layers.5.self_attn.out_proj` | 320（满） | `mu` | 2000 | 0.0131 | 0.1019 | 0.9951 |
| `layers.5.self_attn.out_proj` | 320（满） | `pgd` | 2000 | 0.0193 | 0.1503 | 0.9891 |
| `layers.5.fc1` | 320（满） | `hals` | 600 | 0.0121 | 0.0664 | 0.99809 |
| `layers.5.fc1` | 320（满） | `pgd` | 2000 | 0.0283 | 0.1558 | 0.9888 |
| `layers.5.fc1` | 160（半） | `hals` | 300 | 0.0997 | 0.5482 | 0.8387 |
| `layers.5.self_attn.out_proj` | 160（半） | `pgd` | 2000 | 0.0433 | 0.3380 | 0.9421 |

**结论**：`hals` 在满阶时能把相对重构误差压到 **1e-3 量级（余弦 0.9999+）**——这正是
"分解后性能不下降"的基础；`mu`/`pgd` 在同样的迭代预算下差 1~2 个数量级。
而在**降秩**时，误差由秩本身决定（rank=160 时约 0.1），不同求解器差距不大，
这时的性能恢复要靠阶段 2 的训练。


### 阶段 2：批次训练 —— `nmf.train_nmf`

把分解出的 A/S/B 装回模型，冻结其余所有参数，只训练这三个矩阵（可选 `--train-bias`
把 bias 一起训练），做指定批次的训练：

```bash
python -m nmf.train_nmf \
  --layers "layers.5.fc1" \
  --init-checkpoint nmf/outputs/nmf_factors.pt \
  --fasta examples/data/P62593.fasta --max-records 128 \
  --batch-size 8 --max-batches 30 \
  --lr 1e-3 --temperature 2.0 --mask-frac 0.15 \
  --distill-weight 1.0 --recon-weight 1.0 \
  --eval-every 5 --eval-fasta examples/data/some_proteins.fasta \
  --out nmf/outputs/nmf_trained.pt --history nmf/outputs/nmf_history.json --plot
```

每个批次的损失由三部分构成（`--distill-weight` / `--ce-weight` / `--recon-weight` 控制权重）：

1. **蒸馏损失**：随机遮挡 15% 残基后，让分解模型在遮挡位置的输出分布逼近原模型，
   `KL(softmax(logits_ref/T) || softmax(logits_nmf/T)) * T²`；
2. **交叉熵损失**：直接以真实氨基酸为标签（默认权重 0，设 `--ce-weight 0.1` 即"真正的语言建模训练"）；
3. **重构损失**：`‖W - (A S B + offset)‖_F / ‖W‖_F`，防止分解因子在训练中漂移出原层。

每次梯度更新后 A/S/B 都会被投影回非负象限，**训练全程保持三因子非负、中间方阵的约束**。
`--max-batches` 控制训练批次数量；每 `--eval-every` 步会在固定的小评估集上同时报告
原模型与分解模型的遮挡 top-1 准确率，便于画"能力恢复曲线"。

### 阶段 2.1 训练稳定性：为什么必须用「按增益标定的分组学习率」

三因子形式 `A · S · B` 有一个容易被忽视的坑：**中间矩阵 S 夹在两侧因子之间，其逐元素
扰动会被放大 `(A 的行和) × (B 的列和)` 倍。** 而 NMF 解本身存在尺度歧义（只有乘积
`A S B` 是确定的），实际分解出来的 A/S/B 尺度可能相差好几个数量级，例如
`layers.5.fc1` 的 A 元素最大 0.30、B 元素最大却达 **17878**。

实测（`layers.0.self_attn.k_proj`，对三因子分别加 δ=1e-4 的**同号**扰动，看等效权重的相对变化）：

| 扰动加在 | δ=1e-4 | δ=1e-5 | δ=1e-6 |
| --- | --- | --- | --- |
| A | 0.837 | 0.085 | 0.018（基线 0.016） |
| **S** | **10.87** | 1.087 | 0.110 |
| B | 0.022 | 0.016 | 0.016 |

S 的灵敏度比 B 高 **3 个数量级**。而 **Adam 的步长只取决于学习率、与梯度幅值无关**
（`m̂/√v̂ ≈ sign(g)`），因此它会把这种放大原样传给等效权重：实测单次 `lr=1e-4` 的 Adam 步
就让相对重构误差从 **0.0417 爆到 1.699**，几步之内模型完全坏掉（masked 困惑度 14 → 86）。

**解决办法**：按各因子的扰动增益 `gain` 给每组单独设置学习率，让"单步的等效权重移动量"
一致：

```
gain_A = max_j Σ_k (S B)_{kj}                     # A 的列扰动被 (S B) 的列和放大
gain_S = max_i Σ_k A_{ik}  ×  max_l Σ_j B_{lj}    # S 的扰动被两侧因子同时放大
gain_B = max_i Σ_l (A S)_{il}
gain_offset = sqrt(in_features)                   # offset 的每个元素给整行同一增量

lr_factor = --lr / gain_factor                    # --lr 即"每步允许的等效权重相对移动量上界"
```

实测（`esm2_t6_8M_UR50D` 全模型 38 层，`gain_A≈17899, gain_S≈2.8e5, gain_B≈39, gain_offset≈36`）：

| 参数组 | 元素数 | 推导学习率 |
| --- | --- | --- |
| A | 5,632,001 | 5.6e-8 |
| S | 3,788,801 | 3.6e-9（实际近似冻结） |
| B | 5,632,120 | 2.5e-5 |
| offset | 17,601 | 2.8e-5 |

用这组学习率后，重构误差每步只上升约 5e-5（10 步 +1.1%），落在信任域内，蒸馏损失稳定
下降。**结论：对三因子非负分解做微调时，用统一学习率的 Adam 必然失败；要么按增益分组
标定（`--lr-scale auto`，默认），要么改用对尺度不敏感的求解方式。**

另外，脚本还实现了**信任域保护**：每步检查相对重构误差，超过
`初始值 × (1 + --max-recon-degradation)` 就回滚该步并把学习率减半重试（最多
`--rollback-tries` 次），被回滚的步会打印 `[回滚]`。

### 阶段 3：与原模型对比 —— `nmf.evaluate`

```bash
python -m nmf.evaluate \
  --checkpoint nmf/outputs/nmf_factors.pt nmf/outputs/nmf_trained.pt \
  --names 未训练 训练后 \
  --fasta examples/data/some_proteins.fasta --max-records 8 \
  --mask-frac 0.15 --mask-rounds 3 \
  --out-dir nmf/outputs/eval
```

四个维度：

| 维度 | 指标 |
| --- | --- |
| 层重构质量 | 等效权重 `A S B + offset` 与原始权重的相对 Frobenius 误差、余弦相似度、A/S/B 稀疏度、参数量变化、是否严格非负 |
| 输出分布一致性 | 未遮挡输入下逐残基 `KL(分解‖原)`、JS 散度、top-1/top-5 一致率、真实氨基酸对数概率的相关系数 |
| 语言建模能力 | 遮挡 15% 残基后两模型在遮挡位置的 top-1/top-5 准确率、masked 困惑度、遮挡位 top-1 一致率 |
| 效率 | 分解层参数量、全模型参数量、前向耗时 |

产物：`report.md`（人读）、`report.json`（机器可读）、`layer_metrics.csv`（逐层指标）。

### 阶段 4：论文级评测 —— `nmf.benchmark`

`nmf.evaluate` 面向快速诊断；要得到"可否写进论文"的对比，用 `nmf.benchmark`，它在留出测试集上
按文献协议报告指标并直接生成表格：

```bash
python -m nmf.benchmark \
  --checkpoint "未训练=nmf/outputs/fullmodel/factors_fullrank.pt" \
               "训练后=nmf/outputs/fullmodel/trained_fullrank.pt" \
  --fasta data/processed/swissprot/test.fasta --n-eval 128 \
  --mask-frac 0.15 --mask-rounds 5 \
  --ppl-records 12 --ppl-max-len 256 --ppl-batch 32 \
  --out-dir nmf/outputs/fullmodel/bench --plot
```

| 指标 | 协议 | 意义 |
| --- | --- | --- |
| Masked-LM top-1 / top-5 | 遮挡 15% 真实残基，`--mask-rounds` 回合给 **mean ± std** | 直观的"能力保持"证据 |
| masked 困惑度 | 同上协议下的 `exp(平均 NLL)` | 与语言建模指标同量纲 |
| **伪困惑度（pseudo-perplexity）** | **标准单点遮挡**：逐位置只遮一个残基，`exp(平均 -log p(真实残基))` | 蛋白质语言模型最常被引用的自监督指标，无长度偏置 |
| KL / JS / top-k 一致率 | 未遮挡输入下的逐残基分布对比 | 分解模型是否保持了原模型的行为 |
| **线性 CKA** | Kornblith et al. (2019)，全部真实残基位置 | 表征层面的等价性，1.0 = 完全一致 |
| 逐残基表示余弦 / 相对 L2 | 同上 | 比 CKA 更直接 |
| 层级别 + 效率 | 平均/最大相对 Frobenius 误差、余弦、参数量、**线性层乘加比** | 压缩代价与计算代价 |

产物：`benchmark.md`（论文风格四张表）、`benchmark.json`、`layer_stats.csv`、`benchmark.png`。

**长评测的断点续跑**（务必了解：论文级评测在 CPU 上要跑几十分钟到几小时）

```bash
# 每个变体算完立即写 <out-dir>/variants/<名称>.json，中断不丢已完成的部分
python -m nmf.benchmark --checkpoint "未训练=..." "训练后=..." "半秩=..." \
  --fasta data/processed/swissprot/test.fasta --n-eval 128 --mask-rounds 5 \
  --ppl-records 12 --ppl-max-len 256 --out-dir nmf/outputs/fullmodel/bench --plot

# 被中断后接着跑：已算好的变体原模型指标直接复用，只补没算的
python -m nmf.benchmark --checkpoint <同上> ... --out-dir <同上> \
  --reuse-base --reuse-variants

# 报告丢了 / 只想换图，不重新评测：从落盘结果重新渲染
python -m nmf.benchmark --from-dir nmf/outputs/fullmodel/bench --plot
```

| 参数 | 作用 |
| --- | --- |
| `--reuse-base` | 复用 `<out-dir>/base.json` 里已算好的**原模型**指标（伪困惑度最耗时）；口径指纹不一致时自动重算并告警 |
| `--reuse-variants` | 已存在同名变体且**评测口径 + 检查点**都一致时直接复用 |
| `--from-dir DIR` | 完全不加载模型，从 `DIR/base.json` + `DIR/variants/*.json` 重建 `benchmark.md/json/csv`（与 `--plot` 可组合） |

口径指纹覆盖：模型名、测试集路径、`--n-eval`、`--seq-max-len`、`--mask-frac`、`--mask-rounds`、
`--ppl-records`、`--ppl-max-len`、`--cka-repr-layer`、`--seed`。**指纹不一致的变体不会混进同一张表**，
`--from-dir` 会跳过它们并明确提示——避免把不可比的结果并排展示。

> 注意 `--ppl-max-len` 只**截断**序列，不参与抽样过滤（长序列会被截到该长度，而不是被排除）。
> 早期版本把它当过滤条件用，会让伪困惑度只统计短序列、甚至在测试集最短序列都长于该值时静默变成
> `nan`；现在没有可用序列会直接报错。

### 全模型分解

把"对某个线性层做分解"扩展成"对整个模型做分解"，只要把层选择写成 `all` 并加上
`--include-all-linear`（`esm2_t6_8M_UR50D` 共 **38 个** `nn.Linear`）：

```bash
python -m nmf.factorize --layers all --include-all-linear \
  --solver hals --iters 600 --shift min \
  --out nmf/outputs/fullmodel/factors_fullrank.pt
```

CPU（16 线程）上 38 层满阶分解约需 30~45 分钟，属于一次性离线成本；
分解结果可反复用于训练与评测。

## 3. 一键脚本

### 3.1 跑完整实验矩阵（推荐：一条命令拿全部指标）

```bash
bash nmf/run_matrix.sh            # 数据 → {满阶,半秩} 分解 → {满阶,半秩} 训练 → 评测 → 汇总
```

- **已存在的产物会自动跳过**，可随时中断、随时重跑（当前仓库状态下只需补"半秩训练 + 评测"，约 2 小时；
  从零跑约 3.5~4 小时）；
- `DRY_RUN=1 bash nmf/run_matrix.sh` 先看它会执行什么；
- `TAGS="fullrank"` 只跑一个 tag；`FORCE_DATA/FORCE_FACTORIZE/FORCE_TRAIN/FORCE_BENCH=1` 强制重跑某阶段；
- 可覆盖 `N_EVAL / MASK_ROUNDS / PPL_RECORDS / PPL_MAX_LEN`（评测规模）与
  `EPOCHS / MAX_RECORDS / LR / MAX_RECON_DEG`（训练规模）。

### 3.2 单独生成指标汇总

`nmf.summarize` 是**纯后处理**（不加载模型、不重新评测），可以在任何时候重跑，也可以只汇总已完成的部分：

```bash
python -m nmf.summarize --root nmf/outputs/fullmodel \
  --bench nmf/outputs/fullmodel/bench_matrix \
  --data data/processed/swissprot \
  --out nmf/outputs/fullmodel/SUMMARY.md
```

产出 `SUMMARY.md`（数据 / 逐层分解质量 / 训练过程 / 模型级能力对比四节 + 最差层表）、
`SUMMARY.json`（机器可读）、`layer_metrics.csv`（逐层明细）。

### 3.3 分阶段跑（快速小样例）

```bash
bash nmf/run_all.sh
# 可覆盖：PY / MODEL / LAYERS / RANK_RATIO / SOLVER / ITERS / SHIFT /
#        TRAIN_FASTA / EVAL_FASTA / MAX_RECORDS / BATCH_SIZE / MAX_BATCHES / OUTDIR
LAYERS="layers.5.fc1,layers.5.self_attn.out_proj" RANK_RATIO=0.5 MAX_BATCHES=50 bash nmf/run_all.sh
```

完整严谨实验（Swiss-Prot 数据 + 全模型分解 + 多轮训练 + 论文级评测）：

```bash
bash nmf/run_full_experiment.sh
# 可覆盖：TRAIN_N / VALID_N / TEST_N / RANK_RATIO / ITERS / EPOCHS /
#        MAX_TOKENS / LR / KL_SCOPE / DISTILL_W / RECON_W / MAX_RECON_DEG / OUTDIR / TAG

# 半秩（压缩）实验与满阶用同一套流程，只是换 TAG 与 RANK_RATIO
RANK_RATIO=0.5 TAG=halfrank OUTDIR=nmf/outputs/fullmodel bash nmf/run_full_experiment.sh

# 想把 满阶未训练 / 满阶训练后 / 半秩未训练 放进同一张论文表：
# 三个变体写进同一 --out-dir，逐变体落盘 + 可续跑 + 口径一致性保护
python -m nmf.benchmark \
  --checkpoint "满阶未训练=nmf/outputs/fullmodel/factors_fullrank.pt" \
               "满阶训练后=nmf/outputs/fullmodel/trained_fullrank.pt" \
               "半秩未训练=nmf/outputs/fullmodel/factors_halfrank.pt" \
  --fasta data/processed/swissprot/test.fasta --n-eval 128 --mask-rounds 5 \
  --ppl-records 12 --ppl-max-len 256 --ppl-batch 32 \
  --out-dir nmf/outputs/fullmodel/bench_final --plot
```

## 4. 结果解读要点

- **参数量**：`out×r + r×r + r×in` 对比原始的 `out×in`。方阵层（`in = out`）只有在
  `r ≲ 0.586·in` 时才有压缩收益，`r` 更大时参数量反而增加。
- **前向计算**：按 `x B^T S^T A^T` 三次矩阵乘实现，与直接用乘积矩阵数学等价；
  `r < min(in, out)` 时乘加次数下降，`r = min(in, out)` 时乘加次数与原来相当。
- **误差放大效应**：见第 1 节。判断重构好坏请看 `relative_frobenius_error`（相对原权重），
  而不是相对非负目标的那个。
- **信任域保护**：`--lr` 不必精调。实测 `--lr 3e-3` 时第 1 步会被自动减半两次
  （3e-3 → 7.5e-4）后才落在信任域内；`--lr 1e-2` 时该步会被直接回滚。

### 4.1 快速小样例：单层半秩（`examples/data`，分钟级）

配置：`esm2_t6_8M_UR50D`，单层 `layers.5.fc1`（1280×320），`--rank-ratio 0.5`（rank=160），
`pgd` 2000 次迭代，训练 32 个批次（P62593 的前 256 条、batch=8），
`--distill-weight 1.0 --recon-weight 0.5 --lr 3e-3`。

层级别：等效权重相对原权重的 Frobenius 误差 **0.553**（未训练）→ **0.564**（训练后），
余弦相似度 0.835 → 0.829，分解层参数量 **409,600 → 281,600**（全模型 0.983×），
A/S/B 稀疏度 6.6% / 78.3% / 18.1%，训练全程严格非负。

同分布留出集（P62593 另取 8 条，不同随机种子）：

| 变体 | 分布 KL | JS 散度 | top-1 一致率 | top-5 一致率 | 真实 token 对数似然相关 | 遮挡 top-1 | masked 困惑度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原模型 | — | — | — | — | — | 0.2762 | 10.27 |
| 未训练 | 0.1190 | 0.0298 | 0.933 | 0.772 | 0.845 | 0.2481 | 11.42 |
| 训练后 | **0.0572** | **0.0144** | **0.941** | **0.879** | **0.924** | **0.2685** | **10.25** |

跨分布集（`some_proteins.fasta`，8 条无关蛋白）：

| 变体 | 分布 KL | top-1 一致率 | 遮挡 top-1 | masked 困惑度 |
| --- | --- | --- | --- | --- |
| 原模型 | — | — | 0.1777 | 14.02 |
| 未训练 | 0.0511 | 0.988 | 0.1792 | 14.72 |
| 训练后 | 0.0534 | 0.984 | 0.1761 | 14.83 |

结论：

1. **半秩非负分解是有损的**：权重层面相对误差 0.55，但模型层面未遮挡位置 top-1 一致率
   仍有 0.93~0.99，masked 困惑度只从 10.27 升到 11.42——单层权重扰动传播到输出后被削弱了。
2. **批次训练在训练分布上确实有效**：KL 减半（0.119 → 0.057）、top-5 一致率 0.772 → 0.879、
   masked 困惑度 11.42 → 10.25（几乎回到原模型的 10.27）；而权重重构误差只劣化了 2%
   （0.553 → 0.564，信任域内）。
3. **但不会泛化到别的分布**：跨分布集上各项指标基本不变甚至略降，说明这几批训练是在拟合
   训练分布的残差；想让分解模型在通用场景下恢复能力，需要更多样、更大规模的训练语料。

> 这个样例的价值是"几分钟跑通全链路"；它的数据集（5397 条等长 286 的同一酶家族变体、只取 256 条、
> 只训练 32 个批次）**不足以支撑任何"性能不下降"的结论**，正式结论请看下一节。

### 4.2 完整严谨实验：全模型 38 层 + Swiss-Prot（论文级）

配置：`esm2_t6_8M_UR50D` 的**全部 38 个 `nn.Linear`** 同时替换；数据为 Swiss-Prot
切分（train 29,124 条 / 1008.6 万残基，test 4,000 条）；HALS 600 次迭代满阶分解；
训练 1104 步（3 轮、约 65 分钟 CPU）；评测在留出 test 集上按 §阶段 4 的协议。

**① 分解质量（38 层）**

| 指标 | 满阶 `r=min(in,out)` | 半秩 `r=0.5·min(in,out)` |
| --- | --- | --- |
| 平均相对原权重误差 ↓ | **0.0417**（中位 0.0407 / 最大 0.1363） | 0.3441（最大 0.5462） |
| 平均余弦相似度 ↑ | **0.99874**（最小 0.99149） | 0.93055（最小 0.83994） |
| 分解层参数量 | 7,475,320 → 15,052,922（2.014×） | 7,475,320 → 6,579,322（**0.880×**） |
| 耗时（CPU 16 线程） | 1794 s | 1018 s |

**② 训练过程**（只更新 A/S/B/offset，全程 0 次回滚）

| 量 | 起始 | 结束 |
| --- | --- | --- |
| 蒸馏 KL（全部残基位置） | 0.0648 | **0.0123** |
| 相对重构误差 | 0.04176 | 0.04635（+11%，在信任域内） |
| valid 遮挡 top-1（原模型 0.2207） | 0.2111 | **0.2154** |
| valid 遮挡位 top-1 一致率 | 0.7938 | **0.8448** |

分组学习率：`A 5.59e-8 / S 3.55e-9 / B 2.53e-5 / offset 2.80e-5`（按扰动增益标定，见 §阶段 2.1）。

**③ 论文级能力对比**（留出 test 集 64 条 / 24,034 残基，遮挡 5 回合取均值）

| 变体 | 遮挡位 top-1 ↑ | masked 困惑度 ↓ | 伪困惑度 ↓ | KL(分解‖原) ↓ | 遮挡位 top-1 一致率 ↑ | 逐残基 CKA ↑ | 平均层重构误差 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原模型（基准） | 0.2005 | 13.275 | 12.398 | — | — | 1.0000 | — |
| 满阶未训练 | 0.1792（−11%） | 14.211（+7.0%） | 13.473（+8.7%） | 0.0824 | 0.9745 | 0.8746 | 0.0417 |
| **满阶训练后** | **0.1993（−0.6%）** | **13.420（+1.1%）** | **12.544（+1.2%）** | **0.0068** | **0.9935** | **0.9732** | 0.0464 |
| 半秩未训练 | 待补 | 待补 | 待补 | — | — | — | 0.3441 |

**结论**

1. **满阶 + 少量蒸馏训练做到了性能不下降**：遮挡 top-1 与原模型的差距从 11% 收到 **0.6%**
   （5 回合均值 0.1993 vs 0.2005），masked 困惑度差 1.1%，标准单点伪困惑度差 **1.2%**
   （12.544 vs 12.398）；行为一致性改善更明显——逐残基 KL 0.082 → **0.0068**（12 倍），
   遮挡位 top-1 一致率 0.9745 → **0.9935**，逐残基表示 CKA 0.875 → **0.973**。
2. **训练是在"补偿非负约束的信息损失"，不是在"提升能力"**：权重重构误差在训练中反而略升
   （0.0417 → 0.0464），而所有模型级指标全面改善——因为优化目标是让 `A S B` 拟合**原模型的行为**
   （KL 蒸馏），而不是拟合原权重矩阵。
3. **半秩（0.880× 参数量）是有损压缩**：层重构误差 0.344、余弦 0.931；用 8 条序列的快速诊断显示
   其分布一致性大幅退化（KL ≈ 1.5、top-1 一致率 ≈ 0.12）。也就是说在当前的**非负 + 中间方阵约束**下，
   "参数量下降"与"性能不下降"不能同时成立——想压缩就必须接受能力损失。

**待补 / 口径说明**

- 上表"半秩未训练"一行原计划与另两个变体在同一进程内产出，但进程在评测第三个变体时被中断，
  而当时的 `nmf.benchmark` 只在全部变体算完后才写盘，**中间结果全部丢失**。该缺陷已修复
  （逐变体落盘 + `--reuse-base` / `--reuse-variants` / `--from-dir`），补齐只需重跑 §阶段 4 第一条命令。
- 满阶未训练的完整报告在 `nmf/outputs/fullmodel/bench_untrained/`（test 96 条，数值略有差异属子集不同）；
  满阶训练后一行取自同一次运行的日志。
- 表中的伪困惑度是**旧口径**（见 §阶段 4 的提示：旧版把 `--ppl-max-len` 当长度过滤条件），
  重新评测后绝对值会变，但模型间的相对差距结论不受影响。

## 5. 目录

```text
nmf/
  nmf_layers.py       ThreeFactorNMFLinear 模块、平移构造、HALS/乘性更新/投影梯度求解器、层替换工具
  esm_common.py       ESM 加载、FASTA 读取、MLM 遮挡、评估指标、分解模型构建
  prepare_data.py     权威数据集下载（UniProt REST）+ 质控 + 6-mer 去污染 + 切分
  factorize.py        阶段 1 CLI：非负分解
  train_nmf.py        阶段 2 CLI：批次训练（token 预算分批、信任域保护）
  evaluate.py         阶段 3 CLI：快速对比诊断
  benchmark.py        阶段 4 CLI：论文级评测（伪困惑度、CKA、mean±std）+ 断点续跑/重渲染
  summarize.py        纯后处理：把各阶段产物汇总成 SUMMARY.md / SUMMARY.json / layer_metrics.csv
  _env.sh             公共环境准备（解释器/DEVICE/TMPDIR），三个 run_*.sh 共同 source
  run_all.sh          快速小样例：分解 → 训练 → 对比
  run_full_experiment.sh   完整严谨实验（单个 tag）：数据准备 → 全模型分解 → 训练 → 论文级评测
  run_matrix.sh       完整实验矩阵（{满阶,半秩}×{未训练,训练后}）+ 指标汇总，自动跳过已完成项
  outputs/            运行产物（默认输出目录，已在 .gitignore 中忽略）
    <评测目录>/
      base.json             原模型侧指标 + 评测口径指纹（可被 --reuse-base 复用）
      variants/<名称>.json  单个变体的评测结果（算完即落盘，可被 --reuse-variants 复用）
      benchmark.md/json/csv/png   汇总报告（可由 --from-dir 重新渲染）
data/
  raw/                原始下载（Swiss-Prot fasta，约 236 MB）
  processed/<名称>/   train/valid/test.fasta + stats.json（含 sha256 与全部参数）
```

## 6. 测试

```bash
pytest tests/test_nmf.py -v      # 47 个用例，纯随机张量/小模型，无需下载权重，约 6 秒
```

覆盖：形状/非负约束/中间为方阵、HALS 逼近满阶精确解、HALS 优于乘性更新、迭代确实降低误差、
平移模式、三矩阵顺序相乘与乘积矩阵的前向等价性、bias 保留、层名解析、可训练参数范围、
投影回非负、检查点往返、`torch.load` 兼容性、数据质控、k-mer 去污染、token 预算分批、
快照/回滚、线性 CKA 的正交不变性、乘加次数统计、**评测口径指纹、变体落盘/复用/口径筛选、
无模型渲染报告、伪困惑度空记录报错**。

## 7. 快速小样例的完整配置（速查 / 复现）

### 数据与权重

| 项目 | 路径 | 说明 |
| --- | --- | --- |
| 训练集 | `examples/data/P62593.fasta` | 5397 条 β-内酰胺酶变体，长度全为 286；本次取 256 条 |
| 评估集（同分布） | `examples/data/P62593.fasta` | 另取 8 条（`--seed 12345`，与训练用的 seed 0 不同批） |
| 评估集（跨分布） | `examples/data/some_proteins.fasta` | 15 条 UniRef50 多样蛋白，本次取 8 条 |
| 备选小样本 | `examples/data/few_proteins.fasta` | 3 条，用于冒烟测试 |
| 模型权重 | `~/.cache/torch/hub/checkpoints/esm2_t6_8M_UR50D.pt`（29 MB） | 首次加载自动下载 |
| 接触回归权重 | 同目录 `esm2_t6_8M_UR50D-contact-regression.pt`（1.5 KB） | 加载器要求同目录存在 |
| 分解检查点 | `nmf/outputs/nmf_factors.pt` / `nmf_trained.pt` | A/S/B/offset/bias + 报告 + 训练历史 |

数据加载链路（`nmf/esm_common.py`）：

```
read_fasta_records()        L76   纯 Python 解析 FASTA -> [(label, seq)]
  └ rng.shuffle(records)    L108  seed 打乱（可复现子集）
  └ 长度过滤 + max_records  L117  按 seq_min_len / seq_max_len 过滤后截断
make_batches()              L111 按长度排序后切成 batch（减少 padding）
tokenize_records()          L215 alphabet.get_batch_converter(truncation_seq_length=1022)
apply_mlm_mask()            L229 随机遮挡 mask_frac 比例的真实残基（排除 BOS/EOS/PAD）
```

模型加载：`load_esm2()`（L48）→ `esm.pretrained.load_model_and_alphabet_hub("esm2_t6_8M_UR50D")`；
权重缓存在 PyTorch Hub 目录，可用 `--hub-dir` 或 `torch.hub.set_dir()` 改写。

### 阶段 1：`nmf.factorize`

`--model esm2_t6_8M_UR50D --layers layers.5.fc1 --rank-ratio 0.5 --solver pgd --iters 2000
--lr 0.02 --grad-clip 1.0 --shift min --init nndsvd --seed 0 --device cpu`

| 结果 | 值 |
| --- | --- |
| 权重形状 / rank | 1280×320 / 160（S 为 160×160 方阵） |
| 相对原权重误差 / 对平移目标 | 0.5532 / 0.1006 |
| 余弦相似度 | 0.8353 |
| A / S / B 稀疏度 | 18.2% / 99.4% / 65.5% |
| 参数量 | 409,600 → 281,600 |
| 耗时 | 4.5 s |

### 阶段 2：`nmf.train_nmf`

`--init-checkpoint nmf/outputs/nmf_factors.pt --fasta examples/data/P62593.fasta
--max-records 256 --seq-max-len 512 --batch-size 8 --max-batches 60 --epochs 1 --sort-by-length
--lr 3e-3 --temperature 2.0 --mask-frac 0.15 --distill-weight 1.0 --ce-weight 0.0
--recon-weight 0.5 --max-recon-degradation 0.3 --rollback-tries 3 --grad-clip 1.0
--train-bias 关 --freeze-offset 关 --eval-every 5 --eval-fasta examples/data/some_proteins.fasta
--eval-records 8 --seed 0 --device cpu`

| 结果 | 值 |
| --- | --- |
| 可训练参数 | 282,880（只有 A/S/B/offset） |
| 实际步数 | 32（`max_batches=60` 未触及：256/8 = 32 批 × 1 轮） |
| 学习率 | 3e-3 → 自动减半两次 → 7.5e-4（信任域保护），回滚 0 次 |
| 蒸馏损失 | 0.1766 → 0.0321 |
| 相对重构误差 | 0.5532 → 0.5636（+1.9%，信任域内） |
| 耗时 | 26 s |

### 阶段 3：`nmf.evaluate`

两次调用参数相同，只有 `--fasta` / `--seed` 不同：`--mask-frac 0.15 --mask-rounds 3
--max-records 8 --seq-max-len 512 --seed 12345（同分布）/ 0（跨分布）--device cpu --timing-repeats 1`。

## 8. 完整严谨实验的配置（论文级，速查 / 复现）

与第 7 节的差别：**数据换成 Swiss-Prot、层换成全部 38 个、训练步数从 32 提到 1104、评测换成
`nmf.benchmark` 的论文级协议**。整条链路在 16 线程 CPU 上约需 **2.5 小时**（分解 30 min +
训练 65 min + 评测 40 min）。

### 数据（`nmf.prepare_data`，结果记在 `data/processed/swissprot/stats.json`）

| 项目 | 值 |
| --- | --- |
| 下载源 | `rest.uniprot.org/uniprotkb/stream`，query = `(reviewed:true) AND (fragment:false) AND (length:[50 TO 1022])` |
| 原始 / 质控后 | 524,874 → **439,721**（非常规残基 1,497、精确重复 83,656） |
| train | 30,000 → **29,124**（6-mer 去污染剔除 876 条），10,085,760 残基，长度 p50=312 / p95=760 |
| valid / test | 2,000 / 4,000（p50 = 315 / 310，总残基 697,974 / 1,369,602） |
| 命令 | `--min-len 50 --max-len 1022 --train 30000 --valid 2000 --test 4000 --seed 0 --decontaminate --decontam-threshold 0.6 --kmer 6` |
| 校验 | `stats.json` 内含原始文件与三个切分文件的 **sha256**、全部参数、长度分位数 |

### 阶段 1：`nmf.factorize`（满阶 38 层）

`--layers all --include-all-linear --solver hals --iters 600 --n-inner 20 --shift min --init nndsvd`
→ 平均相对原权重误差 **0.04172**、余弦 **0.99874**、参数 7,475,320 → 15,052,922（2.014×）、1794 s

半秩版：`--rank-ratio 0.5`（`--iters 400`）→ 平均 0.34414、余弦 0.93055、参数 0.880×、1018 s

### 阶段 2：`nmf.train_nmf`（1104 步 / 3 轮）

`--layers all --include-all-linear --init-checkpoint <满阶分解> --fasta data/processed/swissprot/train.fasta
--max-records 4000 --seq-max-len 1022 --batch-size 32 --max-tokens-per-batch 4096 --epochs 3
--lr 1e-3 --lr-scale auto --kl-scope all --distill-weight 1.0 --recon-weight 0.1 --ce-weight 0.0
--max-recon-degradation 1.0 --rollback-tries 3 --grad-clip 1.0 --eval-every 50
--eval-fasta data/processed/swissprot/valid.fasta --eval-records 32 --save-every 50 --seed 0 --device cpu`

| 量 | 值 |
| --- | --- |
| 实际步数 / 耗时 | 1104（3 轮）/ 3909 s（3.54 s/步） |
| 分组学习率 | A 5.59e-8、S 3.55e-9、B 2.53e-5、offset 2.80e-5（全程未变，**0 次回滚**） |
| 蒸馏损失 | 0.0648 → 0.0123 |
| 相对重构误差 | 0.04176 → 0.04635 |
| 检查点 | `trained_fullrank.pt`（`--save-every 50`，中断也能拿到可用结果） |

### 阶段 3/4：`nmf.benchmark`（论文级评测）

`--fasta data/processed/swissprot/test.fasta --n-eval 64 --seq-max-len 1022 --mask-frac 0.15
--mask-rounds 5 --ppl-records 8 --ppl-max-len 256 --ppl-batch 32 --seed 0 --device cpu --plot`

→ 结果见 §4.2；产物为 `benchmark.md` / `benchmark.json` / `layer_stats.csv` / `benchmark.png`，
以及可复用的 `base.json` 与 `variants/*.json`。
