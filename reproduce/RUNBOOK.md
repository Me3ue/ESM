# 运行手册（环境无关版）

本文档假设你在**一台正常的 GPU 机器**上，只讲「怎么跑」和「论文本身用了多少机器」。
本机环境问题（缺依赖、CPU 版 torch 等）一概不在此讨论。

配套工具包：`reproduce/`。所有命令都以仓库根 `/path/to/esm` 为起点。

目录：

| 文档 | 内容 |
|:--|:--|
| **本文（RUNBOOK.md）** | 环境安装 → 逐实验命令 → 论文的设备与规模 → 算力推算 → 分期排期 |
| [`DATASETS.md`](DATASETS.md) | 数据集：从哪下、多大、放哪、怎么组织、怎么校验、踩坑清单 |
| [`README.md`](README.md) | 工具包设计、与论文不一致之处、本机排错 |
| [`outputs/p2/paper_data_report/PAPER_DATA_REPORT.md`](outputs/p2/paper_data_report/PAPER_DATA_REPORT.md) | 论文二公开实验数据的逐项复算（7/8 项精确命中） |

---

## 1. 标准环境

### 1.1 推荐配置

| 项目 | 建议 |
|:--|:--|
| GPU | ≥1 张 24–32 GB 显存的 NVIDIA 卡（论文二用的是 **32GB Volta**）。复现论文一的多聚体/长链设计建议 ≥40 GB |
| CUDA | 与驱动匹配的 cu121 / cu124 |
| Python | **两个环境**：论文二用 3.10；论文一（ESMFold/OpenFold）建议单独建 **3.9** 环境 |
| 磁盘 | 起步 **150 GB**；要复现序列新颖性给 **300 GB**；要 TM-score 穷举或重训投影层给 **1 TB**。明细见 [`DATASETS.md` §4](DATASETS.md) |
| 外部工具 | `jackhmmer` 3.3.2、`TM-align` ≥20210107、`Foldseek`、`Rosetta`（过滤指标）、`aria2/wget` |

**数据获取与组织**：见 **[`DATASETS.md`](DATASETS.md)**——每份数据从哪下、多大、怎么放、怎么校验、
有哪些坑。一键入口：

```bash
bash reproduce/fetch_datasets.sh plan     # 先看清单（不下载）
bash reproduce/fetch_datasets.sh core     # 必需部分：PDB + 权重 + 论文数据包（≈ 8.8 GB）
bash reproduce/fetch_datasets.sh seqdb    # 可选：UniRef90（论文二序列新颖性，≈ 32 GB）
bash reproduce/fetch_datasets.sh verify   # 校验已下载文件（存在性 + 体积 + sha256）
```

### 1.2 一次性安装

```bash
# ---------- 环境 A：论文二（lm-design） ----------
conda create -y -n esm python=3.10 pip
conda activate esm
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd /path/to/esm && pip install -e .
pip install -r reproduce/requirements-repro.txt     # biotite/rich/hydra/omegaconf/nltk/pandas/scipy
conda install -y -c bioconda hmmer=3.3.2            # jackhmmer
conda install -y -c conda-forge aria2 wget

# ---------- 环境 B：论文一（ESMFold / OpenFold，需 Python 3.9） ----------
conda create -y -n esmfold python=3.9
conda activate esmfold
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd /path/to/esm && pip install -e .
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install biotite rich

# ---------- 验证 ----------
python -c "import torch,esm; m=esm.pretrained.esmfold_v1().eval().cuda(); \
print(m.infer_pdb('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ')[:60])"
```

### 1.3 自检

```bash
bash reproduce/00_setup_env.sh check
# 或分别检查两个环境
$PY_ESMFOLD reproduce/paper1_programming/design_programs.py --check
$PY_ESM      reproduce/paper2_lm_design/run_lm_design_batch.py --check
```

> 论文一的一切命令都必须在**装了 openfold 的环境**里执行（`PY` 指 `esmfold` 环境）。
> 论文二、以及论文二的公开数据复算，在普通环境即可。

---

## 2. 论文一：逐实验命令

论文一没有 CLI，全部通过 `design_programs.py` 驱动（`--task` 选实验，`--grid paper` 用论文规模）。

```bash
export PY=/path/to/envs/esmfold/bin/python
export OUT=/data/repro/p1
cd /path/to/esm/reproduce/paper1_programming

# 先看要提交什么
$PY design_programs.py --list-tasks
$PY design_programs.py --task symmetric_monomer --grid paper --dry-run
```

### 2.1 九类设计实验（对应正文图 2~图 5）

```bash
# 图 2A-C  自由幻觉：200 seeds × 30,000 步
$PY design_programs.py --task free_hallucination --grid paper --length 100 \
    --device cuda:0 --out-dir $OUT

# 图 2D-F  固定骨架设计：6 个 de novo target（1QYS/5L33/6D0T/6MRS/6W3W/6WVS）× 50 seeds
$PY design_programs.py --task fixed_backbone --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 2G/S1  二级结构设计：3 程序（全α/全β/混合）× 10 seeds
$PY design_programs.py --task secondary_structure --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 2H/S1D 单功能位点脚手架：5 位点 × 1,000 seeds = 5,000 条跑动  ← 最贵
$PY design_programs.py --task functional_site_scaffolding --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 3A-D/S2 单链对称：K=3..8 × 总长 200/300/400 × 10 seeds = 180 条
$PY design_programs.py --task symmetric_monomer --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 3E  同源寡聚体：4/6/8 聚体 × 10 seeds（总长 720）
$PY design_programs.py --task homo_oligomer --grid paper --total-length 720 \
    --device cuda:0 --out-dir $OUT

# 图 4A-D/S3 双层对称：顶层 2~4 × 底层 2~4 = 9 程序 × 10 seeds = 90 条
$PY design_programs.py --task two_level_multimer --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 5A-B/S4A-C 对称功能位点脚手架：3 程序 × 20 seeds = 60 条
$PY design_programs.py --task symmetric_binding --grid paper \
    --device cuda:0 --out-dir $OUT

# 图 5C-F/S4D-E 非对称层级组合：3 程序 × 10 seeds = 30 条
$PY design_programs.py --task hierarchical_asymmetric --grid paper \
    --device cuda:0 --out-dir $OUT
```

> 多卡并行：脚本按 `task/spec/seed` 分目录、自动跳过已完成项，所以可以直接
> **开会话多开**，把 `--task` 或 `--seeds` 切片丢给不同 GPU：
> ```bash
> CUDA_VISIBLE_DEVICES=0 $PY design_programs.py --task functional_site_scaffolding \
>     --seeds 0-249 --out-dir $OUT &
> CUDA_VISIBLE_DEVICES=1 $PY design_programs.py --task functional_site_scaffolding \
>     --seeds 250-499 --out-dir $OUT &
> ```

### 2.2 逆折叠 roundtrip（图 3C-D / 4C-D / S2D）

```bash
# 1,000 个结构 × 10 条 ESM-IF1 采样（温度 0.1）+ 10 次 ESMFold 回折
$PY roundtrip.py --pdb-root $OUT \
    --tasks symmetric_monomer two_level_multimer \
    --n-structures 1000 --num-samples 10 --temperature 0.1 \
    --device cuda:0 --out $OUT/roundtrip

# 只算 ESM-IF1 perplexity（省一半算力）
$PY roundtrip.py --pdb-root $OUT --tasks symmetric_monomer \
    --n-structures 1000 --num-samples 10 --no-esmfold --out $OUT/roundtrip
```

图 S2D 用的是 **ProteinMPNN** 而非 ESM-IF1，需另装 ProteinMPNN 仓库后同法替换采样器。

### 2.3 结构新颖性 TM-score（图 S2C / S3C）

```bash
# 需要 PDB 2022-08 快照（约 19 万个结构）与 tm-align
wget -r -np -nH --cut-dirs=1 https://files.rcsb.org/pub/pdb/data/structures/divided/pdb/
# 对每个设计结构穷举 TM-align（论文用的是 TM-align 20210107，按设计长度归一化）
for pdb in $OUT/symmetric_monomer/*/seed*/design.pdb; do
  tm-align -byresi "$pdb" /data/pdb_all/*.pdb > "${pdb%.pdb}.tmalign.txt"
done
# 取 TM-score 最大值即为「与最近天然结构的相似度」
```

### 2.4 汇总

```bash
$PY aggregate_p1.py --root $OUT
# → $OUT/summary/P1_SUMMARY.md    对照论文图 2B/2E/2F/2H/3B/4B/5B 的结论表
#   $OUT/summary/p1_runs.csv     逐跑动明细
#   $OUT/summary/p1_by_spec.csv  按 task×spec 聚合
#   $OUT/summary/fig_*.png       箱线图 / 直方图 / roundtrip 散点
```

### 2.5 只跑一小段（先验证链路）

```bash
$PY design_programs.py --task symmetric_monomer --grid smoke --out-dir /tmp/p1_smoke
# smoke = 每 spec 2 seed × 200 步；跑通后再换 --grid paper
$PY design_programs.py --task free_hallucination --grid paper --seeds 8 --steps 5000 --out-dir $OUT
```

---

## 3. 论文二：逐实验命令

```bash
export PY=/path/to/envs/esm/bin/python
export OUT=/data/repro/p2
cd /path/to/esm/reproduce/paper2_lm_design
```

### 3.1 准备 target

```bash
# 统一入口（推荐）：PDB + 权重 + 论文数据包一次到位，见 DATASETS.md
bash reproduce/fetch_datasets.sh core
bash reproduce/fetch_datasets.sh processing    # 清洗成单链纯蛋白，并生成位点定义 JSON

# 也可以只下 target（会下 37 个唯一 PDB 文件）
$PY run_lm_design_batch.py --fetch-pdbs --pdb-dir $OUT/de_novo_targets
# 仓库自带的样例 target 在 examples/lm-design/2N2U.pdb
```

> 论文 App A.1.1 列的是 **39 个** target，但 `6DKM A/B`、`6DLM A/B` 共用 2 个 PDB 文件，
> 所以**去重后只有 37 个 PDB**（长度 67 ≤ L ≤ 184）。清洗后的单链版本在
> `data/processed/lm_design_targets/`，多链或含配体的原始 entry 会让 `pdb_loader` 出错。

### 3.2 固定骨架设计（图 2A-F）

```bash
# 论文规模：39 个 target × 200 designs，每条 170,000 步 MCMC
for T in 1QYS 2KL8 2KPO 2LN3 2LTA 2LVB 2N2T 2N2U 2N3Z 2N76 \
         4KY3 4KYZ 5CW9 5KPE 5KPH 5L33 5TPJ 5TRV 6CZG 6CZH \
         6CZI 6CZJ 6D0T 6DG6 6DKM 6DLM 6E5C 6LLQ 6MRR 6MRS \
         6MSP 6NUK 6W3F 6W3W 6WI5 6WVS 7MCD; do
  CUDA_VISIBLE_DEVICES=0 $PY run_lm_design_batch.py --task fixedbb --pdb $T \
      --pdb-dir $OUT/de_novo_targets --seeds 0-199 --num-iter 170000 \
      --oracle esmfold --device cuda:0 --out-dir $OUT
done
```

### 3.3 自由生成（图 3 / 图 4A-B）

```bash
# 论文规模：10k（Round 1 前）+ 15k（Round 2）= 25,000 条，定长 L=100，无结构约束
$PY run_lm_design_batch.py --task free_generation --length 100 \
    --seeds 0-9999 --num-iter 170000 --device cuda:0 --out-dir $OUT      # 第一批 10k
$PY run_lm_design_batch.py --task free_generation --length 100 \
    --seeds 10000-24999 --num-iter 170000 --device cuda:0 --out-dir $OUT # 第二批 15k
```

### 3.4 序列新颖性（图 2G / 4F-G）

```bash
# 库里版本选择见 DATASETS.md §3.4：
#   当前版 UniRef90  → 32 GB（推荐）
#   论文原版 2021_04 → 158 GB（官方打包含 UniRef50/90/100）
bash reproduce/fetch_datasets.sh seqdb
gunzip -k data/raw/uniref90/uniref90.fasta.gz

$PY analyze_novelty.py --fasta-dir $OUT \
    --db data/raw/uniref90/uniref90.fasta --db-name uniref90_current \
    --purge ../../examples/lm-design/paper-data/uniref90_jackhmmer_purge_ids.txt \
            ../../examples/lm-design/paper-data/artificial_sequence_purge_ids.txt \
    --cpu 64 --out $OUT/novelty
```

脚本已固化论文的三个非默认设置：`-n 1`、`--seed 0`、按 **best-domain E-value** 排序，
并默认读入两个 purge 列表（人工序列 1,027 条 + de novo 反查命中 58,462 条，App A.1.2）。
注意这两个列表是按 2021_04 编的，**换成当前版库后会略有错位**，报告里要注明库名。

### 3.5 结构投影层训练（App A.2.2，通常不需要）

论文这一环**没有开源训练脚本**，仓库只发布**已训练权重**（0.5 MB，`lm_design` 运行时会自动下载），
所以**默认不需要下载任何训练数据**。只有要自己重训时才需要：Yang et al. 2020 的非冗余 PDB 集
（**15,051** 条，结构发布日期早于 **2018-05-01**，取 80% 作训练集），冻结 ESM2 权重，
训练 **10 个 epoch、batch size 4、lr 1e-2**，损失为预测 distogram 与真值 distogram 的
categorical cross-entropy。重建路线见 [`DATASETS.md` §3.7](DATASETS.md)。

### 3.6 汇总

```bash
$PY aggregate_p2.py --root $OUT
# → $OUT/summary/P2_SUMMARY.md
#   $OUT/summary/p2_runs.csv / p2_by_config.csv
#   $OUT/summary/fig_energy_trajectory.png（图 2D 同款）
#   $OUT/summary/fig_lm_perplexity.png   （图 2E 同款）
#   $OUT/summary/fig_oracle_rmsd.png     （图 2A 同款）
```

### 3.7 零成本：论文公开实验数据复算

```bash
$PY paper_data_report.py
# 不加载任何模型，直接用 examples/lm-design/paper-data/data.csv 复算论文二数字
```

### 3.8 论文中需要外部资源的两块

```bash
# ① no-LM 基线（图 2C/2F）：ColabDesign，commit e7bb3def，3stage() AfDesign
#    1500 soft iters + 500 temp iters + 50 hard iters，跨 5 个 AlphaFold pTM replica 优化
git clone https://github.com/sokrypton/ColabDesign && cd ColabDesign && git checkout e7bb3def
#    产出的 designs 导出成 CSV 后并入 aggregate_p2.py

# ② AlphaFold oracle（图 2A/3C）：5 个公开模型选 pLDDT 最高 + Amber 松弛
colabfold_batch designs.fasta out_dir --num-recycle 1 --model-type alphafold2_ptm
#    注意论文用的是单序列（无 MSA、无 template）模式
```

---

## 4. 论文中的设备与规模

### 4.1 论文明确写出的硬件

| 论文 | 原文 | 出处 |
|:--|:--|:--|
| **论文二** | “Full design trajectories take **≈ 10 hours** for a fixed backbone design with sequence length ≈ 100 on a **single 32GB Volta gpu**.” | Methods A.3.1 |
| **论文二** | 部分命中序列因为“**GPU memory limitations**”（长度 > 1000）导致 oracle 结构预测失败 | 附录 A.5.3 |
| **论文一** | **论文全文没有给出硬件或总算力**（已逐页检索 `GPU/CPU/Volta/hour/compute` 等词） | — |

> 论文二原文里的“32GB Volta”即 **NVIDIA V100 32GB**。也就是说：**单卡、L≈100、17 万步 ≈ 10 小时**。
> 论文一没有声明硬件；其依赖的 ESMFold 单次前向在 V100 量级的卡上约 0.1–0.3 秒（L≈100），
> 可据此推算（见 4.3）。

### 4.2 论文提交的实验规模（全部来自 Methods，逐条核对）

#### 论文一

| 图 | 实验 | 步数/条 | 条数 | 出处 |
|:--|:--|--:|--:|:--|
| 2A-C | 自由幻觉 | 30,000 | **200** seeds | A.3.1 |
| 2D-F | 固定骨架设计（6 个 de novo target） | 30,000 | **≥300**（每 target ≥50 seeds） | A.3.2 |
| 2G / S1 | 二级结构设计（全α/全β/混合） | 30,000 | **30**（3 程序 × 10） | A.3.3 |
| 2H / S1D | 单功能位点脚手架（5 个天然位点） | 30,000 | **5,000**（5 × 1,000） | A.3.4 |
| 3A-D / S2 | 单链对称（K=3–8 × 长度 200/300/400） | 30,000 | **180**（6 × 3 × 10） | A.3.5 |
| 3E | 同源寡聚体（4/6/8 聚体） | 30,000 | **40** | A.3.5 |
| 4A-D / S3 | 双层对称（顶层 2–4 × 底层 2–4） | 30,000 | **90**（9 × 10） | A.3.6 |
| 5A-B / S4A-C | 对称功能位点脚手架 | 30,000 | **60**（3 × 20） | A.3.9 |
| 5C-F / S4D-E | 非对称层级组合 | 30,000 | **30**（3 × 10） | A.3.10 |
| 3C-D / 4C-D | 逆折叠 roundtrip | — | **1,000 + 1,000** 个结构 × 10 条 IF1 采样（T=0.1） | A.3.8 |
| S2C / S3C | 结构新颖性 TM-score | — | 全部 seed 对 **PDB 2022-08 全库**穷举 TM-align | A.3.7 |
| 2C | ssAF2 对照 | — | 对 200 条最终序列跑单序列 AlphaFold2 | A.3.1 |

**设计类跑动合计 ≈ 5,930 条**，每条 30,000 步 ⇒ **≈ 1.78 亿次 ESMFold 前向**。
（外加 roundtrip 的 20,000 次 ESMFold、200 次 ssAF2、以及 ~270 个设计 × PDB 全库的 TM-align。）

#### 论文二

| 图 / 附录 | 实验 | 步数/条 | 条数 | 出处 |
|:--|:--|--:|--:|:--|
| 2A | 固定骨架设计（39 个 de novo target） | 170,000 | **7,800**（39 × 200） | A.3.1 / A.6.2 |
| — | 扩充设计池（1QYS 1990、6MRS 1500、6D0T 1604、6W3W 1968） | 170,000 | **9,060**（原文如此） | A.6.2 |
| 2C / 2F | LM vs no-LM 对照（4 个 target，各 200 designs） | 170,000 / 1500 soft iters | **800 + 800** | A.3.2 |
| 3 / 4A-B | 自由生成（L=100，无结构约束） | 170,000 | **25,000**（10k + 15k） | A.3.3 / A.6.3 |
| 4A | 结构聚类（TM-score 0.75 阈值） | — | 25,000 条 → **7,663 个簇** | 正文 |
| 4B | 二级结构构成 | — | 52% 以 α 为主 / 22% 以 β 为主 / 28% 混合 | 正文 |
| 4D | 与 AlphaFold DB 全库对照 | — | 25,000 generations + ≈15k 天然蛋白 | A.5.2 |
| 4D | 远距比例 | — | **15.5%** 落在 Seq-id<0.2 且 TM-score<0.5 象限；16.6% 无显著命中 | 正文 |
| S1/S2/表S1 | 结构理解基线（天然蛋白对照） | — | **214** 条天然蛋白 | A.1.1 |
| A.2.2 | 结构投影层训练 | 10 epoch | **15,051** 条 PDB（取 80%），batch 4，lr 1e-2，冻结 ESM2 | A.1.3 / A.2.2 |
| A.5 | jackhmmer 检索 | — | 228×10 = **2,280** 次命中分析；库为 **UniRef90 2021_04（>1 亿序列）** | A.5.3 |
| A.5.4 | motif 分析 | — | Foldseek v7d0c07f89a，`--alignment-type 1`，库为 AlphaFold DB | A.5.4 |
| A.4 | 结构 oracle | — | **AlphaFold 全部 5 个公开模型**，按 pLDDT 选最优，再做 Amber 松弛 | A.4.1 |
| A.6 / A.7 | 湿实验验证 | — | **276** 条蛋白，两轮：Round1 = 44 固定 + 48 生成 + 4 GT；Round2 = 95 固定 + 81 生成 + 4 GT | A.6.1 / A.7 |
| A.7 | SEC 分析 | — | Superdex **S75 5/150** 柱 | A.7 |

**设计类跑动合计 ≈ 42,660 条**（7,800 + 9,060 + 25,000 + 1,600），每条 170,000 步。
25,000 条自由生成额外需要 AlphaFold oracle 折叠、以及 vs AlphaFold DB 的 TM-align 检索。

### 4.3 总算力推算

论文都**没有给出总算力**，下表是按论文自身给出的单价 + ESMFold/AlphaFold 的公开推理成本推的
**量级估算**（仅用于排期，不要当精确值）：

| 论文 | 单价依据 | 条数 | 估算 GPU 小时 | 折合 |
|:--|:--|--:|--:|--:|
| 论文二 · 固定骨架 | 论文自己给的 **10 h/条**（32GB Volta，L≈100） | 7,800 | **≈ 78,000** | ≈ 9 GPU·年 |
| 论文二 · 扩充设计池 | 同上 | 9,060 | **≈ 90,000** | ≈ 10 GPU·年 |
| 论文二 · 自由生成 | 每步含 1 次 p(y\|x) + 3 次 p(x\|y)，与固定骨架同量级 | 25,000 | **≈ 250,000** | ≈ 29 GPU·年 |
| 论文二 · 小计 | | ≈42,660 | **≈ 4.2 × 10⁵** | **≈ 48 GPU·年** |
| 论文一 · 全部设计 | ESMFold ≈0.2 s/前向（L≈100，V100 量级），每条 30,000 步 | 5,930 | **≈ 10,000** | ≈ 1.1 GPU·年 |
| 论文一 · roundtrip + ssAF2 + TM-align | 20,000 次 ESMFold + 200 次 AF2 + ~270 × PDB 全库 | — | **≈ 1,000+** | — |

**结论：论文一是「千卡时」量级，论文二是「十万卡时」量级，两者差 1~2 个数量级。**
所以复现策略要分开：

- **论文一**：目标可以是「完整复现」。按单张 A100/V100，全部跑完约 **1.5 个月**；
  用 8 卡并行约 **5 天**。
- **论文二**：全量复现在个人算力下**不可行**。可行的是：
  1. 跑 `paper_data_report.py`（0 成本）核对论文实验数字；
  2. 单 target × 20~200 designs × 170,000 步，复现图 2A/2D/2E 的**趋势**（单 target ≈ 80~800 GPU 小时）；
  3. 自由生成取 100~1,000 条复现图 3/4A-B 的**分布形状**（≈ 1,000~10,000 GPU 小时）；
  4. 新颖性检索用小库（Swiss-Prot）打通流程，再决定是否上 UniRef90。

### 4.4 分期排期建议

| 期 | 目标 | 规模 | 估算 |
|:--|:--|:--|--:|
| P0 | 链路验证 | 所有任务 `--grid smoke`（2 seed × 200 步） | <1 GPU 小时 |
| P1 | 论文一主力图 | 自由幻觉 200 seeds + 单链对称 180 + 双层对称 90 + 二级结构 30 + 寡聚体 40 + 层级 30 ≈ 570 条 | ≈ 1,000 GPU 小时 |
| P2 | 论文一最难的一块 | 功能位点脚手架 5,000 条 | ≈ 8,300 GPU 小时 |
| P3 | 论文一收尾 | 对称位点脚手架 60 + 固定骨架 300 + roundtrip | ≈ 1,500 GPU 小时 |
| P4 | 论文二抽样 | 1~2 个 target × 50~200 designs，L=100 自由生成 200 条 | ≈ 1,000 GPU 小时 |
| P5 | 论文二收尾（可选） | 新颖性检索 + AF oracle + Rosetta 过滤 | 视库规模 |

---

## 5. 一键入口（与上面命令等价）

```bash
# 数据集：先跑这个
bash reproduce/fetch_datasets.sh core          # 必需数据 ≈ 8.8 GB（PDB + 权重 + 论文数据包）
bash reproduce/fetch_datasets.sh verify        # 校验

# 论文一：设计 + roundtrip + 汇总
PY=/path/to/envs/esmfold/bin/python MODE=full OUT_ROOT=/data/repro \
  WITH_ROUNDTRIP=1 bash reproduce/paper1_programming/run_p1.sh

# 论文二：设计 + 新颖性 + 汇总
PY=/path/to/envs/esm/bin/python MODE=full OUT_ROOT=/data/repro \
  ALL_TARGETS=1 NUM_ITER=170000 ORACLE=esmfold \
  NOVELTY_DB=/data/uniref90.fasta bash reproduce/paper2_lm_design/run_p2.sh

# 两篇一起
MODE=full OUT_ROOT=/data/repro bash reproduce/run_all.sh
# 只做论文公开数据复算（秒级，不需要 GPU）
ONLY=paperdata bash reproduce/run_all.sh
```

所有脚本都保证：**已完成的最小单位自动跳过**，中断后重跑同一命令即可续跑；
`DRY_RUN=1` 只打印命令。
