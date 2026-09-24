# ESM 项目使用、结构与数据集指南

本仓库是 Meta FAIR 发布的 **ESM（Evolutionary Scale Modeling）** Python/PyTorch 实现，提供预训练蛋白质语言模型、结构预测、逆折叠以及若干研究示例。它主要用于推理和下游研究，不包含完整的预训练训练流水线或所有大规模原始数据。当前仓库版本为 `2.0.1`（见 `esm/version.py` / `fair_esm.egg-info`）。

> 本指南在原始项目介绍的基础上做了三处补充：补全目录树、模型清单与常见问题（第 2~4、9、10 节），并新增第 11 节——**线性层的三因子非负矩阵分解（NMF）实验**，包含可运行的代码、脚本与评估方法。

## 1. 能力概览

- **ESM-2**：从单条蛋白质序列提取残基级/序列级表示、logits 和接触图。
- **ESMFold**：从单条序列直接预测三维结构并输出 PDB。
- **ESM-MSA-1 / ESM-MSA-1b**：处理多序列比对（MSA）的 Transformer。
- **ESM-1 / ESM-1b**：最早的通用蛋白语言模型（650M / 670M 等）。
- **ESM-1v**：零样本预测突变对蛋白功能的影响。
- **ESM-IF1**：以骨架坐标为条件采样或评分蛋白质序列（逆折叠）。
- **研究示例**：接触预测、监督变体预测、LM 蛋白设计、蛋白质编程语言、Atlas 下载等。

预训练权重不随仓库提交。通过 `esm.pretrained` 加载模型时，权重会下载并缓存到 PyTorch Hub 缓存目录（通常为 `~/.cache/torch/hub/checkpoints`）。

## 2. 安装和环境

先安装与设备/CUDA 匹配的 PyTorch，然后安装本项目：

```bash
pip install fair-esm
# 或在当前仓库根目录安装
pip install .
```

`setup.py` 注册两个命令行工具：

- `esm-extract`：从 FASTA 批量提取表示；
- `esm-fold`：从 FASTA 批量预测 ESMFold 结构。

### 2.1 本机已配置的 conda 环境 `ESM`

本机已创建并配置好名为 **`ESM`** 的 conda 环境（Python 3.10 + PyTorch CPU 版 + 本仓库可编辑安装 + matplotlib），可直接使用：

```bash
conda activate ESM
/home/zzj/anaconda3/envs/ESM/bin/python -c "import torch, esm; print(torch.__version__)"
```

从零重建（`conda` 位于 `/home/zzj/miniconda3` 或 `/home/zzj/anaconda3`）：

```bash
conda create -y -n ESM python=3.10 pip
conda run -n ESM pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版
conda run -n ESM pip install -e .                                                     # 安装本仓库
conda run -n ESM pip install matplotlib                                               # 绘图（可选）
```

注意：

- 若要用 GPU，把 torch 换成与驱动匹配的 CUDA 版本（`pip install torch` 走 PyPI 默认即带 CUDA）；
  本仓库脚本通过 `--device auto` 自动选择设备，CPU 环境下自动回退。
- 本机 `/tmp` 是 10 MB 的 tmpfs，`pip` 下载大轮子会报 `No space left on device`，
  需先 `export TMPDIR=/home/zzj/tmp`（`/home` 所在分区剩余空间充足）。
- 首次加载模型会从 `dl.fbaipublicfiles.com` 下载权重并缓存到
  `~/.cache/torch/hub/checkpoints`；`esm2_t6_8M_UR50D` 约 30 MB。

### ESMFold 额外依赖

ESMFold 依赖 OpenFold，建议使用 Python 3.9 或更早版本、已安装 CUDA PyTorch 的环境，且 OpenFold 安装通常需要 `nvcc`：

```bash
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

`environment.yml` 是仓库提供的历史 ESMFold Conda 环境参考。逆折叠、LM 设计等示例还有各自额外依赖，应按对应子目录 README 安装。

## 3. 核心目录结构

```text
esm/
  __init__.py                 导出 data / model / pretrained 等子模块
  version.py                  版本号（2.0.1）
  constants.py                蛋白质数据集常量
  data.py                     FASTA/MSA 读取、Alphabet、BatchConverter、token 化、
                              ESMStructuralSplitDataset（结构分割数据集）
  pretrained.py               权重下载、检查点适配、模型加载工厂
  model/
    esm1.py                   ESM-1、ESM-1b 模型
    esm2.py                   ESM-2 模型
    msa_transformer.py        MSA Transformer
  inverse_folding/            ESM-IF1 的几何编码（GVP）、解码及多链工具
  esmfold/v1/                 ESMFold 推理、trunk、IPA、结构模块
  modules.py                  TransformerLayer、LayerNorm、接触预测头、LM head 等
  multihead_attention.py      多头注意力
  axial_attention.py          轴向注意力（MSA Transformer 用）
  rotary_embedding.py         旋转位置编码
scripts/
  __init__.py
  extract.py                  esm-extract 的源码入口
  fold.py                     esm-fold 的源码入口
  download_weights.sh         批量下载 ESM-IF1 等模型权重的脚本
  atlas/                      ESM Atlas 下载清单与说明
examples/
  data/                       FASTA、A3M 等小型输入样例
  inverse_folding/            ESM-IF1 脚本、PDB/mmCIF、Notebook
  variant-prediction/         ESM-1v 变体预测脚本和 DMS 数据
  lm-design/                  固定骨架/自由生成设计示例（含 conf/、paper-data/）
  protein-programming-language/  可编程蛋白质设计示例（language/、programs/）
  contact_prediction.ipynb              ESM-2 / MSA Transformer 接触预测
  sup_variant_prediction.ipynb          用 ESM 表示训练监督变体分类器
  esm_structural_dataset.ipynb          结构分割数据集使用示例
  esm2_infer_fairscale_fsdp_cpu_offloading.py   15B 模型 FSDP + CPU offload 推理
nmf/                          线性层三因子非负矩阵分解实验（见第 11 节，本仓库新增）
  nmf_layers.py               ThreeFactorNMFLinear、HALS/乘性更新/投影梯度求解器、层替换工具
  esm_common.py               ESM 加载、FASTA、MLM 遮挡、评估指标、分解模型构建
  prepare_data.py             权威数据集下载（UniProt REST）+ 质控 + 去污染 + 切分
  factorize.py / train_nmf.py / evaluate.py / benchmark.py
                              非负分解 / 批次训练 / 快速对比 / 论文级评测
  summarize.py                纯后处理：把各阶段产物汇总成 SUMMARY.md / .json / layer_metrics.csv
  run_all.sh                  快速小样例（单层 + 自带小数据）
  run_full_experiment.sh      完整严谨实验（单个 tag：Swiss-Prot + 全模型 38 层 + 论文级评测）
  run_matrix.sh               ★ 完整实验矩阵 {满阶,半秩}×{未训练,训练后} + 指标汇总（自动跳过已完成项）
  README.md                   该实验的详细说明
data/                         实验数据（本仓库新增）
  raw/                        原始下载（Swiss-Prot reviewed fasta，约 236 MB）
  processed/<名称>/           train/valid/test.fasta 与 stats.json（含 sha256）
tests/                        模型、字母表、逆折叠、README/Notebook 一致性测试，
                              test_nmf.py 为非负分解实验的单元测试（44 个用例，秒级）
hubconf.py                    PyTorch Hub 入口
setup.py                      包配置及 CLI 注册（已把 nmf 一并打包）
pyproject.toml / .flake8      构建与代码风格配置
README.md                     上游模型清单、示例、论文与许可
```

典型数据流：**FASTA / A3M / PDB / mmCIF / CSV -> 数据转换器 -> 模型 -> 表示、接触图、PDB 或评分结果**。

## 4. 模型选择

| 模型入口 | 用途 | 注意事项 |
| --- | --- | --- |
| `esm2_t6_8M_UR50D` 到 `esm2_t48_15B_UR50D` | 通用序列表示和语言模型推理 | 参数量 8M 到 15B，建议先用小模型验证流程 |
| `esmfold_v1()` / `esmfold_v0()` | 单序列结构预测 | 依赖 OpenFold，显存需求高 |
| `esmfold_structure_module_only_*` | 仅结构模块（IPA only）的 ESMFold | 8M/35M/150M/650M/3B/15B，均有 270K 蒸馏版本 |
| `esm_msa1_t12_100M_UR50S()` / `esm_msa1b_t12_100M_UR50S()` | MSA 表示和接触预测 | 输入必须是等长多序列比对 |
| `esm1b_t33_650M_UR50S()` | ESM-1b 通用表示 | 需配套 contact-regression 权重才能预测接触图 |
| `esm1_t34_670M_UR50S/_UR50D/_UR100()`、`esm1_t12_85M_UR50S()`、`esm1_t6_43M_UR50S()` | 早期 ESM-1 系列 | 逐步被 ESM-2 取代 |
| `esm1v_t33_650M_UR90S_1()` 到 `_5()` | 零样本突变预测 | 通常使用 5 模型集成 |
| `esm_if1_gvp4_t16_142M_UR50()` | 逆折叠、骨架条件序列设计 | 输入 N/CA/C 主链坐标 |

完整模型名称、层数、参数量和训练数据参见 `README.md` 与 `esm/pretrained.py`；
离线加载本地权重用 `esm.pretrained.load_model_and_alphabet_local("模型文件.pt")`。

## 5. ESM-2 Python 基本用法

```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()

records = [
    ("protein_a", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ"),
    ("protein_b", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ"),
]
labels, sequences, tokens = alphabet.get_batch_converter()(records)

with torch.no_grad():
    output = model(tokens, repr_layers=[33], return_contacts=True)

residue_representations = output["representations"][33]
contacts = output["contacts"]
```

约定：

- 输入记录形式为 `(label, sequence)`；batch converter 负责特殊 token 和 padding。
- 推理时使用 `model.eval()` 与 `torch.no_grad()`。
- `repr_layers` 指定返回表示的层；层 0 为 embedding，之后为 Transformer 层。
- 残基表示通常需排除开头 BOS、结尾 EOS（若有）和 padding。
- 不需要接触图时关闭 `return_contacts`，可节省内存。

一个可直接运行的完整例子（含逐残基表示的切片、masked 位置预测、伪困惑度计算）：

```python
import math
import torch
import esm

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()                       # 必须 eval：ESM-2 的 token_dropout 仅训练模式生效

records = [("protein_a", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ")]
labels, strs, tokens = alphabet.get_batch_converter()(records)
print("tokens:", tokens.shape)     # (batch, len + 2)，含 BOS/EOS

with torch.no_grad():
    out = model(tokens, repr_layers=[6], return_contacts=False)
logits = out["logits"]             # (batch, len + 2, alphabet_size)
reps = out["representations"][6]   # (batch, len + 2, embed_dim)

# 去掉 BOS/EOS：逐残基表示、逐残基 logits
rep_per_res = reps[0, 1 : len(strs[0]) + 1]
logits_per_res = logits[0, 1 : len(strs[0]) + 1]

# 序列级表示：对残基表示取平均
seq_rep = rep_per_res.mean(dim=0)
print(seq_rep.shape, rep_per_res.shape)

# 人为遮挡第 0 个残基，看模型能否复原
masked = tokens.clone()
masked[0, 1] = alphabet.mask_idx
with torch.no_grad():
    mlogits = model(masked, repr_layers=[], return_contacts=False)["logits"][0, 1]
pred_idx, top5 = mlogits.topk(5)
print("true:", strs[0][0], "pred:", "".join(alphabet.get_tok(i) for i in top5))

# 单点伪困惑度（逐位置只遮挡一个残基）
ppl_terms = []
for i in range(len(strs[0])):
    mt = tokens.clone()
    mt[0, i + 1] = alphabet.mask_idx
    with torch.no_grad():
        lp = torch.log_softmax(model(mt, repr_layers=[], return_contacts=False)["logits"][0, i + 1], -1)
    ppl_terms.append(-lp[tokens[0, i + 1]].item())
print("pseudo-perplexity:", math.exp(sum(ppl_terms) / len(ppl_terms)))
```

`alphabet.get_tok(idx)` 可以把 token 索引还原成字符；`alphabet.mask_idx` 是遮挡 token
（ESM-1b 字母表中为 32），这些是第 11 节评估脚本的基础。

## 6. 批量 FASTA 表示提取

```bash
esm-extract esm2_t33_650M_UR50D input.fasta output_embeddings \
  --repr_layers 33 --include mean per_tok

# 不安装 CLI 时：
python scripts/extract.py esm2_t33_650M_UR50D input.fasta output_embeddings \
  --repr_layers 33 --include mean per_tok
```

常用参数：

- `--repr_layers`：需要保存的层；
- `--include`：`mean`、`per_tok`、`bos`、`contacts`；
- `--toks_per_batch`：最大 token 批量，OOM 时降低；
- `--truncation_seq_length`：脚本默认截断至 1022；
- `--nogpu`：禁用 GPU。

每个 FASTA 记录会保存一个 `<label>.pt`。读取示例：

```python
import torch
result = torch.load("output_embeddings/protein_0001.pt", map_location="cpu")
print(result.keys())
```

## 7. ESMFold 结构预测

```python
import torch
import esm

model = esm.pretrained.esmfold_v1().eval().cuda()
model.set_chunk_size(128)  # 可选：减小显存，牺牲速度

with torch.no_grad():
    pdb = model.infer_pdb("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ")

open("result.pdb", "w").write(pdb)
```

批量预测：

```bash
esm-fold -i input.fasta -o predicted_structures \
  --chunk-size 128 --max-tokens-per-batch 512
```

多聚体在同一序列中以 `:` 分隔链。显存不足时减小 `--max-tokens-per-batch` 或 `--chunk-size`；也可研究 `--cpu-offload`。当前 CLI 支持 `--cpu-only`、`--num-recycles` 和 `--model-dir` 等参数，以 `python scripts/fold.py --help` 为准。

## 8. 权重与数据集获取

### 8.1 预训练权重下载地址

`esm.pretrained` 根据模型名拼接 Meta 公共文件地址：

```text
模型权重： https://dl.fbaipublicfiles.com/fair-esm/models/<模型名>.pt
接触回归： https://dl.fbaipublicfiles.com/fair-esm/regression/<模型名>-contact-regression.pt
```

调用预训练加载函数会自动下载，并缓存到 PyTorch Hub 的 `checkpoints` 目录（默认通常是 `~/.cache/torch/hub/checkpoints`）。例如：

```python
import torch
import esm

torch.hub.set_dir("D:/model-cache/torch")  # 可选；应在加载模型前设置
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
```

常用模型直链（均为仓库 README 中列出的官方地址）：

| 模型 | 官方权重 |
| --- | --- |
| ESM-2 8M | [esm2_t6_8M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt) |
| ESM-2 35M | [esm2_t12_35M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt) |
| ESM-2 150M | [esm2_t30_150M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt) |
| ESM-2 650M | [esm2_t33_650M_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) |
| ESM-2 3B | [esm2_t36_3B_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt) |
| ESM-2 15B | [esm2_t48_15B_UR50D.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt) |
| ESMFold v1 | [esmfold_3B_v1.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt) |
| ESMFold v0 | [esmfold_3B_v0.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v0.pt) |
| ESM-IF1 | [esm_if1_gvp4_t16_142M_UR50.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt) |
| ESM-MSA-1b | [esm_msa1b_t12_100M_UR50S.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt) |
| ESM-1v ensemble 1-5 | [模型 1](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt)、[2](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_2.pt)、[3](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_3.pt)、[4](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_4.pt)、[5](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_5.pt) |

直接下载文件时可用浏览器或 `curl -L`，之后通过 `torch.hub.set_dir()` 将包含 `checkpoints/` 的缓存根目录指向下载位置，或使用 `esm.pretrained.load_model_and_alphabet_local("模型文件.pt")`。本地加载器对大多数 ESM-1/ESM-2/MSA 模型还要求同目录存在相配套的 `模型名-contact-regression.pt`；ESM-1v 和 ESM-IF1 不需要此 regression 权重。ESMFold 应通过 `esm.pretrained.esmfold_v1()` 等工厂函数加载，因为除权重外还需正确安装 ESMFold/OpenFold 软件依赖。

这些链接指向上游发布的固定权重版本。版本、架构名应与本仓库 `esm/pretrained.py` 对应；权重文件较大，下载前检查磁盘空间和网络策略。模型权重和数据集许可可能不同，Atlas 数据请另行查看其许可。

### 8.2 数据集官方获取入口

| 数据 | 下载入口 | 获取说明 |
| --- | --- | --- |
| ESM Structural Split 的 train/valid split | [splits.tar.gz](https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz) | `ESMStructuralSplitDataset(download=True)` 自动下载；也可手动解压到缓存根目录下 `structural-data/splits/` |
| ESM Structural Split 的序列/SSP/距离/坐标 pickle | [pkl.tar.gz](https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz) | 与 splits 配套；自动下载后放在 `structural-data/pkl/` |
| ESM Structural Split 的 MSA | [msas.tar.gz](https://dl.fbaipublicfiles.com/fair-esm/structural-data/msas.tar.gz) | Notebook/自定义 MSA 实验可选下载；Dataset 类本身只自动下载 splits 与 pkl |
| ESM-IF1 CATH 4.3 backbone 和序列 | [chain_set.jsonl](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl) | 逆折叠训练/评估的链级数据 |
| ESM-IF1 CATH 4.3 split | [splits.json](https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/splits.json) | 与上行 chain set 配套的拓扑切分 |
| UniRef50 预训练验证集 ID | [uniref201803_ur50_valid_headers.txt.gz](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur50_valid_headers.txt.gz) | 仅包含 UniRef 2018-03 版本 ID，不含序列库本身 |
| UniRef100 预训练验证集 ID | [uniref201803_ur100_valid_headers.txt.gz](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur100_valid_headers.txt.gz) | 与 README 描述的 UniRef 2018-03 数据版本配套 |
| P62593 示例预计算表示 | [P62593_reprs.tar.gz](https://dl.fbaipublicfiles.com/fair-esm/examples/P62593_reprs.tar.gz) | 监督变体预测 Notebook 的可选预计算输入 |

Structural Split 数据集的可编程下载与读取方式见下一节。下载后先检查实际解压目录，确保传给构造函数的 `root_path` 下形成 `structural-data/splits` 和 `structural-data/pkl` 两个目录。UniRef 验证文件只是评估划分 ID，不是用于重建整个预训练语料的下载包。

### 8.3 数据集组织与使用

本仓库不包含完整预训练语料及完整 Atlas 数据库；它提供读取接口、示例数据、结构基准集下载逻辑和 Atlas URL 清单。不同任务使用不同格式。

#### 8.3.1 数据类型与位置

| 类型 | 位置/来源 | 格式 | 用途 |
| --- | --- | --- | --- |
| 单序列 | `examples/data/*.fasta` 或用户文件 | FASTA | ESM-1/2、ESMFold |
| 多序列比对 | `examples/data/*.a3m` | A3M | MSA Transformer |
| 结构 | `examples/inverse_folding/data/*.pdb`、`*.cif` | PDB/mmCIF | ESM-IF1、固定骨架设计 |
| 变体数据 | `examples/variant-prediction/data/*.csv` | CSV | ESM-1v 预测/评估 |
| 结构分割集 | 运行时下载 | splits + pickle | 结构泛化评测 |
| Atlas | `scripts/atlas/` URL 清单 | metadata + 分片文件 | 大规模结构/嵌入分析 |

建议自行组织数据：

```text
data/
  raw/            # 原始 FASTA、A3M、PDB/mmCIF、CSV
  processed/      # 清洗、切分、标准化后的数据
  embeddings/     # esm-extract 输出 .pt
  structures/     # esm-fold 输出 PDB
  scores/         # 变体/逆折叠评分 CSV
```

不要将模型权重、Atlas 文件、大规模 embedding 或批量 PDB 提交到 Git。

#### 8.3.2 FASTA：单序列数据

```text
>protein_0001
MKTIIALSYIFCLVFADYKDDDDK
>protein_0002
GAVLILKKKGHHEAELKPLAQSHATK
```

`FastaBatchedDataset.from_file()` 将 `>` 后的整行用作 label，并拼接后续全部序列行。label 必须唯一；它还会被用于 `.pt`/`.pdb` 输出文件名，因此建议使用安全、简短的 ID，额外描述写入独立 CSV/Parquet。

```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
dataset = esm.FastaBatchedDataset.from_file("data/raw/proteins.fasta")
batches = dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1)
loader = torch.utils.data.DataLoader(
    dataset, batch_sampler=batches,
    collate_fn=alphabet.get_batch_converter(truncation_seq_length=1022),
)
for labels, sequences, tokens in loader:
    print(labels, tokens.shape)
```

`get_batch_indices()` 按长度分组，减少 padding 并限制单批 token 数。应记录截断长度和原始序列长度。ESMFold 多聚体必须写成一条 FASTA 记录，链用 `:` 分隔。

#### 8.3.3 A3M/MSA：MSA Transformer 数据

一个 MSA 样本是一组针对同一 query 的**已对齐**同源序列，而不是普通 FASTA 批次。A3M 是 FASTA 式格式：`-` 表示 gap，小写字符通常为 insertion。

```python
import esm

msa = list(esm.data.read_fasta(
    "data/raw/family.a3m",
    keep_gaps=True, keep_insertions=False, to_upper=True,
))
model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
labels, strings, tokens = alphabet.get_batch_converter()([msa])
# [batch, alignment_depth, aligned_length + special_tokens]
```

`MSABatchConverter` 要求一个 MSA 内的序列等长。`examples/variant-prediction/predict.py` 会去除 A3M 小写 insertion、`.`、`*`，并默认取开头 400 条序列。建议 query 放在第一条，并在入模前完成 MSA 构建、过滤和深度裁剪。

#### 8.3.4 ESM Structural Split Dataset

`ESMStructuralSplitDataset` 是仓库内置的 PyTorch `Dataset`，基于 SCOPe 结构域，提供 `family`、`superfamily`、`fold` 三种结构隔离难度的 5 折划分。每个样本包含：

- `seq`：长度 L 的序列；
- `ssp`：长度 L 的二级结构标签，缺失处为 `-`；
- `dist`：`L x L` 距离矩阵；
- `coords`：`L x 3` 的 Cβ 坐标，缺失处为 NaN。

```python
from esm.data import ESMStructuralSplitDataset

train_set = ESMStructuralSplitDataset(
    split_level="superfamily", cv_partition="0", split="train",
    root_path="data/esm_cache", download=True,
)
valid_set = ESMStructuralSplitDataset(
    split_level="superfamily", cv_partition="0", split="valid",
    root_path="data/esm_cache", download=False,
)
sample = train_set[0]
print(sample["coords"].shape, sample["dist"].shape)
```

默认缓存位置为 `~/.cache/torch/data/esm/structural-data/`，目录包含 `splits/<level>/<fold>/{train,valid}.txt` 和 `pkl/<ID前两字符>/<ID>.pkl`。必须将同一 level/fold 的 train 和 valid 配套使用。该类不提供变长矩阵的 `collate_fn`，下游训练需自行 padding 和构造 mask。

#### 8.3.5 逆折叠结构数据

ESM-IF1 使用每个残基的 N、CA、C 主链坐标，单链坐标形状为 `L x 3 x 3`：

```python
import esm.inverse_folding

structure = esm.inverse_folding.util.load_structure("data/raw/target.pdb", chain_id="A")
coords, native_sequence = esm.inverse_folding.util.extract_coords_from_structure(structure)
print(coords.shape)  # (L, 3, 3)
```

多链使用 `extract_coords_from_complex()`，返回按 chain ID 映射的坐标和序列。缺失坐标可用 `np.inf` 标记。`examples/inverse_folding/data/` 提供 PDB/mmCIF、待评分 FASTA 和输出示例；其中 `example.json` 形式为 `{"coords": ..., "seq": "..."}`，`Infinity` 表示缺失坐标。

采样：

```bash
python examples/inverse_folding/sample_sequences.py \
  examples/inverse_folding/data/5YH2.pdb --chain C \
  --num-samples 3 --outpath output/sampled.fasta
```

评分：

```bash
python examples/inverse_folding/score_log_likelihoods.py \
  examples/inverse_folding/data/5YH2.pdb candidates.fasta \
  --chain C --outpath output/scores.csv
```

候选 FASTA 序列应与目标链的长度和残基位置相容。加 `--multichain-backbone` 可使用整个复合物作为条件骨架。

#### 8.3.6 深度突变扫描（DMS）CSV

`examples/variant-prediction/data/BLAT_ECOLX_Ranganathan2015.csv` 是 DMS 输入示例。默认突变列名是 `mutant`，突变格式为 `AiB`，如 `A42V`：首/末字符表示野生型/突变氨基酸，中间数字表示实验残基编号。

```csv
mutant,experimental_score
A42V,0.73
L43P,-1.20
```

`--offset-idx` 将实验编号映射为 Python 0-based 索引；程序会断言参考序列对应位置确实为野生型氨基酸。输出 CSV 保留原列，增加以模型名命名的预测分数列。`rho_pp.csv` 和 `aggregated_rho.csv` 等是论文统计结果，不是模型输入。

#### 8.3.7 ESM Metagenomic Atlas

`scripts/atlas/` 保存 Atlas `v0`、`v2023_02` 的下载清单，不保存数据本体。PDB 结构、Foldseek DB 和预计算 ESM-2 embedding 按 pTM/pLDDT 区间分桶并拆分为大量分片，规模可达 TB 级。

推荐先查询 metadata Parquet/SQLite：

```python
import pandas as pd
metadata = pd.read_parquet("metadata.parquet")
selected = metadata[(metadata.plddt >= 0.7) & (metadata.ptm >= 0.7)]
selected = selected[~selected.plddt.isna()]
```

关键 metadata 列包括 `id`、`ptm`、`plddt`、`num_conf`、`len`、`is_fragment`、`sequenceChecksum`、`esmfold_version`、`atlas_version`、`sequence_dbs`。长度大于 1280 的蛋白没有折叠结果。筛选后再选择对应置信度 bin 的 URL 清单下载：

```bash
aria2c --dir data/atlas --input-file scripts/atlas/v2023_02/full/tarballs.txt
```

实际使用时应选择更小的目标分桶清单，先估算磁盘容量，并遵守 Atlas 许可条款。

## 9. 其他示例

- `examples/contact_prediction.ipynb`：ESM-2 与 MSA Transformer 接触预测；
- `examples/sup_variant_prediction.ipynb`：使用 ESM 表示训练监督变体分类器；
- `examples/esm_structural_dataset.ipynb`：结构分割数据集使用示例；
- `examples/lm-design/`：固定骨架和自由生成设计，需安装 `additional_requirements.txt`；
- `examples/protein-programming-language/`：高层目标驱动的生成设计，教程在 `tutorial.ipynb`；
- `examples/esm2_infer_fairscale_fsdp_cpu_offloading.py`：15B ESM-2 的 FSDP/CPU offload 推理；
- `examples/inverse_folding/`：`sample_sequences.py`、`score_log_likelihoods.py` 等逆折叠脚本；
- `examples/variant-prediction/`：ESM-1v 零样本变体效应预测与 DMS 数据。

## 10. 测试与常见问题

运行测试：

```bash
pytest                          # 全部测试
pytest tests/test_nmf.py -v     # 只跑非负分解实验的单元测试（44 个用例，无需下载权重，约 5 秒）
```

`tests/test_nmf.py` 覆盖形状/非负约束/方阵、迭代确实降低误差、降秩确实变差、平移模式、
前向等价性、bias 保留、层名解析、可训练参数范围、检查点往返，以及评测口径指纹/逐变体落盘/
无模型渲染报告、指标汇总各节渲染等 44 个用例。

`tests/test_load_all.py` 会加载很多预训练模型，可能触发大量下载并占用资源；
`tests/test_notebooks.py` / `tests/test_readme.py` 会执行 Notebook 与 README 中的片段，
在有网络且已装齐依赖的环境下才有意义。常见问题：

- **权重下载失败**：检查模型名、网络、缓存目录权限；可用 `torch.hub.set_dir()` 更改缓存目录。
- **CUDA OOM**：使用更小模型，降低 `--toks_per_batch`、`--max-tokens-per-batch` 或 ESMFold 的 chunk size。
- **FASTA 标签重复**：`FastaBatchedDataset` 不允许重复 label。
- **MSA 长度不一致**：同一个 MSA 内必须等长。
- **逆折叠坐标缺失**：按 ESM-IF1 约定用 `np.inf` 标记。
- **示例依赖缺失**：基础 ESM、ESMFold、逆折叠和设计示例的依赖不同，应分别安装。
- **表示里混入了特殊 token**：`representations` 与 `logits` 都包含 BOS/EOS，取残基级结果时要按
  `[..., 1 : 1 + len(seq)]` 切片。
- **训练模式下结果每次不同**：ESM-2 默认 `token_dropout=True`，`model.train()` 时会随机遮挡；
  做确定性推理/评估请先 `model.eval()`。
- **`torch.load` 报 `Weights only load failed`**：PyTorch ≥ 2.6 默认 `weights_only=True`。
  本仓库 `nmf` 产出的检查点只含张量与基础类型，可直接加载；若换成含自定义对象的检查点，
  需显式 `torch.load(..., weights_only=False)`（仅在信任来源时使用）。
- **`/tmp` 空间不足导致 pip 失败**：本机 `/tmp` 为 10 MB tmpfs，先 `export TMPDIR=/home/zzj/tmp`。

## 11. 线性层的三因子非负矩阵分解（NMF）实验

> 代码位于 `nmf/`，详细说明与全部实测数据见 `nmf/README.md`。本节给出目标、数据、
> 求解器、命令、评测协议与完整结果。

### 11.1 目标与形式化

对 ESM 中任意一个 `nn.Linear` 的权重 `W`（`out_features × in_features`）做三因子非负分解，
**中间为方阵**：

```
W ≈ A @ S @ B + offset
A: (out_features, rank)  非负
S: (rank, rank)          非负方阵     <- 中间矩阵
B: (rank, in_features)   非负
offset: (out_features,)  平移项
```

**为什么需要 `offset`**：NMF 要求目标矩阵非负，而预训练权重有正有负。因此先把权重整体平移
（默认 `offset = min(W)`）得到非负目标 `W' = W - offset[:, None]`，再对 `W'` 做非负分解；
`offset` 在前向中以 `x.sum(-1) * offset` 精确加回，等价于给权重的每一行加一个常数。
于是**除这一行常数外，原层的信息全部由 A/S/B 三个非负矩阵承载**。

两个必须注意的效应：

- `‖W'‖ / ‖W‖ ≈ 3 ~ 5.5`，所以同一个分解"相对非负目标的误差"很小，而"相对原权重的误差"
  会被放大同样的倍数——脚本同时输出两个数字，判断重构质量请看后者。
- `rank = min(in, out)`（满阶）时形式上存在**精确解**（`A = W'`、`S = I`、`B = I`），
  此时非负分解是"换参数化"而非压缩；**真正压缩的区间是 `rank < min(in, out)`**。

### 11.2 数据：UniProtKB/Swiss-Prot（权威、多样、已去污染）

本仓库自带的样例（`examples/data/*.fasta`）规模小、来源单一（例如 5397 条等长 286 的
同一酶家族变体），不足以支撑"性能不下降"的结论。正式实验用 `nmf/prepare_data.py`
从 **UniProtKB/Swiss-Prot（reviewed，人工审阅）** 通过 REST 流式接口下载：

```
https://rest.uniprot.org/uniprotkb/stream
  ?query=(reviewed:true) AND (fragment:false) AND (length:[50 TO 1022])
  &format=fasta
```

（`rest.uniprot.org` 实测约 2 MB/s；`ftp.uniprot.org` 仅 ~30 KB/s，不要用。）
ESM-2 的预训练语料是体量更大的 UniRef50 2021_04（约 4900 万条）；Swiss-Prot 是其人工审阅
子集，下载量与算力可控且分布多样，适合做严谨对比。

```bash
python -m nmf.prepare_data --source swissprot \
  --raw-file data/raw/swissprot_reviewed.fasta \
  --out-dir data/processed/swissprot \
  --min-len 50 --max-len 1022 \
  --train 30000 --valid 2000 --test 4000 --seed 0 \
  --decontaminate --decontam-threshold 0.6 --kmer 6
```

实际得到的数据（完整记录见 `data/processed/swissprot/stats.json`，含 sha256）：

| 阶段 | 条数 | 说明 |
| --- | --- | --- |
| 原始下载 | 524,874 | reviewed + 非 fragment + 长度 50~1022 |
| 质控后 | 439,721 | 丢弃非常规残基 1,497、精确重复 83,656 |
| train | 30,000 → **29,124** | 去污染剔除 876 条 |
| valid / test | 2,000 / 4,000 | 长度 p50≈312、p95≈740、max 1020 |

train 长度分布：**min 50、p50 312、p95 760、max 1022、平均 346 残基，共 10,085,760 残基**。

**去污染**：用 6-mer 包含度作为同源性代理，剔除与 valid/test 高度相似（覆盖度 ≥ 60%）的
train 序列——这是 ESM-2 论文用 MMseqs 以 50% 序列一致性清洗训练集的**无需外部二进制的
近似实现**，避免测试集泄漏。

### 11.3 求解器：HALS 让满阶分解逼近精确解

离线分解提供三个求解器，默认 `hals`：对 A 的每一列、B 的每一行做**精确** rank-1 非负
最小二乘，S 的更新在 `r × r` 的约化问题上用 FISTA 求解。实测对比（`esm2_t6_8M_UR50D`）：

| 层 | rank | 求解器 | 迭代 | 相对非负目标误差 | **相对原权重误差** | 余弦相似度 |
| --- | --- | --- | --- | --- | --- | --- |
| `layers.0.self_attn.k_proj` | 320（满） | `hals` | 600 | **0.0016** | **0.0160** | **0.99987** |
| `layers.5.self_attn.out_proj` | 320（满） | `hals` | 300 | 0.0026 | 0.0205 | 0.99980 |
| `layers.5.self_attn.out_proj` | 320（满） | `mu` | 2000 | 0.0131 | 0.1019 | 0.9951 |
| `layers.5.self_attn.out_proj` | 320（满） | `pgd` | 2000 | 0.0193 | 0.1503 | 0.9891 |
| `layers.5.fc1` | 160（半） | `hals` | 300 | 0.0997 | 0.5482 | 0.8387 |

**HALS 在满阶时把相对重构误差压到 1e-3 量级（余弦 0.9999+）**，比乘性更新/投影梯度好
1~2 个数量级——这是"分解后性能不下降"的基础。降秩时误差由秩本身决定，与求解器关系不大。

### 11.4 命令

```bash
# 阶段 1：全模型 38 个线性层的非负分解（满阶）
python -m nmf.factorize --layers all --include-all-linear \
  --solver hals --iters 600 --shift min \
  --out nmf/outputs/fullmodel/factors_fullrank.pt

# 阶段 2：批次训练（只更新 A/S/B/offset，冻结其余参数）
python -m nmf.train_nmf --layers all --include-all-linear \
  --init-checkpoint nmf/outputs/fullmodel/factors_fullrank.pt \
  --fasta data/processed/swissprot/train.fasta --max-records 4000 \
  --max-tokens-per-batch 4096 --epochs 3 --lr 1e-3 --kl-scope all \
  --distill-weight 1.0 --recon-weight 0.1 --max-recon-degradation 1.0 \
  --eval-every 50 --eval-fasta data/processed/swissprot/valid.fasta \
  --out nmf/outputs/fullmodel/trained_fullrank.pt --save-every 50 --plot

# 阶段 3：快速对比诊断
python -m nmf.evaluate --checkpoint <ckpt...> --names 未训练 训练后 \
  --fasta data/processed/swissprot/test.fasta --max-records 8

# 阶段 4：论文级评测（伪困惑度、CKA、mean±std）
#   每个变体算完立即落盘到 <out-dir>/variants/，长评测中断不会丢结果
python -m nmf.benchmark \
  --checkpoint "满阶未训练=nmf/outputs/fullmodel/factors_fullrank.pt" \
               "满阶训练后=nmf/outputs/fullmodel/trained_fullrank.pt" \
               "半秩未训练=nmf/outputs/fullmodel/factors_halfrank.pt" \
  --fasta data/processed/swissprot/test.fasta --n-eval 96 \
  --mask-frac 0.15 --mask-rounds 5 --ppl-records 10 --ppl-max-len 256 \
  --out-dir nmf/outputs/fullmodel/bench_final --plot

# 中断后续跑（复用已算好的原模型指标与已完成变体）
python -m nmf.benchmark --checkpoint <同上> --fasta <同上> --out-dir <同上> \
  --reuse-base --reuse-variants

# 只重画表 / 报告丢了：不加载模型，从落盘结果重新渲染
python -m nmf.benchmark --from-dir nmf/outputs/fullmodel/bench_final --plot
```

一键脚本：`bash nmf/run_all.sh`（快速小样例）、`bash nmf/run_full_experiment.sh`（单个 tag 的完整严谨实验）、
**`bash nmf/run_matrix.sh`（完整实验矩阵 + 指标汇总，已存在的产物自动跳过，推荐）**。

所有指标汇总成一份可直接引用的表（纯后处理，不加载模型，可随时重跑）：

```bash
python -m nmf.summarize --root nmf/outputs/fullmodel \
  --bench nmf/outputs/fullmodel/bench_matrix --data data/processed/swissprot \
  --out nmf/outputs/fullmodel/SUMMARY.md
# -> SUMMARY.md（数据 / 逐层分解质量 / 训练过程 / 模型级能力对比）
#    SUMMARY.json（机器可读）、layer_metrics.csv（逐层明细）
```

`DRY_RUN=1 bash nmf/run_matrix.sh` 可以先看它会做什么。

### 11.5 训练细节（含一个关键的稳定性陷阱）

- 冻结全部其余参数，只更新 A/S/B（可选 `--train-bias`、`--freeze-offset`）；
- 每步梯度更新后把 A/S/B `clamp` 回 ≥ 0，**全程保持三因子非负、中间方阵约束**；
- 损失 = `distill_weight × KL(原模型‖分解模型)`（遮挡 15%，`--kl-scope all` 时统计全部残基位置）
  + `ce_weight × CE(预测, 真实氨基酸)` + `recon_weight × ‖W − (A S B + offset)‖_F / ‖W‖_F`；
- **陷阱：统一学习率的 Adam 必然训坏模型。** 中间矩阵 S 夹在两侧因子之间，其逐元素扰动被
  放大 `(A 行和)×(B 列和)` 倍；而 NMF 解的尺度存在歧义（实测 `layers.5.fc1` 的 A 元素最大
  0.30、B 元素最大 17878）。实测对 S 加 δ=1e-4 的同号扰动会让该层相对误差从 0.016 涨到
  **10.87**（比 B 灵敏 3 个数量级），而 Adam 的步长与梯度幅值无关（≈ sign(g)），于是单次
  `lr=1e-4` 的步就把全模型相对重构误差从 **0.0417 爆到 1.699**，模型几秒内坏掉。
  **修复**：按各因子增益分组标定学习率，`lr_factor = --lr / gain_factor`（`--lr-scale auto`，
  默认），使各因子单步的等效权重移动量一致。实测得到 A=5.6e-8、S=3.6e-9、B=2.5e-5、
  offset=2.8e-5，训练全程稳定。
- **信任域保护**：每步检查相对重构误差，超过 `初始值 × (1 + --max-recon-degradation)` 就
  回滚该步并把学习率减半重试（最多 `--rollback-tries` 次），回滚的步打印 `[回滚]`。
- `--save-every` 周期性保存检查点，长训练中断也能拿到可用结果。

### 11.6 论文级评测协议

`nmf.benchmark` 在留出测试集上报告：

| 指标 | 协议 | 意义 |
| --- | --- | --- |
| Masked-LM top-1 / top-5 | 遮挡 15% 真实残基，5 回合 **mean ± std** | 能力保持 |
| masked 困惑度 | 同上协议的 `exp(平均 NLL)` | 与语言建模同量纲 |
| **伪困惑度** | **标准单点遮挡**：逐位置只遮一个残基 | 蛋白语言模型最常引用的自监督指标 |
| KL / JS / top-k 一致率 | 未遮挡输入的逐残基分布对比 | 行为一致性 |
| **线性 CKA** | Kornblith et al. (2019)，全部真实残基位置 | 表征等价性（1.0 = 完全一致） |
| 层级别 + 效率 | 平均/最大相对 Frobenius 误差、余弦、参数量、**线性层乘加比** | 压缩与计算代价 |

### 11.7 完整实验的实测结果

规模：`esm2_t6_8M_UR50D`（7,512,474 参数）的**全部 38 个线性层**同时替换为三因子非负分解层；
数据为 Swiss-Prot 切分出的 train 29,124 条（1008.6 万残基）；评测在留出 test 集上按上面的协议进行。

**第一步：满阶非负分解（HALS，600 次迭代，CPU 约 30 分钟）**

| 指标 | 满阶（r = min(in,out)） | 半秩（r = 0.5·min(in,out)） |
| --- | --- | --- |
| 分解层数 | 38 | 38 |
| 平均相对原权重误差 ↓ | **0.0417**（中位 0.0407，最大 0.1363） | 0.3441（最大 0.5462） |
| 平均余弦相似度 ↑ | **0.99874**（最小 0.99149） | 0.93055（最小 0.83994） |
| 分解层参数量（原 → 分解） | 7,475,320 → 15,052,922（**2.014×**） | 7,475,320 → 6,579,322（**0.880×**） |
| 耗时 | 1794 s | 1018 s |

即：满阶时非负约束几乎不损失信息（余弦 0.9987），代价是参数量翻倍；半秩时参数量降到 0.88×，
但层重构误差上升到 0.34。

**第二步：批次训练（只更新 A/S/B/offset，1104 步 = 3 轮，CPU 约 65 分钟）**

- 分组学习率（按扰动增益标定）：`A 5.59e-8 / S 3.55e-9 / B 2.53e-5 / offset 2.80e-5`，全程 **0 次回滚**；
- 蒸馏损失 **0.0648 → 0.0123**（KL(原‖分解)，统计全部残基位置）；
- 相对重构误差 0.04176 → 0.04635（+11%，在 `--max-recon-degradation 1.0` 的信任域内）；
- 训练中在 valid（32 条）上的快速评估单调改善：遮挡 top-1 0.2111 → 0.2154（原模型 0.2207），
  KL 0.0339 → 0.0165，top-1 一致率 0.7938 → 0.8448。

**第三步：论文级能力对比（留出 test 集，64 条 / 24,034 残基，遮挡 5 回合取均值）**

| 变体 | 遮挡位 top-1 ↑ | masked 困惑度 ↓ | 伪困惑度 ↓ | KL(分解‖原) ↓ | 遮挡位 top-1 一致率 ↑ | 逐残基 CKA ↑ | 平均层重构误差 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原模型（基准） | 0.2005 | 13.275 | 12.398 | — | — | 1.0000 | — |
| 满阶未训练 | 0.1792（−11%） | 14.211（+7.0%） | 13.473（+8.7%） | 0.0824 | 0.9745 | 0.8746 | 0.0417 |
| **满阶训练后** | **0.1993（−0.6%）** | **13.420（+1.1%）** | **12.544（+1.2%）** | **0.0068** | **0.9935** | **0.9732** | 0.0464 |
| 半秩未训练 | 待补（见下方说明） | 待补 | 待补 | — | — | — | 0.3441 |

**结论**：

1. **"满阶 + 少量蒸馏训练"确实做到了性能不下降**：训练后遮挡 top-1 与原模型的差距从 11% 收到 **0.6%**
   （0.1993 vs 0.2005，5 回合均值），masked 困惑度差距 1.1%，标准单点伪困惑度差距 **1.2%**
   （12.544 vs 12.398）；行为一致性大幅提高——逐残基 KL 从 0.082 降到 **0.0068**（降低 12 倍），
   遮挡位 top-1 一致率 0.9745 → **0.9935**，逐残基表示 CKA 0.875 → **0.973**。
2. **训练的作用是"补偿非负约束带来的信息损失"，不是"提升能力"**：权重重构误差在训练中略有上升
   （0.0417 → 0.0464），但模型级指标全面改善——说明优化目标是让 `A S B` 去拟合**原模型的行为**
   （KL 蒸馏），而不是拟合原权重本身。
3. **半秩（0.88× 参数量）是有损压缩**：层重构误差 0.344、余弦 0.931。用 8 条序列的快速诊断显示
   其分布一致性大幅退化（KL ≈ 1.5、top-1 一致率 ≈ 0.12），即在这个压缩比下模型行为已明显偏离——
   这也说明**当前的非负约束下，"参数量下降"与"性能不下降"不能同时成立**（详见 §11.9）。

**未跑完 / 待补的部分（诚实说明）**：

- 上表"半秩未训练"一行原计划在同一个评测进程里一起产出，但该进程在评测到第三个变体时被中断
  （运行环境回收），而旧版 `nmf.benchmark` 只在全部变体算完后才写盘，因此**中间结果全部丢失**。
  这个健壮性缺陷已经修复（现在每个变体算完立即落盘到 `<out-dir>/variants/<名称>.json`，并支持
  `--reuse-base` / `--reuse-variants` 续跑、`--from-dir` 从落盘结果重新渲染报告），可用一条命令补齐。
- 满阶未训练一行的完整报告在 `nmf/outputs/fullmodel/bench_untrained/`（test **96** 条，其中的
  数值与上表略有差异属于子集不同）；满阶训练后一行的数值直接取自该次运行的日志。
- **伪困惑度的口径修正**：旧版把 `--ppl-max-len` 同时当作长度过滤条件，导致伪困惑度只在"长度 ≤ 256"
  的子集上统计（长序列被排除而不是截断）。现已改为只截断不过滤，因此**上表中的伪困惑度绝对值为旧口径**，
  重新评测后数值会变化（模型间的相对差距结论不受影响）。


### 11.8 使用建议

- **参数量**：分解层为 `out×r + r×r + r×in`（原始 `out×in`）。方阵层（`in = out`）只有
  `r ≲ 0.586·in` 时才有压缩收益；满阶时参数量约 2×（换参数化，非压缩）。
- **求解器**：优先 `hals`；`mu`/`pgd` 保留作对照。超大层可用更少迭代或 `mu`。
- **秩的选择**：先用 `--rank-ratio 0.5` 观察代价，再决定是否值得为性能保留满阶。
- **多试几层**：`--layers all` 分解全部线性层；也可只替换 `self_attn.*_proj` 或只替换 FFN，
  考察不同模块对非负约束的敏感度。
- **训练**：务必用 `--lr-scale auto`；`--lr` 是"每步允许的等效权重相对移动量"，默认 1e-3
  与 `--max-recon-degradation 1.0` 匹配（约 1000 步耗尽信任域预算）。
- **对照实验**：同一套评测可直接用于比较 SVD/低秩分解、量化等其他压缩方式。

### 11.9 已知限制与待补项

**已修复（本轮审计发现）**

| 问题 | 影响 | 修复 |
| --- | --- | --- |
| `nmf.benchmark` 只在全部变体算完后写盘 | 多窗口长评测中途被中断 → 所有结果丢失（本次真实发生） | 每个变体算完立即写 `<out-dir>/variants/<名称>.json`；新增 `--reuse-base`/`--reuse-variants` 续跑、`--from-dir` 从落盘结果重新渲染报告；变体口径（模型/测试集/条数/遮挡/种子）不一致时明确跳过并提示 |
| `--ppl-max-len` 被当作长度过滤条件 | 伪困惑度只在"长度 ≤ 该值"的子集上统计（长序列被排除）；若测试集最短序列都长于该值则静默返回 `nan` | 改为只在 `pseudo_perplexity` 内部截断、不参与抽样过滤；无可用序列时显式报错而不是返回 `nan` |
| `nmf_layers` 模块文档称"迭代求解器达不到满阶精确解" | 与 HALS 实测（单层 1.6e-3）矛盾，误导秩的选择 | 已改写，并说明 HALS 满阶精度与默认值 |
| `factorize_matrix` 文档写 `solver={"pgd","mu"}`、`shift={"row","min",...}` | 与实现的 `hals` 默认、`min` 默认不符 | 已按实现更新，并补充 `n_inner` |
| `build_parameter_groups` 文档描述旧的缩放公式 | 文档与 `lr/gain` 实现不符 | 已改写并说明 `--lr` 的语义 |
| `PROJECT_GUIDE.md` §11.7 残留 `{{RESULTS}}` 占位符 | 交付文档不完整 | 已用真实结果填充 |

**仍然存在的限制（不打算在本次修，需明确知晓）**

1. **半秩的论文级指标缺一行**：`factors_halfrank.pt` 已生成（层误差 0.344、参数 0.880×），
   但缺同一口径下的模型级评测。补齐命令见 §11.4 / `nmf/README.md` §3。
2. **伪困惑度的历史数值是旧口径**（长度过滤版），重新评测后绝对值会变。
3. **没有多种子重复**：`--mask-rounds` 给的是同一模型多次遮挡的标准差，不是跨训练种子的方差；
   严格论文级结论应至少用 3 个训练种子各跑一次，用 `--out-dir` 分目录后用 `--from-dir` 汇总对比。
4. **没有外部基线**：目前只对比"原模型 vs 训练后的原模型"，缺少 SVD/低秩（`W ≈ U V`）与
   量化等压缩基线的同口径对照。现成的 `nmf.benchmark` 可直接复用，只需先造出对应检查点。
5. **只在 CPU、只测 8M 模型**：`--device cuda` 与更大的 ESM-2（35M/150M/650M）未验证；
   38 层的 HALS 分解在 CPU 上需 30 分钟，650M 模型需按层分批做。
6. **满阶时参数量是 2.014×**：这不是压缩，而是"用非负参数化重写线性层"；若目标是压缩，
   必须接受降秩带来的能力损失（见 §11.7 结论 3）。

## 12. 许可与引用

代码采用 MIT License，见 `LICENSE`。Atlas 数据另有 CC BY 4.0 与 Meta 相关条款。学术使用应根据实际使用的模型引用 README 中对应论文：ESM-2/ESMFold、ESM-1/1b、MSA Transformer、ESM-1v 或 ESM-IF1。
