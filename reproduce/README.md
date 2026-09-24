# 两篇论文的复现工具包

> **想直接看「怎么跑」和「论文用了多少机器」？** → 看 **[`RUNBOOK.md`](RUNBOOK.md)**
> （环境无关版：标准环境安装、逐实验命令、论文原文的设备与规模、算力预算与分期排期）。
> 本 README 主要讲工具包的设计、与论文不一致之处、以及本机现状的排错。

覆盖仓库 `examples/` 下两篇论文的实验：

| 论文 | 仓库对应目录 | 论文里的实验 | 本工具包入口 |
|:--|:--|:--|:--|
| **A high-level programming language for generative protein design**<br>（Hie et al., 2022, bioRxiv 2022.12.21.521526） | `examples/protein-programming-language/` | 图 2 ~ 图 5 的全部设计实验（自由幻觉、固定骨架、二级结构、功能位点脚手架、对称单体、寡聚体、双层对称、层级非对称）+ 逆折叠 roundtrip | `paper1_programming/run_p1.sh` |
| **Language models generalize beyond natural proteins**<br>（Verkuil, Kabeli et al., 2022, bioRxiv 2022.12.21.521521） | `examples/lm-design/` | 固定骨架设计、自由生成、序列新颖性、湿实验统计（图 2 ~ 图 4 + 附录 A） | `paper2_lm_design/run_p2.sh` |

---

## 0. 先看这一节：能不能跑、要花多少代价

**这个仓库只给了「零件」，没给「流水线」。** 两篇论文的实验都需要你自己把它们串起来。
关于**论文本身用了什么设备、提交了多少跑动、总共要多少卡时**，见
[`RUNBOOK.md` 第 4 节](RUNBOOK.md#4-论文中的设备与规模)。三个必须知道的事实：

### 0.1 论文一：仓库完全没有命令行入口

`examples/protein-programming-language/` 里只有：
- `language/` —— 语言运行时（`ProgramNode`、12 个能量约束、模拟退火 `run_simulated_annealing`、ESMFold 回调）；
- `programs/*.py` —— 7 个**构造函数**（如 `free_hallucination(100)` 返回一个 `ProgramNode`），**不是** 可执行脚本；
- `tutorial.ipynb` —— 教你怎么写 program 并跑 1 次优化。

论文正文里每一次「跑动」= 一次 **30,000 步模拟退火**，每一步都要过一次 ESMFold。
论文实际提交的规模是 **5,930 次跑动 ≈ 1.78 亿次 ESMFold 前向**（逐条核对见
[`RUNBOOK.md` 第 4.2 节](RUNBOOK.md#42-论文提交的实验规模全部来自-methods逐条核对)）。
所以本工具包的 `design_programs.py` 就是那个缺失的驱动：它把论文的
spec × seed 网格枚举出来、逐条跑、逐条落盘、可断点续跑。

### 0.2 论文二：有 CLI，但一次只跑一个 seed，且结果不落盘

`examples/lm-design/lm_design.py` 是个 hydra 脚本，一次只能：
- 跑 1 个 seed × 1 个 target 的固定骨架设计，或 1 条自由生成；
- 结果只写日志（`logging.l`），不落盘任何序列/指标文件。

`run_lm_design_batch.py` 补上了批量 + 多 seed + 落盘 + 能量轨迹记录。

### 0.3 本机现状（**仅供本机排错**；正常 GPU 环境请直接看 [`RUNBOOK.md`](RUNBOOK.md) 第 1 节）

| 项目 | 现状 | 影响 |
|:--|:--|:--|
| GPU | 有 **NVIDIA RTX 5070 Ti Laptop**，但 conda 环境 `ESM` 里装的是 **CPU 版 torch** | 论文规模**跑不动**，必须先装 CUDA 版 torch |
| OpenFold / ESMFold | **未安装** | 论文一的**任何**设计实验都跑不了；论文二的核心 LM 部分不受影响 |
| `biotite / hydra / nltk / scipy / pandas` | 未安装 | 论文一与论文二的脚本都会 import 失败 |
| `/tmp` | 10 MB tmpfs | 下载/编译前必须 `export TMPDIR=/home/zzj/tmp`（工具包已自动处理） |

**因此建议的路线是：**

```
① bash 00_setup_env.sh check              # 看缺什么（30 秒）
② MODE=smoke bash run_all.sh              # 打通流程（论文二可跑，论文一需先装 openfold）
③ bash 00_setup_env.sh base               # 装 CPU 可跑的依赖
④ bash 00_setup_env.sh esmfold            # 单独建环境装 ESMFold（要 CUDA，见脚本内说明）
⑤ MODE=full  bash run_all.sh              # 上论文规模，按需追加 n=1 的算力（见 1.3 节估算）
```

> **注意**：如果你只是想核对论文的实验结论，**不需要跑任何模型**：
> `python paper2_lm_design/paper_data_report.py` 直接用仓库自带的
> `paper-data/data.csv` 复算了论文二摘要里的全部关键数字，**7/8 项精确命中**
> （唯一对不上的是「49 条远距自由生成」，原因见报告）。

---

## 1. 论文一：全部实验与命令

### 1.1 实验 → 任务 → 论文规模 → 命令

| 论文位置 | 实验 | 论文规模（正文/Methods） | 任务名 | 仓库是否有现成程序 |
|:--|:--|:--|:--|:--|
| 图 2A-C | 自由幻觉 | 200 seeds × 30,000 步（A.3.1） | `free_hallucination` | ✅ `programs/free_hallucination.py` |
| 图 2D-F | 固定骨架设计 | 6 个 de novo target（1QYS/5L33/6D0T/6MRS/6W3W/6WVS）× ≥50 seeds（A.3.2） | `fixed_backbone` | ⚠️ 只有 `fixed_backbone_6mrs()` 且没实现论文权重 |
| 图 2G / S1A-C | 二级结构设计 | 3 个程序（全 α / 全 β / 混合）× 10 seeds（A.3.3） | `secondary_structure` | ✅ `programs/secondary_structure.py` |
| 图 2H / S1D | 单功能位点脚手架 | 5 个位点 × 1,000 seeds = 5,000 跑动（A.3.4） | `functional_site_scaffolding` | ⚠️ 只有 ACE2 |
| 图 3A-D / S2 | 单链对称 | K=3~8 × 总长 200/300/400 × 10 seeds = 180 跑动（A.3.5） | `symmetric_monomer` | ✅ 但长度硬编码 50/protomer |
| 图 3E | 同源寡聚体 | 4/6/8 聚体 × 10 seeds，总长 720（A.3.5） | `homo_oligomer` | ❌ 无 |
| 图 3C-D / 4C-D / S2D | 逆折叠 roundtrip | 1,000 个结构 × 10 条 ESM-IF1 采样（A.3.8） | `roundtrip.py` | ⚠️ 需自己写 |
| 图 4A-D / S3 | 双层对称寡聚体 | 顶层 2~4 × 底层 2~4 = 9 程序 × 10 seeds = 90 跑动（A.3.6） | `two_level_multimer` | ✅ `programs/symmetric_two_level_multimer.py` |
| 图 5A-B / S4A-C | 对称功能位点脚手架 | 3 程序 × 20 seeds = 60 跑动（A.3.9） | `symmetric_binding` | ⚠️ 只有 IL10 |
| 图 5C-F / S4D-E | 非对称层级组合 | 3 程序 × 10 seeds = 30 跑动（A.3.10） | `hierarchical_asymmetric` | ❌ 无 |
| 图 S2C / S3C | 结构新颖性 TM-score | 对 PDB 2022-08 全库穷举 TM-align（A.3.7） | — | ❌ 需外部 TM-align |
| Discussion | 湿实验验证 | 基因合成 + 表达 + SEC | — | ❌ 不可计算复现 |

**统一跑法**（`design_programs.py` 把所有任务做成了同一个接口）：

```bash
cd reproduce/paper1_programming
PY=/home/zzj/anaconda3/envs/ESM/bin/python

$PY design_programs.py --list-tasks                    # 看任务清单
$PY design_programs.py --check                         # 体检依赖
$PY design_programs.py --task free_hallucination --grid paper --dry-run    # 先看计划

# 冒烟（每 spec 2 seed / 200 步）
$PY design_programs.py --task free_hallucination --grid smoke --out-dir ../outputs/p1

# 论文规模
$PY design_programs.py --task symmetric_monomer --grid paper \
     --out-dir ../outputs/p1 --device cuda:0
```

### 1.2 论文一的超参数（全部来自 Methods，已写进脚本默认值）

| 参数 | 值 | 出处 |
|:--|:--|:--|
| 模拟退火步数 `M` | 30,000 | A.3.1 ~ A.3.10 各处 |
| `Tmax` / `Tmin` | 1 / 1e-4 | A.1.4 |
| 退火率 | `(Tmin/Tmax)^(1/M)` ≈ 0.99969 | A.1.4 公式 |
| 变异操作概率 | 替换 60% / 插入 20% / 删除 20% | A.1.4 |
| 氨基酸分布 | 均匀，且**排除半胱氨酸** | A.1.4 |
| 自由幻觉权重 | pTM=1, pLDDT=1, hydrophobics=1 | A.3.1 |
| 固定骨架权重 | dRMSD=2, cRMSD=1, pTM=1, pLDDT=1, hydrophobics=0.5 | A.3.2 |
| 二级结构权重 | SS=10, pTM=1, pLDDT=1, hydrophobics=1 | A.3.3 |
| 单功能位点脚手架 | cRMSD=2, dRMSD=2, surface_exposure=1, pTM=1, pLDDT=1, hydrophobics=1 | A.3.4 |
| 单链对称 | symmetry=1, pTM=1, pLDDT=1, hydrophobics=1 + 长度硬约束 | A.3.5 |
| 寡聚体 | symmetry=1, pTM=1, pLDDT=1, hydrophobics=1, globularity(**每个 terminal**)=0.1 | A.3.5 |
| 双层对称 | 所有项 =1（含 x2 上的 globularity） | A.3.6 |
| 对称位点脚手架 | cRMSD=10, dRMSD=10, pTM=1, pLDDT=1, rotational sym=1, surface exposure=1, hydrophobics=1 | A.3.9 |
| 层级非对称 | 所有项 =1 | A.3.10 |
| ESM-IF1 采样 | 10 条 / 温度 0.1 | A.3.8 |

> `--weights repo` 可切到「仓库自带程序的原始权重」（多数是全 1），用于对比作者
> 实际提交的代码与论文正文描述的差异。

### 1.3 论文一的算力估算（别被数字骗到）

每一次跑动 = 30,000 次 ESMFold 前向。按 ESMFold 单条 L≈100 序列在
A100/V100 上约 0.1~0.3 秒计：

| 任务 | 跑动数 | 折叠次数 | 估算 GPU 时（按 0.2 s/次） |
|:--|--:|--:|--:|
| 自由幻觉 | 200 | 6.0 M | ≈ 330 h |
| 固定骨架 | 300 | 9.0 M | ≈ 500 h |
| 功能位点脚手架 | 5,000 | 150 M | ≈ 8,300 h |
| 单链对称 | 180 | 5.4 M | ≈ 300 h |
| 双层对称 | 90 | 2.7 M | ≈ 150 h |
| 其余（二级结构 30 + 寡聚体 40 + 对称位点 60 + 层级非对称 30） | 160 | 4.8 M | ≈ 270 h |
| **合计** | **5,930** | **≈ 177.9 M** | **≈ 9,850 GPU 小时 ≈ 1.1 GPU·年** |

**所以请务必分层推进**：
1. `--grid smoke`（每 spec 2 seed × 200 步）先把每条链路跑通；
2. 挑 1~2 个任务用 `--seeds 10` 出可看的统计；
3. 只有真正需要复现的图才上 `--grid paper`，并且**按 spec 分批**跑（脚本自动跳过已完成项）。

在 CPU 上这个估算要再乘 50~200 倍，**等于不可完成**，请务必先解决 GPU。

### 1.4 论文一汇总

```bash
PY=/home/zzj/anaconda3/envs/ESM/bin/python
$PY aggregate_p1.py --root ../outputs/p1
# → ../outputs/p1/summary/P1_SUMMARY.md  （对照论文图 2B/2E/2F/2H/3B/4B/5B 的结论表）
#   p1_runs.csv     逐跑动明细
#   p1_by_spec.csv  按 task×spec 聚合
#   fig_*.png       箱线图 / 直方图 / roundtrip 散点
```

逆折叠 roundtrip（图 3C-D / 4C-D）：

```bash
$PY roundtrip.py --pdb-root ../outputs/p1 \
    --tasks symmetric_monomer two_level_multimer \
    --n-structures 1000 --num-samples 10 --temperature 0.1 \
    --device cuda:0 --out ../outputs/p1/roundtrip
# 加 --no-esmfold 只算 ESM-IF1 perplexity，省一半算力
# 结果按行追加到 roundtrip.jsonl，中断可续跑
```

---

## 2. 论文二：全部实验与命令

### 2.1 实验 → 命令

| 论文位置 | 实验 | 论文规模 | 命令 |
|:--|:--|:--|:--|
| 图 2A | 39 个 de novo target 的固定骨架设计 | 200 designs/target × 170,000 步（A.3.1） | `run_lm_design_batch.py --task fixedbb --pdb <ID> --seeds 0-199 --num-iter 170000` |
| 图 2B | 79 个设计的湿实验（SEC） | 6 target，见 A.6.2 | ❌ 湿实验；可用 `paper_data_report.py` 复算统计 |
| 图 2C / 2F | LM vs no-LM (AlphaFold) 对照 | 200 designs/target 各取 top-5 | ⚠️ no-LM 侧需外部 **ColabDesign** |
| 图 2D | 优化轨迹能量 vs RMSD | — | ✅ `trajectory.csv` 已记录能量；RMSD 需 oracle |
| 图 2E | LM 困惑度分布 | — | ✅ `metrics.json` 的 `lm_perplexity` |
| 图 2G | 远距设计的结构叠加 | — | ⚠️ 需 jackhmmer + AF DB（见下） |
| 图 3 / 4A-B | 自由生成（blocked Gibbs） | 25,000 条 × 170,000 步（A.3.3） | `run_lm_design_batch.py --task free_generation --seed 0-... --num-iter 170000` |
| 图 4C | pLDDT / pTM 分布 | 25k generations | ✅ `oracle=esmfold`（论文用 AlphaFold） |
| 图 4D / 4F-G | 与天然蛋白的序列/结构距离 | 25k × AlphaFold DB | ⚠️ 需下载 AF DB；部分可用 `analyze_novelty.py` |
| 表 S1 / 图 S2-S3 | 结构理解（contact 精度、perplexity 随 checkpoint） | 214 条天然 + 39 de novo | ⚠️ 需 ESM2 各 checkpoint |
| App A.2.2 | 结构投影层训练 | 15,051 条 PDB，10 epoch，bs 4，lr 1e-2 | ⚠️ 仓库只提供**已训练权重**（自动下载），训练脚本未开源 |
| App A.3.2 | no-LM 基线 | ColabDesign `3stage()` AfDesign | ❌ 外部仓库（commit e7bb3def） |
| App A.4.2-4.4 | Rosetta 过滤（溶解度/堆积/球状性） | PackStat>0.55、SSShapeComp>0.6、rel Rg<1.5、rel SASA<3、SAP≤0.4 | ❌ 需安装 Rosetta |
| App A.5.4 | motif 分析 | Foldseek `--alignment-type 1` | ❌ 需 Foldseek |
| App A.6 / A.7 | 276 条送检蛋白的湿实验 | 两轮，SEC S75 5/150 | ❌ 湿实验 |
| **摘要 / 图 2B / 2C 的统计** | **276 条蛋白的成功率与新颖性** | — | ✅ **`paper_data_report.py` 可直接复算，7/8 项精确命中** |

### 2.2 统一跑法

```bash
cd reproduce/paper2_lm_design
PY=/home/zzj/anaconda3/envs/ESM/bin/python

$PY run_lm_design_batch.py --check                       # 体检
$PY run_lm_design_batch.py --fetch-pdbs --pdb-dir ../outputs/p2/de_novo_targets   # 下 39 个 target

# 冒烟：2N2U 上 1 个 seed、300 步
$PY run_lm_design_batch.py --task fixedbb --pdb 2N2U --pdb-dir ../../examples/lm-design \
   --seeds 0 --num-iter 300 --out-dir ../outputs/p2

# 自由生成
$PY run_lm_design_batch.py --task free_generation --length 100 \
   --seeds 0 --num-iter 300 --out-dir ../outputs/p2

# 论文规模（单个 target 的 200 designs）
$PY run_lm_design_batch.py --task fixedbb --pdb 1QYS \
   --pdb-dir ../outputs/p2/de_novo_targets \
   --seeds 0-199 --num-iter 170000 --oracle esmfold \
   --device cuda:0 --out-dir ../outputs/p2
```

### 2.3 论文二的超参数（config.yaml + Methods，已写成脚本默认值）

| 参数 | 值 | 出处 |
|:--|:--|:--|
| MCMC 步数 | 170,000 | A.3.1 / A.3.3 |
| 温度调度 | 每 10,000 步 ×0.5，初值 8 → 终值 ≈6e-5 | A.3.1 |
| 能量权重 | `λ_p=3`（结构投影）, `λ_LM=2`（LM 负对数似然）, `λ_n=1`（n-gram KL） | A.3.1 |
| 每步提议 | 随机一个位置 + 均匀氨基酸替换（**禁止 Cys**） | A.3.1 |
| 接受率 | Metropolis，`α = min(1, exp(-ΔE/T))` | A.3.1 |
| 结构投影 | 660 维 attention → 18 个距离 bin，11,898 参数 | A.2.2 |
| 自由生成 | blocked Gibbs：先 `y~p(y|x)`（温度恒为 1），再 `x~p(x|y)`（3 步固定骨架协议） | A.3.3 |
| LM 模型 | ESM2 650M（`esm2_t33_650M_UR50D`） | A.2.1 |
| 单条耗时 | 固定骨架 L≈100 在 32GB Volta 上 **≈10 小时** | A.3.1 |

### 2.4 论文二的算力估算

论文规模：**39 target × 200 designs + 扩充池 9,060 + 自由生成 25,000 + 对照 1,600 ≈ 42,660 条跑动**，
每条 170,000 步；按论文自己给的单价 **10 GPU 小时/条**（32GB Volta、L≈100）
⇒ **≈ 4.2 × 10⁵ GPU 小时 ≈ 48 GPU·年**（逐项拆解见
[`RUNBOOK.md` 第 4.3 节](RUNBOOK.md#43-总算力推算)）。
这显然不是个人能重跑的规模。实际可行的复现策略：

1. **先跑 `paper_data_report.py`**（0 成本）——拿到论文实验数据的独立复算；
2. **`--num-iter 20000 --seeds 0-4`** 复现图 2D 的能量下降趋势与图 2E 的困惑度分布
   （这两个图只需要「趋势正确」，不需要 170k 步）；
3. 需要 图 2A 的 oracle RMSD 时，跑 `--seeds 0-19 --num-iter 170000` 单 target 即可看出量级。

### 2.5 序列新颖性（图 2G / 4F-G）

```bash
# 需要本地序列库（论文：UniRef90 2021_04，约 15GB）与 jackhmmer
$PY analyze_novelty.py --fasta-dir ../outputs/p2 \
    --db /data/uniref90.fasta --db-name uniref90_2021_04 \
    --purge ../../examples/lm-design/paper-data/uniref90_jackhmmer_purge_ids.txt \
            ../../examples/lm-design/paper-data/artificial_sequence_purge_ids.txt \
    --out ../outputs/p2/novelty
```

论文的三个非默认设置已固化在脚本里：`-n 1`、`--seed 0`、按 **best-domain E-value** 排序。
跳过 purge 列表会显著高估「新颖性」，脚本默认会把两个 purge 文件读进来。

### 2.6 论文二汇总

```bash
$PY aggregate_p2.py --root ../outputs/p2
# → ../outputs/p2/summary/P2_SUMMARY.md
#   p2_runs.csv / p2_by_config.csv
#   fig_energy_trajectory.png（图 2D 同款）
#   fig_lm_perplexity.png（图 2E 同款）
#   fig_oracle_rmsd.png（图 2A 同款，oracle 用 ESMFold）
```

---

## 3. 一键脚本

| 脚本 | 作用 |
|:--|:--|
| `00_setup_env.sh {check,base,esmfold,extern,all}` | 环境准备与体检 |
| **`fetch_datasets.sh {plan,core,pdb,weights,paperdata,seqdb,afdb,pdb-snapshot,processing,verify}`** | **数据集获取与组织**，详见 [`DATASETS.md`](DATASETS.md) |
| `run_all.sh` | 顶层：论文一 + 论文二 + 公开数据复算 + 总汇总 |
| `paper1_programming/run_p1.sh` | 论文一全部设计任务 + roundtrip + 汇总 |
| `paper2_lm_design/run_p2.sh` | 论文二全部设计任务 + 新颖性 + 汇总 |
| `aggregate_all.py` | 总汇总 → `outputs/REPRO_SUMMARY.md` |

```bash
# 数据集（先做这个）
bash fetch_datasets.sh plan     # 看清单
bash fetch_datasets.sh core     # 必需部分 ≈ 8.8 GB
bash fetch_datasets.sh verify   # 校验

```bash
# 最小验证（几十秒，不需要 GPU，不需要 openfold）
DRY_RUN=1 MODE=smoke bash run_all.sh                    # 先看计划
ONLY=paperdata bash run_all.sh                          # 只做论文数据复算（能真跑）

# 冒烟
MODE=smoke bash run_all.sh

# 论文规模
MODE=full OUT_ROOT=/data/repro bash run_all.sh

# 只做其中一块
ONLY=p1 MODE=smoke TASKS="free_hallucination symmetric_monomer" bash run_all.sh
ONLY=p2 MODE=full  PDBS="1QYS 6MRS" NUM_ITER=170000 ORACLE=esmfold bash run_all.sh

# 有本地 UniRef90 时顺带跑新颖性
ONLY=p2 MODE=smoke NOVELTY_DB=/data/uniref90.fasta bash run_all.sh
```

常用开关：`DRY_RUN=1`、`MODE=smoke|full`、`DEVICE=cuda:0|cpu`、`SEEDS`、`STEPS`、
`NUM_ITER`、`TASKS`、`PDBS`、`ALL_TARGETS=1`、`ORACLE=esmfold`、`WITH_ROUNDTRIP=1`、
`NOVELTY_DB`、`SKIP_AGG=1`。

所有脚本都做同一件事：**已完成的单条跑动自动跳过**，所以中断后重跑同一命令即可续跑。

---

## 4. 目录与产物

```
reproduce/
├── RUNBOOK.md                     ★ 环境无关的运行手册（安装 / 逐实验命令 / 论文设备与规模 / 排期）
├── DATASETS.md                    ★ 数据集获取与组织（来源 / 体积 / 目录规范 / 校验 / 踩坑）
├── README.md                      ← 本文件
├── 00_setup_env.sh                环境准备/体检
├── fetch_datasets.sh              数据集下载/清洗/校验
├── requirements-repro.txt         额外 pip 依赖
├── run_all.sh                     顶层一键
├── aggregate_all.py               总汇总
├── lib/common.sh                  公共库（日志/设备探测/dry-run/TMPDIR）
├── paper1_programming/
│   ├── design_programs.py         ★ 论文一全部设计任务的统一驱动
│   ├── roundtrip.py               ESM-IF1 逆折叠 roundtrip
│   ├── aggregate_p1.py            汇总 + 画图
│   └── run_p1.sh                  一键
├── paper2_lm_design/
│   ├── run_lm_design_batch.py     ★ 论文二多 seed 批量驱动
│   ├── analyze_novelty.py         jackhmmer 序列新颖性
│   ├── paper_data_report.py       ★ 论文二公开实验数据复算（能真跑）
│   ├── aggregate_p2.py            汇总 + 画图
│   └── run_p2.sh                  一键
└── outputs/                       （默认产物根，可用 OUT_ROOT 改）
    ├── REPRO_SUMMARY.md           总报告
    ├── p1/…                       论文一产物
    └── p2/…                       论文二产物
```

每条跑动固定落盘四件东西：

```
<task>/<spec>/seed<k>/result.json     完整记录（序列/能量/各约束值/pLDDT/pTM/耗时/权重口径）
                     design.pdb       ESMFold 预测结构
                     design.fasta     设计序列
                     status.json      状态，用于续跑判定
```

---

## 5. 与论文不一致的地方（务必知道，别当成 bug）

| # | 事项 | 说明 |
|:--|:--|:--|
| 1 | 论文一 4 个任务仓库**没有**程序 | `fixed_backbone`（除 6MRS）、`homo_oligomer`、`hierarchical_asymmetric` 的通用版、非 ACE2 位点脚手架，本工具包按 Methods 的权重与约束**等价实现**；`result.json` 里的 `weights_profile` 字段标明口径。**这不是作者代码**，数值会有差异。 |
| 2 | 论文一的 SS 约束权重 | 仓库 `secondary_structure.py` 未设权重（全 1），论文 A.3.3 要求 10。默认用 `--weights paper`（10），`--weights repo` 可切回作者代码口径。 |
| 3 | 论文一固定骨架 target | 论文用 1QYS/5L33/6D0T/6MRS/6W3W/6WVS；仓库只提供 6MRS。脚本从 RCSB 在线拉取。 |
| 4 | roundtrip 的结构来源 | 论文从 180 条轨迹里**均匀采样中间结构**；本工具包只能对**终态**结构做 roundtrip（因为驱动只落盘终态）。趋势可比，绝对值不宜逐点对齐。 |
| 5 | 论文二 oracle | 论文用 **AlphaFold**（5 pTM 模型 + Amber）；本工具包只能用 **ESMFold**。pLDDT/RMSD 绝对值不可直接比较。 |
| 6 | 论文二 `save_every`/轨迹 | 原脚本不落盘任何数据；本工具包通过包装 `calc_total_loss` 记录能量轨迹。**这是对原实现的运行时包装**，不改变采样过程。 |
| 7 | 论文二「49 条远距自由生成」 | `data.csv` 无法精确重建这个子集（它依赖 Fig 4D 的 AlphaFold DB 全库检索结果）。见 `PAPER_DATA_REPORT.md` 第 2 节。 |
| 8 | 尺寸/权重下载 | ESM2 650M 约 2.5 GB，ESM-IF1 约 1.4 GB，ESMFold 3B 约 12 GB，UniRef90 约 15 GB；首次运行前先确认磁盘与网络。 |

---

## 6. 快速排错

| 现象 | 原因 / 处理 |
|:--|:--|
| `ModuleNotFoundError: biotite / rich / hydra` | `bash 00_setup_env.sh base` |
| `ModuleNotFoundError: openfold` | 论文一必须有它：`bash 00_setup_env.sh esmfold`（要 CUDA） |
| `No space left on device` | `/tmp` 只有 10 MB：`export TMPDIR=/home/zzj/tmp`（脚本已自动设） |
| `CUDA OOM` | 降 `--num-iter`/批量；论文一降 ESMFold 的 chunk（在 `EsmFoldv1.load` 前 `model.set_chunk_size(64)`） |
| 论文一某条 spec 报错 | 看 `<spec>/seed<k>/traceback.txt`；常见是 RCSB 下载失败或 PDB 取链问题（重跑该命令即可续跑） |
| 论文二 `assert Path.cwd().name == 'lm-design'` | 必须由 `run_lm_design_batch.py` 启动（它会自动 `chdir`），不要直接 `python lm_design.py` |
| 结果全是 CPU 跑出来的、慢到不可用 | 装 CUDA 版 torch：`pip install torch --index-url https://download.pytorch.org/whl/cu124`（建议在**新环境**里做，别破坏现有 `ESM` 环境） |
| 想确认论文二数字对不对 | `python paper2_lm_design/paper_data_report.py`（不需要任何模型/GPU） |

---

## 7. 引用

- Hie, Candido, Lin, Kabeli, Rao, Smetanin, Sercu, Rives. *A high-level programming language for generative protein design.* bioRxiv 2022.12.21.521526.
- Verkuil\*, Kabeli\*, Du, Wicky, Milles, Dauparas, Baker, Ovchinnikov, Sercu, Rives. *Language models generalize beyond natural proteins.* bioRxiv 2022.12.21.521521.

代码许可证见仓库根 `LICENSE`（MIT）。论文二的部分数据（`paper-data/`）与
ESM Atlas 数据另有许可条款，使用前请查阅。
