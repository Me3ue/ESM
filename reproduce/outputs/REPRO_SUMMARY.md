# 两篇论文复现总汇总

| 论文 | 主题 | 复现工具包入口 |
| --- | --- | --- |
| *A high-level programming language for generative protein design* (Hie et al., 2022) | 可编程蛋白设计（能量/模拟退火） | `paper1_programming/run_p1.sh` |
| *Language models generalize beyond natural proteins* (Verkuil, Kabeli et al., 2022) | 语言模型生成式设计 | `paper2_lm_design/run_p2.sh` |

## 1. 完成度快照

| 指标 | 当前值 |
| --- | --- |
| 论文一：跑动总数 | 0 |
| 论文一：逆折叠 roundtrip 记录 | 0 |
| 论文二：跑动总数 | 0 |
| 论文二：新颖性检索记录 | 0 |

> 目前还没有任何跑动。先执行 `MODE=smoke bash run_all.sh` 打通流程，再上 `MODE=full`。

## 2. 论文一：实验 → 任务 → 完成度

| 任务 | 论文位置 | 论文规模 | 已完成 | 覆盖说明 |
| --- | --- | --- | --- | --- |
| `free_hallucination` | 图 2A-C | 200 seeds × 30k 步 | 0 | 完全覆盖 |
| `fixed_backbone` | 图 2D-F | 6 target × ≥50 seeds | 0 | 完全覆盖（权重按论文 Methods） |
| `secondary_structure` | 图 2G / S1 | 3 程序 × 10 seeds | 0 | 完全覆盖 |
| `functional_site_scaffolding` | 图 2H / S1D | 5 位点 × 1000 seeds | 0 | ACE2 用仓库实现；其余 4 个位点为等价实现 |
| `symmetric_monomer` | 图 3A-D / S2 | 6 对称 × 3 长度 × 10 seeds | 0 | 完全覆盖 |
| `homo_oligomer` | 图 3E | 4/6/8 聚体 × 10 seeds | 0 | 等价实现（仓库无此程序） |
| `two_level_multimer` | 图 4 / S3 | 9 程序 × 10 seeds | 0 | 完全覆盖（仓库自带） |
| `symmetric_binding` | 图 5A-B / S4A-C | 3 程序 × 20 seeds | 0 | IL10 用仓库实现；ACE2 需换模板 |
| `hierarchical_asymmetric` | 图 5C-F / S4D-E | 3 程序 × 10 seeds | 0 | 等价实现（仓库无此程序） |
| `roundtrip` | 图 3C-D / 4C-D | 1000 结构 × 10 采样 | 0 | 覆盖终态结构（非中间结构） |

## 3. 论文二：实验 → 任务 → 完成度

| 任务 | 论文位置 | 论文规模 | 已完成 | 覆盖说明 |
| --- | --- | --- | --- | --- |
| `fixedbb` | 图 2A-F | 39 target × 200 designs × 170k 步 | 0 | 流程覆盖；oracle 用 ESMFold 代 AlphaFold |
| `free_generation` | 图 3 / 4A-D | 25,000 条 × 170k 步 | 0 | 流程覆盖 |
| `novelty` | 图 2G / 4F-G | jackhmmer vs UniRef90 | 0 | 完全覆盖（需自备序列库） |
| `paper_data` | 摘要 + 图 2B/2C | 276 条送检蛋白 | 0 | 完全覆盖（7/8 项精确复算） |

## 4. 产物索引

| 产物 | 路径 |
| --- | --- |
| 论文一汇总报告 | `/home/zzj/protein/esm/reproduce/outputs/p1/summary/P1_SUMMARY.md` |
| 论文一逐跑动明细 | `/home/zzj/protein/esm/reproduce/outputs/p1/summary/p1_runs.csv` |
| 论文一逐 spec 聚合 | `/home/zzj/protein/esm/reproduce/outputs/p1/summary/p1_by_spec.csv` |
| 论文二汇总报告 | `/home/zzj/protein/esm/reproduce/outputs/p2/summary/P2_SUMMARY.md` |
| 论文二逐跑动明细 | `/home/zzj/protein/esm/reproduce/outputs/p2/summary/p2_runs.csv` |
| 论文二公开数据复算 | `/home/zzj/protein/esm/reproduce/outputs/p2/paper_data_report/PAPER_DATA_REPORT.md` |
| 论文二新颖性 | `/home/zzj/protein/esm/reproduce/outputs/p2/novelty/NOVELTY.md` |

## 5. 无法用代码完成的部分

| 论文 | 内容 | 原因 |
| --- | --- | --- |
| 一 | Discussion 的实验验证、图 2E/2F 之外的湿实验 | 需要基因合成与体外实验 |
| 一 | 图 2C ssAF2 pLDDT | 需要单序列 AlphaFold2 |
| 一 | 图 S2C/S3C 结构新颖性 TM-score | 需要 PDB 2022-08 全库 + TM-align |
| 二 | 图 2B/2F/3C 的 SEC 曲线与产率 | 湿实验；原始数据在 `data.hdf5` |
| 二 | 图 2C 的 no-LM 基线 | 需要外部 ColabDesign |
| 二 | App A.4 的 Rosetta 过滤指标 | 需要安装 Rosetta |

---

生成方式：`python aggregate_all.py --root /home/zzj/protein/esm/reproduce/outputs`
