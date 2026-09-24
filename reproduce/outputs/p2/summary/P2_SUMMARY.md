# 论文二复现结果汇总

论文：*Language models generalize beyond natural proteins*（Verkuil, Kabeli et al., 2022）

- 本次跑动数：**0**，涉及 0 个 (task, tag) 组合
- 新颖性检索：**尚未运行**（`analyze_novelty.py`，需要 jackhmmer + 序列库）

> ⚠️ 本工具包的结构 oracle 是 **ESMFold**，而论文用的是 **AlphaFold**（5 个 pTM 模型选最优 + Amber 松弛，App A.4.1）。
> 两者的 RMSD / pLDDT 绝对值不可直接比较，只能看相对趋势。

## 1. 逐配置结果

| task | tag | n | 长度 | LM 困惑度 中位 ↓ | 困惑度最小 | oracle CA-RMSD 中位 ↓ | 终态能量 | 单条耗时(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

论文对应规模：固定骨架 **39 个 de novo target × 200 designs**、自由生成 **25,000 条**，每条 **170,000 步** MCMC（App A.3.1 / A.3.3）。

## 2. 与论文公开实验数据的对照

- 仓库 `examples/lm-design/paper-data/data.csv` 共 276 条送检蛋白，其中 LM 设计 228 条，成功 152 条 (67%)，论文摘要为 152/228 (67%)。
- 论文这一部分的完整复算见 `paper_data_report.py` 产出的 `PAPER_DATA_REPORT.md`（默认在 `<root>/paper_data_report/`）。

## 3. 本工具包不覆盖的部分（需要外部资源）

| 论文内容 | 缺口 | 建议做法 |
| --- | --- | --- |
| App A.4.1 AlphaFold oracle（图 2A/3C） | 需要 AF2 + 5 个 pTM 模型 + Amber | ColabFold batch，或本地 AF2；结果存 CSV 后并入本汇总 |
| App A.3.2 no-LM 基线（图 2C/2F） | 需要 ColabDesign `3stage()` AfDesign | 外部仓库，`commit e7bb3def` |
| App A.4.2/4.3/4.4 溶解度/堆积/球状性过滤 | 需要 Rosetta（PackStat、SSShapeComplementarity、TotalSasa、SAP） | 装 Rosetta 后按阈值过滤：packing>0.55、shape comp>0.6、rel Rg<1.5、rel SASA<3、SAP≤0.4 |
| App A.5.2 图 4D（vs AlphaFold DB 全库） | 需要 AlphaFold DB 序列 + 结构 | 下载 AF DB fasta + `AF-<UniProtID>-F1-model_v3.pdb` |
| App A.5.4 motif 分析（图 3D-E） | 需要 Foldseek（`--alignment-type 1`） | `conda install -c bioconda foldseek` |
| App A.7 湿实验（SEC、SEC 曲线图） | 基因合成 + 表达 + 纯化 | 不可计算复现；论文 SEC 原始曲线在 `data.hdf5` |

## 4. 一键补齐的建议顺序

```bash
bash run_p2.sh                          # 设计 + 汇总（MODE=smoke 先打通）
python analyze_novelty.py --db <uniref90.fasta> --db-name uniref90_2021_04 \
    --out outputs/p2/novelty            # 序列新颖性（图 2G / 4F-G）
python aggregate_p2.py --root outputs/p2   # 把新颖性并入总报告
```
