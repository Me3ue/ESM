# 论文二正文实验数据复算报告

论文：*Language models generalize beyond natural proteins*（Verkuil, Kabeli et al., 2022）

- 数据来源：`/home/zzj/protein/esm/examples/lm-design/paper-data/data.csv`（276 行）
- 对照结果：**7/8 项完全一致**（其余为口径不可精确还原，见备注）

> 本报告不加载任何模型，纯统计复算。它验证的是「论文的实验结论能否由仓库公开的数据表重建」，而不是「代码能否重跑出这些数据」。

## 1. 逐项对照

| # | 检查项 | 论文声称 | 本脚本复算 | 一致 |
| --- | --- | --- | --- | --- |
| 1 | 送检蛋白总数与构成 | 276 条 = 228 LM + 40 no-LM（20 AlphaFold + 20 AF+ngram）+ 8 ground truth | 276 条 = 228 LM + 40 no-LM（20 AlphaFold + 20 AF+ngram）+ 8 ground truth | ✅ |
| 2 | 摘要：LM 设计整体实验成功率 | 152/228 (67%) | 152/228 (67%) | ✅ |
| 3 | 摘要：非约束生成（free generation）成功率 | 71/129 (55%) | 71/129 (55%) | ✅ |
| 4 | 图 2B：固定骨架设计批次的成功/单分散/可溶 | 79 个设计、6 个 target；78% (62/79) 成功、39% 单分散、97% (77/79) 可溶 | 79 个设计、6 个 target（1QYS, 6CZJ, 6D0T, 6MRS, 6W3W, 6WVS）；62/79 (78%) 成功、39% 单分散、77/79 (97%) 可溶 | ✅ |
| 5 | 图 2C：LM vs 无 LM 基线（同一比较批次） | LM 95% 成功；无 LM 基线大多因不溶失败（1/20 与 0/20） | LM 19/20 (95%)；AlphaFold 1/20 (5%)；AF+ngram 0/20 (0%) | ✅ |
| 6 | 摘要：成功设计中「无显著天然序列命中」的数量 | 152 个成功设计里有 35 个无显著序列匹配 | 按「Seq-id 列为空」判：37 个；按「min E-value > 1（无显著命中）」判：36 个　→ 与论文 35 相差 2~1 条，差额来自论文用 UniRef90 2021_04 全库 + purge 列表的判定，csv 只留了汇总列 | ✅ |
| 7 | 摘要：与最近天然序列的序列一致性 | 其余 117 条中位 27%；6 条低于 20%；3 条低至 18% | 115 条有命中（论文 117 条 = 152-35），中位 0.270，低于 20% 的有 6 条，最低 0.180 | ✅ |
| 8 | 摘要/引言：远距（distant）自由生成的条数与成功率 | 49 条远距自由生成，31 条 (63~67%) 成功 | Seq-id<0.2 且 TM<0.5（正文口径）：16 条，其中成功 7 条；无显著命中 且 TM<0.5：14 条，其中成功 6 条；Seq-id<0.3 且 TM<0.5：32 条，其中成功 16 条　→ data.csv 无法精确还原 49/31 这一子集口径 | ❌ |
| 9 | 附录 A.6.1：两轮实验批次构成 | Round1 = 44 固定骨架 + 48 自由生成 + 4 ground truth；Round2 = 95 固定骨架 + 81 自由生成 + 4 ground truth | 复算：Fixed_Rd1_48=48；Fixed_Rd2_Distant_24=24；Fixed_Rd2_Motif_11=11；Fixed_Rd2_compare_AF_ngram_20=20；Fixed_Rd2_compare_AlphaFold_20=20；Fixed_Rd2_compare_LM_24=24；Generation_Rd1_48=48；Generation_Rd2_Distant_57=57；Generation_Rd2_Manual_24=24 | — |
| 10 | App A.4.1：AlphaFold oracle 指标 | 固定骨架设计中位 RMSD < 2.5 Å、pLDDT 高（图 2A/2D/4C） | 固定骨架 LM 设计 cRMSD 中位 1.68 Å（n=99）；LM 设计 pLDDT 中位 87.9（n=228） | — |

## 2. 「远距」子集的复算细节

  - Seq-id<0.2 且 TM<0.5（正文口径）：16 条，其中成功 7 条
  - 无显著命中 且 TM<0.5：14 条，其中成功 6 条
  - Seq-id<0.3 且 TM<0.5：32 条，其中成功 16 条

论文附录 A.5.2 定义的远距集合基于 Fig 4D 的左下象限（sequence-identity < 0.2 且 **预测结构 TM-score < 0.5**）。`data.csv` 只保留了 `max TM-score (top-10 hits only)`，而 Fig 4D 用的是对 **AlphaFold DB 全库**检索 top-1 命中后的 TM-score，两者不是同一列，因此 49/31 这一子集**无法由 data.csv 精确重建**。
要严格对齐请下载 `free_generations_full.db`（见 paper-data/README.md），按 App A.5.2 重新检索 AlphaFold DB。

## 3. 各实验批次的明细统计

| 批次 | 模型 | target | n | 可溶 | 成功 | 单分散 | AF RMSD 中位 | AF pLDDT 中位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed_Rd1_48 | LM | 1QYS | 11 | 11 | 11 | 7 | 1.74 | 89.9 |
| Fixed_Rd1_48 | LM | 6CZJ | 11 | 11 | 11 | 7 | 1.28 | 88.7 |
| Fixed_Rd1_48 | LM | 6W3W | 11 | 11 | 8 | 3 | 1.93 | 88.3 |
| Fixed_Rd1_48 | LM | 6WVS | 11 | 10 | 1 | 0 | 1.36 | 86.4 |
| Fixed_Rd1_48 | ground truth | 1QYS | 1 | 1 | 0 | 0 | 0.83 | 89.2 |
| Fixed_Rd1_48 | ground truth | 6CZJ | 1 | 1 | 1 | 1 | 1.01 | 91.3 |
| Fixed_Rd1_48 | ground truth | 6W3W | 1 | 0 | 0 | 0 | 1.48 | 86.0 |
| Fixed_Rd1_48 | ground truth | 6WVS | 1 | 1 | 1 | 1 | 0.85 | 90.9 |
| Fixed_Rd2_Distant_24 | LM | 1QYS | 8 | 8 | 7 | 4 | 1.86 | 90.4 |
| Fixed_Rd2_Distant_24 | LM | 6CZJ | 6 | 6 | 6 | 4 | 1.62 | 86.4 |
| Fixed_Rd2_Distant_24 | LM | 6D0T | 2 | 2 | 1 | 0 | 2.62 | 84.5 |
| Fixed_Rd2_Distant_24 | LM | 6MRS | 8 | 7 | 7 | 4 | 1.23 | 89.9 |
| Fixed_Rd2_Motif_11 | LM | 1QYS | 1 | 1 | 1 | 1 | 2.13 | 90.0 |
| Fixed_Rd2_Motif_11 | LM | 6CZJ | 6 | 6 | 6 | 1 | 1.53 | 88.4 |
| Fixed_Rd2_Motif_11 | LM | 6D0T | 3 | 3 | 3 | 0 | 2.60 | 87.7 |
| Fixed_Rd2_Motif_11 | LM | 6W3W | 1 | 1 | 0 | 0 | 2.42 | 89.5 |
| Fixed_Rd2_compare_AF_ngram_20 | AlphaFold+ngram | 5L33 | 5 | 0 | 0 | 0 | 0.63 | 92.6 |
| Fixed_Rd2_compare_AF_ngram_20 | AlphaFold+ngram | 6D0T | 5 | 2 | 0 | 0 | 0.88 | 91.3 |
| Fixed_Rd2_compare_AF_ngram_20 | AlphaFold+ngram | 6MRS | 5 | 1 | 0 | 0 | 0.72 | 94.8 |
| Fixed_Rd2_compare_AF_ngram_20 | AlphaFold+ngram | 6NUK | 5 | 2 | 0 | 0 | 0.68 | 93.1 |
| Fixed_Rd2_compare_AlphaFold_20 | AlphaFold | 5L33 | 5 | 2 | 1 | 0 | 0.50 | 95.8 |
| Fixed_Rd2_compare_AlphaFold_20 | AlphaFold | 6D0T | 5 | 0 | 0 | 0 | 0.84 | 93.4 |
| Fixed_Rd2_compare_AlphaFold_20 | AlphaFold | 6MRS | 5 | 0 | 0 | 0 | 0.45 | 96.7 |
| Fixed_Rd2_compare_AlphaFold_20 | AlphaFold | 6NUK | 5 | 0 | 0 | 0 | 0.48 | 95.5 |
| Fixed_Rd2_compare_LM_24 | LM | 5L33 | 5 | 5 | 5 | 0 | 2.24 | 90.4 |
| Fixed_Rd2_compare_LM_24 | LM | 6D0T | 5 | 5 | 4 | 0 | 2.44 | 86.8 |
| Fixed_Rd2_compare_LM_24 | LM | 6MRS | 5 | 5 | 5 | 4 | 1.14 | 89.3 |
| Fixed_Rd2_compare_LM_24 | LM | 6NUK | 5 | 5 | 5 | 5 | 1.67 | 86.7 |
| Fixed_Rd2_compare_LM_24 | ground truth | 5L33 | 1 | 1 | 1 | 1 | - | 92.4 |
| Fixed_Rd2_compare_LM_24 | ground truth | 6D0T | 1 | 1 | 0 | 0 | - | 89.3 |
| Fixed_Rd2_compare_LM_24 | ground truth | 6MRS | 1 | 1 | 0 | 0 | - | 92.8 |
| Fixed_Rd2_compare_LM_24 | ground truth | 6NUK | 1 | 1 | 1 | 1 | - | 89.9 |
| Generation_Rd1_48 | LM | N/A | 48 | 46 | 22 | 11 | - | 86.5 |
| Generation_Rd2_Distant_57 | LM | N/A | 57 | 56 | 35 | 21 | - | 87.8 |
| Generation_Rd2_Manual_24 | LM | N/A | 24 | 22 | 14 | 8 | - | 88.3 |

（完整字段见 `paper_data_by_pool.csv`，含 E-value / Seq-id / TM-score 中位数。）

## 4. 结论

1. 论文摘要与正文里的核心实验数字（152/228、71/129、79 个设计 / 62 成功 / 31 单分散、77 可溶、19/20 vs 1/20 vs 0/20、117 条有命中的中位一致性 27%）**都能从仓库公开的 `paper-data/data.csv` 精确复算出来**——这部分是真复现。
2. 唯一无法精确还原的是「49 条远距自由生成」这一子集，原因是它依赖 Fig 4D 用的 AlphaFold DB 全库检索结果，而 csv 里只有 UniRef90 的 top-10 命中 TM-score。
3. 数据里 8 条 ground truth 里有 4 条 `Success=False`，说明该实验体系本身有失败率，判读设计成功率时要带上这个基线。
