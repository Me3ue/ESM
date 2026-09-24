# 论文一复现结果汇总

论文：*A high-level programming language for generative protein design*（Hie et al., 2022）

- 产物根目录：`/home/zzj/protein/esm/reproduce/outputs/p1`
- 成功落盘的跑动总数：**0**，涉及 **0** 个 spec、**0** 个任务
- 逆折叠 roundtrip：**尚未运行**（见 `roundtrip.py`）

> 判定口径取自论文正文：pLDDT > 0.7 为高置信；固定骨架设计 RMSD < 1.6 Å；位点脚手架 RMSD 红线 2 Å；结构新颖性 TM-score 红线 0.6。

## 1. 任务总览

| 任务 | 论文位置 | spec 数 | 跑动数 | pLDDT 中位 | pLDDT>0.7 占比 |
| --- | --- | --- | --- | --- | --- |

## 11. 逆折叠 roundtrip（图 3C-D / 4C-D / S2D）

尚未运行。命令：

```bash
python roundtrip.py --pdb-root <p1 产物根> --tasks symmetric_monomer two_level_multimer \
    --n-structures 1000 --num-samples 10 --temperature 0.1 --out <root>/roundtrip
```

## 12. 本工具包不覆盖的部分（需要外部资源）

| 论文内容 | 缺口 | 建议做法 |
| --- | --- | --- |
| 图 2C ssAF2 pLDDT | 需要单序列 AlphaFold2 | ColabFold `--single-sequence` 或 AF2 官方代码 |
| 图 S2C / S3C 结构新颖性 | 需要 TM-align + PDB 2022-08 全库（>10 万结构） | 下载 PDB 后 `tm-align -byresi` 穷举；也可用 Foldseek 预筛 |
| 图 S2D ProteinMPNN roundtrip | 需要 ProteinMPNN 外部仓库 | `examples/inverse_folding` 里有 IF1；MPNN 需另装 |
| Discussion 的实验验证 | 湿实验（基因合成 + 表达 + SEC） | 不可计算复现，只能送样 |
| 图 2E/2F 的“50 or more seeds” | 本工具包 paper 网格默认 50 seeds/target | `--seeds 100` 可加大 |

---

生成方式：`python aggregate_p1.py --root /home/zzj/protein/esm/reproduce/outputs/p1`