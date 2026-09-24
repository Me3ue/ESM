#!/usr/bin/env python3
"""
总汇总：把两篇论文的所有复现产物汇到一张总表 + 一份总报告。

它会：
  1. 扫描 <root>/p1 与 <root>/p2，统计各任务完成了多少跑动；
  2. 调用 aggregate_p1.py / aggregate_p2.py 生成各自的报告与 CSV；
  3. 跑 paper_data_report.py（零成本，复算论文二公开实验数据）；
  4. 输出 <root>/REPRO_SUMMARY.md —— 一张总览表 + 完成度 + 缺口清单。

用法
----
  python aggregate_all.py                      # 默认 <repo>/reproduce/outputs
  python aggregate_all.py --root outputs --no-sub    # 不递归调用子汇总
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PY = sys.executable


# 论文 → 实验 → 产出 的权威映射（README 也引用同一张表）
PAPER1_TASKS: List[Tuple[str, str, str, str, str]] = [
    # task, 论文图, 论文规模, 产物特征, 覆盖情况
    ("free_hallucination", "图 2A-C", "200 seeds × 30k 步", "result.json", "完全覆盖"),
    ("fixed_backbone", "图 2D-F", "6 target × ≥50 seeds", "result.json", "完全覆盖（权重按论文 Methods）"),
    ("secondary_structure", "图 2G / S1", "3 程序 × 10 seeds", "result.json", "完全覆盖"),
    ("functional_site_scaffolding", "图 2H / S1D", "5 位点 × 1000 seeds", "result.json",
     "ACE2 用仓库实现；其余 4 个位点为等价实现"),
    ("symmetric_monomer", "图 3A-D / S2", "6 对称 × 3 长度 × 10 seeds", "result.json", "完全覆盖"),
    ("homo_oligomer", "图 3E", "4/6/8 聚体 × 10 seeds", "result.json", "等价实现（仓库无此程序）"),
    ("two_level_multimer", "图 4 / S3", "9 程序 × 10 seeds", "result.json", "完全覆盖（仓库自带）"),
    ("symmetric_binding", "图 5A-B / S4A-C", "3 程序 × 20 seeds", "result.json",
     "IL10 用仓库实现；ACE2 需换模板"),
    ("hierarchical_asymmetric", "图 5C-F / S4D-E", "3 程序 × 10 seeds", "result.json",
     "等价实现（仓库无此程序）"),
    ("roundtrip", "图 3C-D / 4C-D", "1000 结构 × 10 采样", "roundtrip.jsonl", "覆盖终态结构（非中间结构）"),
]

PAPER2_TASKS: List[Tuple[str, str, str, str, str]] = [
    ("fixedbb", "图 2A-F", "39 target × 200 designs × 170k 步", "metrics.json",
     "流程覆盖；oracle 用 ESMFold 代 AlphaFold"),
    ("free_generation", "图 3 / 4A-D", "25,000 条 × 170k 步", "metrics.json", "流程覆盖"),
    ("novelty", "图 2G / 4F-G", "jackhmmer vs UniRef90", "novelty.csv", "完全覆盖（需自备序列库）"),
    ("paper_data", "摘要 + 图 2B/2C", "276 条送检蛋白", "PAPER_DATA_REPORT.md",
     "完全覆盖（7/8 项精确复算）"),
]


def count_files(root: Path, pattern: str) -> int:
    return len(list(root.rglob(pattern))) if root.is_dir() else 0


def run(cmd: List[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, check=False).returncode
    except Exception as exc:
        print(f"  [失败] {exc!r}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="两篇论文复现总汇总")
    ap.add_argument("--root", type=Path, default=HERE / "outputs")
    ap.add_argument("--no-sub", action="store_true", help="不调用子汇总脚本")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    p1 = root / "p1"
    p2 = root / "p2"
    root.mkdir(parents=True, exist_ok=True)

    # ---- 1) 统计
    n_p1_runs = count_files(p1, "result.json")
    n_p1_rt = sum(
        1 for f in p1.rglob("roundtrip.jsonl")
        for _ in f.read_text(encoding="utf-8").splitlines() if _.strip()
    ) if p1.is_dir() else 0
    n_p2_runs = count_files(p2, "metrics.json")
    n_p2_nov = 0
    for f in (p2.rglob("novelty.csv") if p2.is_dir() else []):
        n_p2_nov += max(0, len(f.read_text(encoding="utf-8").splitlines()) - 1)

    p1_by_task: Dict[str, int] = defaultdict(int)
    for f in (p1.rglob("result.json") if p1.is_dir() else []):
        try:
            p1_by_task[json.loads(f.read_text(encoding="utf-8")).get("task", "?")] += 1
        except Exception:
            pass
    p2_by_task: Dict[str, int] = defaultdict(int)
    for f in (p2.rglob("metrics.json") if p2.is_dir() else []):
        try:
            p2_by_task[json.loads(f.read_text(encoding="utf-8")).get("task", "?")] += 1
        except Exception:
            pass

    # ---- 2) 子汇总
    if not args.no_sub:
        print("=" * 78)
        print("调用子汇总")
        print("=" * 78)
        run([PY, str(HERE / "paper1_programming" / "aggregate_p1.py"), "--root", str(p1)]
            + (["--no-plot"] if args.no_plot else []))
        run([PY, str(HERE / "paper2_lm_design" / "aggregate_p2.py"), "--root", str(p2)]
            + (["--no-plot"] if args.no_plot else []))
        run([PY, str(HERE / "paper2_lm_design" / "paper_data_report.py"),
             "--out", str(p2 / "paper_data_report")])

    # ---- 3) 总报告
    md: List[str] = []
    md.append("# 两篇论文复现总汇总")
    md.append("")
    md.append("| 论文 | 主题 | 复现工具包入口 |")
    md.append("| --- | --- | --- |")
    md.append("| *A high-level programming language for generative protein design* (Hie et al., 2022) "
              "| 可编程蛋白设计（能量/模拟退火） | `paper1_programming/run_p1.sh` |")
    md.append("| *Language models generalize beyond natural proteins* (Verkuil, Kabeli et al., 2022) "
              "| 语言模型生成式设计 | `paper2_lm_design/run_p2.sh` |")
    md.append("")
    md.append("## 1. 完成度快照")
    md.append("")
    md.append("| 指标 | 当前值 |")
    md.append("| --- | --- |")
    md.append(f"| 论文一：跑动总数 | {n_p1_runs} |")
    md.append(f"| 论文一：逆折叠 roundtrip 记录 | {n_p1_rt} |")
    md.append(f"| 论文二：跑动总数 | {n_p2_runs} |")
    md.append(f"| 论文二：新颖性检索记录 | {n_p2_nov} |")
    md.append("")
    if not (n_p1_runs or n_p2_runs):
        md.append("> 目前还没有任何跑动。先执行 `MODE=smoke bash run_all.sh` 打通流程，"
                  "再上 `MODE=full`。")
        md.append("")

    def table(rows: List[Tuple[str, str, str, str, str]], counts: Dict[str, int], label: str) -> None:
        md.append(f"## {label}")
        md.append("")
        md.append("| 任务 | 论文位置 | 论文规模 | 已完成 | 覆盖说明 |")
        md.append("| --- | --- | --- | --- | --- |")
        for task, fig, scale, _art, note in rows:
            md.append(f"| `{task}` | {fig} | {scale} | {counts.get(task, 0)} | {note} |")
        md.append("")

    table(PAPER1_TASKS, p1_by_task, "2. 论文一：实验 → 任务 → 完成度")
    table(PAPER2_TASKS, p2_by_task, "3. 论文二：实验 → 任务 → 完成度")

    md.append("## 4. 产物索引")
    md.append("")
    md.append("| 产物 | 路径 |")
    md.append("| --- | --- |")
    md.append(f"| 论文一汇总报告 | `{p1}/summary/P1_SUMMARY.md` |")
    md.append(f"| 论文一逐跑动明细 | `{p1}/summary/p1_runs.csv` |")
    md.append(f"| 论文一逐 spec 聚合 | `{p1}/summary/p1_by_spec.csv` |")
    md.append(f"| 论文二汇总报告 | `{p2}/summary/P2_SUMMARY.md` |")
    md.append(f"| 论文二逐跑动明细 | `{p2}/summary/p2_runs.csv` |")
    md.append(f"| 论文二公开数据复算 | `{p2}/paper_data_report/PAPER_DATA_REPORT.md` |")
    md.append(f"| 论文二新颖性 | `{p2}/novelty/NOVELTY.md` |")
    md.append("")
    md.append("## 5. 无法用代码完成的部分")
    md.append("")
    md.append("| 论文 | 内容 | 原因 |")
    md.append("| --- | --- | --- |")
    md.append("| 一 | Discussion 的实验验证、图 2E/2F 之外的湿实验 | 需要基因合成与体外实验 |")
    md.append("| 一 | 图 2C ssAF2 pLDDT | 需要单序列 AlphaFold2 |")
    md.append("| 一 | 图 S2C/S3C 结构新颖性 TM-score | 需要 PDB 2022-08 全库 + TM-align |")
    md.append("| 二 | 图 2B/2F/3C 的 SEC 曲线与产率 | 湿实验；原始数据在 `data.hdf5` |")
    md.append("| 二 | 图 2C 的 no-LM 基线 | 需要外部 ColabDesign |")
    md.append("| 二 | App A.4 的 Rosetta 过滤指标 | 需要安装 Rosetta |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("生成方式：`python aggregate_all.py --root " + str(root) + "`")
    md.append("")

    out = root / "REPRO_SUMMARY.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("=" * 78)
    print(f"已写出：{out}")
    print(f"论文一跑动 {n_p1_runs} 条；论文二跑动 {n_p2_runs} 条")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
