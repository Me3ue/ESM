#!/usr/bin/env python3
"""
论文一：把 design_programs.py 的产物汇总成可引用的表格与图。

纯后处理：只读 result.json / roundtrip.jsonl / pdb，不加载任何模型，可随时重跑。

产出（默认写到 <root>/summary/）：
  p1_runs.csv        每次跑动一行（task, spec, seed, steps, plddt, ptm, energy, cRMSD, dRMSD, seconds）
  p1_by_spec.csv     按 task×spec 聚合（n / mean / std / median / min / max）
  P1_SUMMARY.md      对照论文正文图示的结论表
  fig_plddt_by_spec.png     图 2E / S2A / S3A 同款：每个 spec 的 pLDDT 箱线图
  fig_crmsd_by_target.png   图 2F / S1D 同款：每个 target 的 RMSD 箱线图
  fig_roundtrip.png         图 3C-D / 4C-D 同款：pLDDT–roundtrip RMSD、perplexity–RMSD 散点
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# matplotlib 需要可写配置目录（容器/受限环境下 ~/.config 可能不可写）
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplconfig"))

# 论文正文里各实验的判定阈值
PLDDT_GOOD = 0.7        # 论文：pLDDT > 0.7 记为高置信
RMSD_FIXEDBB = 1.6      # 正文：固定骨架设计 RMSD < 1.6 Å
RMSD_FIXEDBB_BOX = 2.5  # 图 2F 红线：RMSD = 2.5 Å
RMSD_SITE = 2.0         # 图 2H 红线：位点 RMSD = 2 Å
TM_NOVEL = 0.6          # 图 S2C / S3C 红线


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def term_values(res: Dict[str, Any]) -> Dict[str, float]:
    """把 energy_terms 里各约束的「裸值」抽出来，方便做论文口径的判据。

    注意约束函数返回的是「越小越好」的加权项，例如 MaximizePLDDT 返回 1-pLDDT。
    这里同时给出两种口径：
      *_raw   : 约束函数原始返回值
      *_meas  : 还原成物理量（pLDDT、pTM、RMSD…），便于和论文图对比
    """
    out: Dict[str, float] = {}
    for row in res.get("energy_terms") or []:
        name = str(row.get("name", "")).rsplit(":", 1)[-1]
        val = float(row.get("value", float("nan")))
        out[f"{name}_raw"] = val
        if name == "MaximizePLDDT":
            out["plddt_term"] = 1.0 - val
        elif name == "MaximizePTM":
            out["ptm_term"] = 1.0 - val
        elif name == "MinimizeCRmsd":
            out["crmsd"] = val
        elif name == "MinimizeDRmsd":
            out["drmsd"] = val
        elif name == "MinimizeSurfaceHydrophobics":
            out["hydrophobics"] = val
        elif name == "MaximizeSurfaceExposure":
            out["surface_exposure"] = 1.0 - val
        elif name == "MaximizeGlobularity":
            out["globularity"] = 1.0 - val
        elif name == "SymmetryRing":
            out["symmetry_var"] = val
        elif name == "MatchSecondaryStructure":
            out["ss_mismatch"] = val
            # 一个程序里可能有多个 SS 约束（图 2G 是两个子序列各一个），
            # 用节点前缀顺序记录下来，避免被后面的覆盖掉。
            out.setdefault("ss_mismatches_by_node", {})[str(row.get("name", ""))] = 1.0 - val
    return out


def load_runs(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("result.json")):
        try:
            res = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        res["_path"] = str(p)
        res.update(term_values(res))
        rows.append(res)
    return rows


def load_roundtrip(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("roundtrip.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def agg(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": st.fmean(vals),
        "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
        "median": st.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def group_by_spec(runs: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    g: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        g[(r.get("task", "?"), r.get("spec", "?"))].append(r)
    return g


def fmt(d: Dict[str, float], key: str = "mean", nd: int = 4) -> str:
    if d.get("n", 0) == 0 or key not in d:
        return "—"
    return f"{d[key]:.{nd}f}"


def frac(values: List[Optional[float]], pred) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "—"
    k = sum(1 for v in vals if pred(v))
    return f"{k}/{len(vals)} ({100.0 * k / len(vals):.0f}%)"


# --------------------------------------------------------------------------
# CSV 输出
# --------------------------------------------------------------------------
RUN_COLS = [
    "task", "spec", "seed", "steps", "length", "plddt", "ptm", "energy",
    "crmsd", "drmsd", "hydrophobics", "seconds", "weights_profile", "device",
]


def write_runs_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(runs, key=lambda x: (x.get("task", ""), x.get("spec", ""), x.get("seed", 0))):
            w.writerow({k: r.get(k, "") for k in RUN_COLS})


def write_spec_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    g = group_by_spec(runs)
    cols = ["task", "spec", "n", "plddt_mean", "plddt_std", "plddt_median", "plddt_min", "plddt_max",
            "ptm_mean", "energy_mean", "crmsd_mean", "crmsd_median", "drmsd_mean", "seconds_mean"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (task, spec), rows in sorted(g.items()):
            pl = agg([r.get("plddt") for r in rows])
            pt = agg([r.get("ptm") for r in rows])
            en = agg([r.get("energy") for r in rows])
            cr = agg([r.get("crmsd") for r in rows])
            dr = agg([r.get("drmsd") for r in rows])
            se = agg([r.get("seconds") for r in rows])
            w.writerow({
                "task": task, "spec": spec, "n": len(rows),
                "plddt_mean": fmt(pl), "plddt_std": fmt(pl, "std"),
                "plddt_median": fmt(pl, "median"), "plddt_min": fmt(pl, "min"),
                "plddt_max": fmt(pl, "max"),
                "ptm_mean": fmt(pt), "energy_mean": fmt(en),
                "crmsd_mean": fmt(cr), "crmsd_median": fmt(cr, "median"),
                "drmsd_mean": fmt(dr), "seconds_mean": fmt(se, "mean", 1),
            })


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def build_report(runs: List[Dict[str, Any]], rt: List[Dict[str, Any]], root: Path) -> str:
    g = group_by_spec(runs)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_task[r.get("task", "?")].append(r)

    L: List[str] = []
    L.append("# 论文一复现结果汇总")
    L.append("")
    L.append("论文：*A high-level programming language for generative protein design*（Hie et al., 2022）")
    L.append("")
    L.append(f"- 产物根目录：`{root}`")
    L.append(f"- 成功落盘的跑动总数：**{len(runs)}**，涉及 **{len(g)}** 个 spec、**{len(by_task)}** 个任务")
    L.append(f"- 逆折叠 roundtrip 记录：**{len(rt)}** 条" if rt else "- 逆折叠 roundtrip：**尚未运行**（见 `roundtrip.py`）")
    L.append("")
    L.append("> 判定口径取自论文正文：pLDDT > 0.7 为高置信；固定骨架设计 RMSD < 1.6 Å；"
             "位点脚手架 RMSD 红线 2 Å；结构新颖性 TM-score 红线 0.6。")
    L.append("")

    # ---- 总览
    L.append("## 1. 任务总览")
    L.append("")
    L.append("| 任务 | 论文位置 | spec 数 | 跑动数 | pLDDT 中位 | pLDDT>0.7 占比 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    fig_of = {
        "free_hallucination": "图 2A-C",
        "fixed_backbone": "图 2D-F",
        "secondary_structure": "图 2G / S1A-C",
        "functional_site_scaffolding": "图 2H / S1D",
        "symmetric_monomer": "图 3A-D / S2A-C",
        "homo_oligomer": "图 3E",
        "two_level_multimer": "图 4A-D / S3A-C",
        "symmetric_binding": "图 5A-B / S4A-C",
        "hierarchical_asymmetric": "图 5C-F / S4D-E",
    }
    for task in sorted(by_task):
        rows = by_task[task]
        pl = agg([r.get("plddt") for r in rows])
        L.append(
            f"| `{task}` | {fig_of.get(task, '—')} | "
            f"{len({r.get('spec') for r in rows})} | {len(rows)} | "
            f"{fmt(pl, 'median', 3)} | {frac([r.get('plddt') for r in rows], lambda v: v > PLDDT_GOOD)} |"
        )
    L.append("")

    # ---- 逐实验
    def section(title: str, task: str, body_fn) -> None:
        if task not in by_task:
            return
        L.append(title if title.startswith("## ") else f"## {title}")
        L.append("")
        body_fn(by_task[task])
        L.append("")

    def free_hallucination_body(rows):
        pl = [r.get("plddt") for r in rows]
        pt = [r.get("ptm") for r in rows]
        L.append(f"- 跑动数：{len(rows)}（论文：200 seeds）")
        L.append(f"- pLDDT 均值 {fmt(agg(pl))}，中位 {fmt(agg(pl), 'median')}")
        L.append(f"- **pLDDT > 0.7 的占比：{frac(pl, lambda v: v > PLDDT_GOOD)}**"
                 f"　（论文图 2B：100%）")
        L.append(f"- pTM 均值 {fmt(agg(pt))}")
        L.append("- 论文图 2C 的 ssAF2 pLDDT 需要外部 AlphaFold2（单序列），不在本工具包内。")

    def fixed_backbone_body(rows):
        L.append("| target | 跑动数 | cRMSD 中位 ↓ | cRMSD<1.6Å | pLDDT 中位 ↑ |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            cr = [r.get("crmsd") for r in sub]
            L.append(
                f"| {spec.upper()} | {len(sub)} | {fmt(agg(cr), 'median', 3)} | "
                f"{frac(cr, lambda v: v < RMSD_FIXEDBB)} | {fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} |"
            )
        L.append("")
        L.append("论文图 2E/2F：每个 target ≥50 seeds，pLDDT 与 RMSD 都是箱线图，RMSD 红线 2.5 Å。")

    def secondary_structure_body(rows):
        def ss_of(row: Dict[str, Any], idx: int) -> Optional[float]:
            d = row.get("ss_mismatches_by_node") or {}
            vals = [d[k] for k in sorted(d)]
            return vals[idx] if idx < len(vals) else None

        L.append("| spec | 跑动数 | pLDDT 中位 | 第 1 段匹配率 | 第 2 段匹配率 |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            L.append(
                f"| {spec} | {len(sub)} | "
                f"{fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} | "
                f"{fmt(agg([ss_of(r, 0) for r in sub]), 'mean', 3)} | "
                f"{fmt(agg([ss_of(r, 1) for r in sub]), 'mean', 3)} |"
            )
        L.append("")
        L.append("匹配率 = 1 - `MatchSecondaryStructure` 的返回值，即「属于目标二级结构的残基占比」。"
                 "论文图 2G / S1B-C 分别报告 α 与 β 的残基占比："
                 "本工具包跑 all_alpha / all_beta / mixed_ab 三个 spec，"
                 "分别对应「两段都 α」「两段都 β」「前 α 后 β」。")

    def site_body(rows):
        L.append("| 位点 spec | 跑动数 | cRMSD 中位 ↓ | cRMSD<1Å | cRMSD<2Å | pLDDT 中位 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            cr = [r.get("crmsd") for r in sub]
            L.append(
                f"| {spec} | {len(sub)} | {fmt(agg(cr), 'median', 3)} | "
                f"{frac(cr, lambda v: v < 1.0)} | {frac(cr, lambda v: v < RMSD_SITE)} | "
                f"{fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} |"
            )
        L.append("")
        L.append("论文图 2H：5 个位点、每个 2000 seeds，报告亚埃（<1 Å）级配位；"
                 "红线 2 Å。`*_generic` 是本工具包按论文残基区间实现的通用版，"
                 "`ace2_repo` 是仓库自带实现（两者位点定义不同，不可直接混比）。")

    def symmetric_monomer_body(rows):
        L.append("| 对称度 K | 总长 | 跑动数 | pLDDT 中位 ↑ | pLDDT>0.7 |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            pl = [r.get("plddt") for r in sub]
            K = spec.split("_")[0]
            ln = spec.split("len")[-1]
            L.append(f"| {K} | {ln} | {len(sub)} | {fmt(agg(pl), 'median', 3)} | "
                     f"{frac(pl, lambda v: v > PLDDT_GOOD)} |")
        L.append("")
        L.append("论文图 3B / S2A：6 种对称 × 3 种长度 × 10 seeds = 180 条跑动；"
                 "S2C 另需 TM-align 算与 PDB 最相似结构的 TM-score（见 roundtrip.py 说明）。")

    def oligo_body(rows):
        L.append("| 聚体 | 跑动数 | pLDDT 中位 | pTM 中位 | 对称性方差 |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            L.append(f"| {spec} | {len(sub)} | {fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('ptm') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('symmetry_var') for r in sub]), 'median', 4)} |")

    def two_level_body(rows):
        L.append("| 顶层 × 底层 | 跑动数 | pLDDT 中位 | pTM 中位 | 对称性方差 |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            L.append(f"| {spec} | {len(sub)} | {fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('ptm') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('symmetry_var') for r in sub]), 'median', 4)} |")
        L.append("")
        L.append("论文图 4B / S3A：顶层与底层对称度各 2~4，共 9 个程序 × 10 seeds = 90 条跑动。")

    def binding_body(rows):
        L.append("| spec | 跑动数 | cRMSD 中位 ↓ | 多点平均 cRMSD | pLDDT 中位 |")
        L.append("| --- | --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            L.append(f"| {spec} | {len(sub)} | {fmt(agg([r.get('crmsd') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('crmsd') for r in sub]))} | "
                     f"{fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} |")
        L.append("")
        L.append("论文图 5B / S4A-C：3 个程序各 20 seeds；"
                 "ACE2 位点需自行把模板从 IL10（1Y6K）换成 6M0J，仓库未提供。")

    def asym_body(rows):
        L.append("| spec | 跑动数 | pLDDT 中位 | pTM 中位 |")
        L.append("| --- | --- | --- | --- |")
        for spec in sorted({r.get("spec") for r in rows}):
            sub = [r for r in rows if r.get("spec") == spec]
            L.append(f"| {spec} | {len(sub)} | {fmt(agg([r.get('plddt') for r in sub]), 'median', 3)} | "
                     f"{fmt(agg([r.get('ptm') for r in sub]), 'median', 3)} |")
        L.append("")
        L.append("论文图 5C-F / S4D-E：3 个程序 × 10 seeds = 30 条跑动。"
                 "本工具包的实现把「两个非对称子单元」表达为顶层两个 child（不加对称约束），"
                 "每个子单元内部是双层对称。仓库未提供该程序，属等价实现。")

    section("## 2. 自由幻觉（图 2A-C）", "free_hallucination", free_hallucination_body)
    section("## 3. 固定骨架设计（图 2D-F）", "fixed_backbone", fixed_backbone_body)
    section("## 4. 二级结构设计（图 2G / S1A-C）", "secondary_structure", secondary_structure_body)
    section("## 5. 功能位点脚手架（图 2H / S1D）", "functional_site_scaffolding", site_body)
    section("## 6. 单链对称（图 3A-D / S2）", "symmetric_monomer", symmetric_monomer_body)
    section("## 7. 同源寡聚体（图 3E）", "homo_oligomer", oligo_body)
    section("## 8. 双层对称（图 4 / S3）", "two_level_multimer", two_level_body)
    section("## 9. 对称功能位点脚手架（图 5A-B / S4A-C）", "symmetric_binding", binding_body)
    section("## 10. 非对称层级组合（图 5C-F / S4D-E）", "hierarchical_asymmetric", asym_body)

    # ---- roundtrip
    L.append("## 11. 逆折叠 roundtrip（图 3C-D / 4C-D / S2D）")
    L.append("")
    if not rt:
        L.append("尚未运行。命令：")
        L.append("")
        L.append("```bash")
        L.append("python roundtrip.py --pdb-root <p1 产物根> --tasks symmetric_monomer two_level_multimer \\")
        L.append("    --n-structures 1000 --num-samples 10 --temperature 0.1 --out <root>/roundtrip")
        L.append("```")
    else:
        by_task_rt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rt:
            by_task_rt[r.get("task", "?")].append(r)
        L.append("| 任务 | 结构数 | 采样数 | roundtrip cRMSD 中位 ↓ | 最短 cRMSD<1Å | ESM-IF1 perplexity 中位 ↓ |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for task, rows in sorted(by_task_rt.items()):
            crm = [r.get("roundtrip_crmsd_min") for r in rows]
            ppl = [r.get("if1_perplexity") for r in rows]
            L.append(f"| `{task}` | {len(rows)} | {sum(int(r.get('num_samples') or 0) for r in rows)} | "
                     f"{fmt(agg(crm), 'median', 3)} | {frac(crm, lambda v: v < 1.0)} | "
                     f"{fmt(agg(ppl), 'median', 3)} |")
        L.append("")
        L.append("论文结论（图 3C-D / 4C-D）：**起始结构 pLDDT 越高、ESM-IF1 perplexity 越低，"
                 "roundtrip 越容易成功**。对应散点图见 `fig_roundtrip.png`。")
    L.append("")

    # ---- 未覆盖项
    L.append("## 12. 本工具包不覆盖的部分（需要外部资源）")
    L.append("")
    L.append("| 论文内容 | 缺口 | 建议做法 |")
    L.append("| --- | --- | --- |")
    L.append("| 图 2C ssAF2 pLDDT | 需要单序列 AlphaFold2 | ColabFold `--single-sequence` 或 AF2 官方代码 |")
    L.append("| 图 S2C / S3C 结构新颖性 | 需要 TM-align + PDB 2022-08 全库（>10 万结构） | 下载 PDB 后 `tm-align -byresi` 穷举；也可用 Foldseek 预筛 |")
    L.append("| 图 S2D ProteinMPNN roundtrip | 需要 ProteinMPNN 外部仓库 | `examples/inverse_folding` 里有 IF1；MPNN 需另装 |")
    L.append("| Discussion 的实验验证 | 湿实验（基因合成 + 表达 + SEC） | 不可计算复现，只能送样 |")
    L.append("| 图 2E/2F 的“50 or more seeds” | 本工具包 paper 网格默认 50 seeds/target | `--seeds 100` 可加大 |")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"生成方式：`python aggregate_p1.py --root {root}`")
    return "\n".join(L)


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def make_plots(runs: List[Dict[str, Any]], rt: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[跳过绘图] matplotlib 不可用：{exc}")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    made: List[Path] = []
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_task[r.get("task", "?")].append(r)

    def boxplot(task: str, value_key: str, ylabel: str, title: str, fname: str,
                threshold: Optional[float] = None, ylim=None) -> None:
        rows = by_task.get(task) or []
        if not rows:
            return
        labels, data = [], []
        for spec in sorted({r.get("spec") for r in rows}):
            vals = [r.get(value_key) for r in rows
                    if r.get("spec") == spec and r.get(value_key) is not None]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                labels.append(spec)
                data.append(vals)
        if not data:
            return
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels) + 2), 4.2))
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        for i, vals in enumerate(data, start=1):
            ax.scatter([i] * len(vals), vals, s=8, c="black", alpha=0.5, zorder=3)
        if threshold is not None:
            ax.axhline(threshold, color="red", lw=1.2)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    # 图 2B 同款：自由幻觉 pLDDT 分布
    rows = by_task.get("free_hallucination") or []
    pl = [r.get("plddt") for r in rows if r.get("plddt") is not None]
    if pl:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.hist(pl, bins=20, color="#4c78a8", edgecolor="white")
        ax.axvline(PLDDT_GOOD, color="red", lw=1.2)
        ax.set_xlabel("ESMFold mean pLDDT")
        ax.set_ylabel("count")
        ax.set_title(f"Free hallucination (n={len(pl)})")
        fig.tight_layout()
        p = out_dir / "fig_free_hallucination_plddt.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    boxplot("fixed_backbone", "plddt", "ESMFold mean pLDDT",
            "Fixed backbone design: pLDDT (Fig 2E)", "fig_fixedbb_plddt.png", PLDDT_GOOD)
    boxplot("fixed_backbone", "crmsd", "backbone cRMSD to target (Å)",
            "Fixed backbone design: cRMSD (Fig 2F)", "fig_fixedbb_crmsd.png", RMSD_FIXEDBB_BOX)
    boxplot("functional_site_scaffolding", "crmsd", "binding-site all-atom cRMSD (Å)",
            "Functional site scaffolding (Fig 2H)", "fig_site_crmsd.png", RMSD_SITE)
    boxplot("symmetric_monomer", "plddt", "ESMFold mean pLDDT",
            "Symmetric monomer: pLDDT by (K, length) (Fig S2A)", "fig_symmono_plddt.png", PLDDT_GOOD)
    boxplot("two_level_multimer", "plddt", "ESMFold mean pLDDT",
            "Two-level symmetry: pLDDT (Fig S3A)", "fig_twolevel_plddt.png", PLDDT_GOOD)
    boxplot("symmetric_binding", "crmsd", "binding-site cRMSD (Å)",
            "Symmetric binding-site scaffolding (Fig S4B)", "fig_symbind_crmsd.png", RMSD_SITE)

    # 图 3C-D / 4C-D 同款：roundtrip 散点
    if rt:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        xs = [r.get("starting_plddt") for r in rt if r.get("starting_plddt") is not None]
        ys = [r.get("roundtrip_crmsd_min") for r in rt if r.get("starting_plddt") is not None]
        if xs:
            axes[0].scatter(xs, ys, s=8, alpha=0.45)
        axes[0].set_xlabel("starting structure ESMFold pLDDT")
        axes[0].set_ylabel("min roundtrip cRMSD (Å)")
        axes[0].set_title("Design confidence vs roundtrip (Fig 3C/4C)")
        ppl = [r.get("if1_perplexity") for r in rt if r.get("if1_perplexity") is not None]
        y2 = [r.get("roundtrip_crmsd_min") for r in rt if r.get("if1_perplexity") is not None]
        if ppl:
            axes[1].scatter(ppl, y2, s=8, alpha=0.45)
        axes[1].set_xlabel("ESM-IF1 sample perplexity")
        axes[1].set_ylabel("min roundtrip cRMSD (Å)")
        axes[1].set_title("IF1 perplexity vs roundtrip (Fig 3D/4D)")
        fig.tight_layout()
        p = out_dir / "fig_roundtrip.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    return made


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="论文一结果汇总")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "outputs" / "p1",
                    help="design_programs.py 的 --out-dir")
    ap.add_argument("--out", type=Path, default=None, help="汇总输出目录（默认 <root>/summary）")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    out_dir: Path = args.out or (root / "summary")
    if not root.is_dir():
        print(f"[错误] 找不到产物目录 {root}")
        print("先跑：python design_programs.py --task <TASK> --grid smoke --out-dir", root)
        return 2

    runs = load_runs(root)
    rt = load_roundtrip(root)
    print(f"读到 {len(runs)} 条跑动记录，{len(rt)} 条 roundtrip 记录")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_runs_csv(runs, out_dir / "p1_runs.csv")
    write_spec_csv(runs, out_dir / "p1_by_spec.csv")
    (out_dir / "P1_SUMMARY.md").write_text(build_report(runs, rt, root), encoding="utf-8")
    figs = [] if args.no_plot else make_plots(runs, rt, out_dir)

    print(f"已写出：{out_dir / 'p1_runs.csv'}")
    print(f"已写出：{out_dir / 'p1_by_spec.csv'}")
    print(f"已写出：{out_dir / 'P1_SUMMARY.md'}")
    for f in figs:
        print(f"已写出：{f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
