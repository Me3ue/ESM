#!/usr/bin/env python3
"""
论文二：把 run_lm_design_batch.py（+ analyze_novelty.py）的产物汇总成表格与图。

纯后处理，不加载模型。

产出（默认写到 <root>/summary/）：
  p2_runs.csv         每条跑动一行（task, tag, seed, length, lm_perplexity, oracle_ca_rmsd, seconds…）
  p2_by_config.csv    按 task×tag 聚合
  P2_SUMMARY.md       对照论文正文图示的结论表 + 与 paper-data 的对照
  fig_energy_trajectory.png   图 2D 同款：能量随 MCMC 步数的下降曲线
  fig_lm_perplexity.png       图 2E 同款：LM 对设计的困惑度分布
  fig_oracle_rmsd.png         图 2A 同款：oracle RMSD（本工具包用 ESMFold，非论文的 AlphaFold）
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

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mplconfig"))

REPRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_DIR.parent
PAPER_DATA = REPO_ROOT / "examples" / "lm-design" / "paper-data"


def agg(values: List[Optional[float]]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": st.fmean(vals),
            "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "median": st.median(vals), "min": min(vals), "max": max(vals)}


def f(d: Dict[str, float], key: str = "mean", nd: int = 3) -> str:
    return "—" if d.get("n", 0) == 0 or key not in d else f"{d[key]:.{nd}f}"


def load_runs(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("metrics.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        r["_path"] = str(p)
        r["_traj"] = str(p.parent / "trajectory.csv")
        r["_has_pred"] = (p.parent / "esmfold_pred.pdb").exists()
        rows.append(r)
    return rows


def load_novelty(root: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(root.rglob("novelty.csv")):
        with p.open(newline="", encoding="utf-8") as fh:
            out.extend(list(csv.DictReader(fh)))
    return out


def load_paper_data() -> List[Dict[str, Any]]:
    p = PAPER_DATA / "data.csv"
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
def write_csvs(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    cols = ["task", "tag", "seed", "num_iter", "length", "lm_perplexity",
            "oracle", "oracle_ca_rmsd", "final_total_loss", "final_temperature", "seconds", "pdb_fn"]
    with (out_dir / "p2_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(runs, key=lambda x: (x.get("task", ""), x.get("tag", ""), x.get("seed", 0))):
            w.writerow({k: r.get(k, "") for k in cols})

    g: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        g[(r.get("task"), r.get("tag"))].append(r)
    cols2 = ["task", "tag", "n", "length_mean", "ppl_mean", "ppl_median", "ppl_min",
             "oracle_rmsd_mean", "oracle_rmsd_median", "final_loss_mean", "seconds_mean"]
    with (out_dir / "p2_by_config.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols2)
        w.writeheader()
        for (task, tag), rows in sorted(g.items()):
            w.writerow({
                "task": task, "tag": tag, "n": len(rows),
                "length_mean": f(agg([r.get("length") for r in rows])),
                "ppl_mean": f(agg([r.get("lm_perplexity") for r in rows])),
                "ppl_median": f(agg([r.get("lm_perplexity") for r in rows]), "median"),
                "ppl_min": f(agg([r.get("lm_perplexity") for r in rows]), "min"),
                "oracle_rmsd_mean": f(agg([r.get("oracle_ca_rmsd") for r in rows]), "mean", 2),
                "oracle_rmsd_median": f(agg([r.get("oracle_ca_rmsd") for r in rows]), "median", 2),
                "final_loss_mean": f(agg([r.get("final_total_loss") for r in rows])),
                "seconds_mean": f(agg([r.get("seconds") for r in rows]), "mean", 1),
            })


# --------------------------------------------------------------------------
def build_report(runs: List[Dict[str, Any]], novelty: List[Dict[str, Any]], paper: List[Dict[str, Any]]) -> str:
    g: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        g[(r.get("task"), r.get("tag"))].append(r)

    L: List[str] = []
    L.append("# 论文二复现结果汇总")
    L.append("")
    L.append("论文：*Language models generalize beyond natural proteins*（Verkuil, Kabeli et al., 2022）")
    L.append("")
    L.append(f"- 本次跑动数：**{len(runs)}**，涉及 {len(g)} 个 (task, tag) 组合")
    if novelty:
        L.append(f"- 新颖性检索记录：**{len(novelty)}** 条")
    else:
        L.append("- 新颖性检索：**尚未运行**（`analyze_novelty.py`，需要 jackhmmer + 序列库）")
    L.append("")
    L.append("> ⚠️ 本工具包的结构 oracle 是 **ESMFold**，而论文用的是 **AlphaFold**"
             "（5 个 pTM 模型选最优 + Amber 松弛，App A.4.1）。")
    L.append("> 两者的 RMSD / pLDDT 绝对值不可直接比较，只能看相对趋势。")
    L.append("")

    sec_no = [1]

    def add_section(title: str, lines: List[str]) -> None:
        L.append(f"## {sec_no[0]}. {title}")
        sec_no[0] += 1
        L.append("")
        L.extend(lines)
        L.append("")

    body = [
        "| task | tag | n | 长度 | LM 困惑度 中位 ↓ | 困惑度最小 | oracle CA-RMSD 中位 ↓ | 终态能量 | 单条耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (task, tag), rows in sorted(g.items()):
        ppl = agg([r.get("lm_perplexity") for r in rows])
        rm = agg([r.get("oracle_ca_rmsd") for r in rows])
        body.append(
            f"| `{task}` | {tag} | {len(rows)} | {f(agg([r.get('length') for r in rows]))} | "
            f"{f(ppl, 'median')} | {f(ppl, 'min')} | {f(rm, 'median', 2)} | "
            f"{f(agg([r.get('final_total_loss') for r in rows]))} | "
            f"{f(agg([r.get('seconds') for r in rows]), 'mean', 1)} |"
        )
    body.append("")
    body.append("论文对应规模：固定骨架 **39 个 de novo target × 200 designs**、"
                "自由生成 **25,000 条**，每条 **170,000 步** MCMC（App A.3.1 / A.3.3）。")
    add_section("逐配置结果", body)

    if novelty:
        n_tot = len(novelty)
        def numf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        nosig = sum(1 for r in novelty if not (r.get("max_seq_identity_sig") or "").strip()
                    and not (r.get("n_significant") or "").strip())
        ids = [numf(r.get("max_seq_identity_sig")) for r in novelty]
        ids = [v for v in ids if v is not None]
        lines = [f"- 检索设计数：{n_tot}；**无显著命中**：{nosig}"]
        if ids:
            lines.append(f"- 序列一致性中位：{st.median(ids):.3f}；"
                         f"低于 20% 的设计：{sum(1 for v in ids if v < 0.20)} 条；"
                         f"低于 30% 的设计：{sum(1 for v in ids if v < 0.30)} 条")
        lines.append("")
        lines.append("论文对照（对 UniRef90 2021_04 + purge 列表检索）："
                     "152 个成功设计里 35 个无显著匹配；有命中的 117 条中位一致性 27%，"
                     "6 条 < 20%，最低 18%。")
        add_section("序列新颖性", lines)

    if paper:
        lm = [r for r in paper if r.get("Design Model") == "LM"]
        suc = sum(1 for r in lm if (r.get("Success") or "").strip() == "True")
        add_section("与论文公开实验数据的对照", [
            f"- 仓库 `examples/lm-design/paper-data/data.csv` 共 {len(paper)} 条送检蛋白，"
            f"其中 LM 设计 {len(lm)} 条，成功 {suc} 条 "
            f"({100.0*suc/len(lm):.0f}%)，论文摘要为 152/228 (67%)。",
            "- 论文这一部分的完整复算见 `paper_data_report.py` 产出的 "
            "`PAPER_DATA_REPORT.md`（默认在 `<root>/paper_data_report/`）。",
        ])

    add_section("本工具包不覆盖的部分（需要外部资源）", [
        "| 论文内容 | 缺口 | 建议做法 |",
        "| --- | --- | --- |",
        "| App A.4.1 AlphaFold oracle（图 2A/3C） | 需要 AF2 + 5 个 pTM 模型 + Amber | ColabFold batch，或本地 AF2；结果存 CSV 后并入本汇总 |",
        "| App A.3.2 no-LM 基线（图 2C/2F） | 需要 ColabDesign `3stage()` AfDesign | 外部仓库，`commit e7bb3def` |",
        "| App A.4.2/4.3/4.4 溶解度/堆积/球状性过滤 | 需要 Rosetta（PackStat、SSShapeComplementarity、TotalSasa、SAP） | 装 Rosetta 后按阈值过滤：packing>0.55、shape comp>0.6、rel Rg<1.5、rel SASA<3、SAP≤0.4 |",
        "| App A.5.2 图 4D（vs AlphaFold DB 全库） | 需要 AlphaFold DB 序列 + 结构 | 下载 AF DB fasta + `AF-<UniProtID>-F1-model_v3.pdb` |",
        "| App A.5.4 motif 分析（图 3D-E） | 需要 Foldseek（`--alignment-type 1`） | `conda install -c bioconda foldseek` |",
        "| App A.7 湿实验（SEC、SEC 曲线图） | 基因合成 + 表达 + 纯化 | 不可计算复现；论文 SEC 原始曲线在 `data.hdf5` |",
    ])

    add_section("一键补齐的建议顺序", [
        "```bash",
        "bash run_p2.sh                          # 设计 + 汇总（MODE=smoke 先打通）",
        "python analyze_novelty.py --db <uniref90.fasta> --db-name uniref90_2021_04 \\",
        "    --out outputs/p2/novelty            # 序列新颖性（图 2G / 4F-G）",
        "python aggregate_p2.py --root outputs/p2   # 把新颖性并入总报告",
        "```",
    ])
    return "\n".join(L)


# --------------------------------------------------------------------------
def make_plots(runs: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[跳过绘图] matplotlib 不可用：{exc}")
        return []

    made: List[Path] = []

    # ---- 图 2D 同款：能量下降曲线
    trajs = []
    for r in runs:
        p = Path(r["_traj"])
        if not p.exists():
            continue
        steps, losses = [], []
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    steps.append(int(row["step"]))
                    losses.append(float(row["total_loss"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if steps:
            trajs.append((f"{r.get('tag')}/{r.get('seed')}", steps, losses))
    if trajs:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for label, steps, losses in trajs[:50]:
            ax.plot(steps, losses, lw=1.0, alpha=0.7, label=label if len(trajs) <= 6 else None)
        ax.set_xlabel("MCMC step")
        ax.set_ylabel("total loss  E(x)")
        ax.set_title("Optimization trajectory (Fig 2D analogue)")
        if len(trajs) <= 6:
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "fig_energy_trajectory.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    # ---- 图 2E 同款：LM 困惑度分布
    ppl = [r.get("lm_perplexity") for r in runs if r.get("lm_perplexity") is not None]
    if ppl:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.hist(ppl, bins=20, color="#4c78a8", edgecolor="white")
        ax.set_xlabel("LM pseudo-perplexity of the design")
        ax.set_ylabel("count")
        ax.set_title(f"Language-model perplexity (n={len(ppl)})")
        fig.tight_layout()
        p = out_dir / "fig_lm_perplexity.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    # ---- 图 2A 同款：oracle RMSD
    g: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        v = r.get("oracle_ca_rmsd")
        if v is not None:
            g[str(r.get("tag"))].append(float(v))
    if g:
        labels = sorted(g)
        fig, ax = plt.subplots(figsize=(max(4.5, 0.6 * len(labels) + 2), 3.8))
        ax.boxplot([g[k] for k in labels], tick_labels=labels, showfliers=True)
        for i, k in enumerate(labels, start=1):
            ax.scatter([i] * len(g[k]), g[k], s=10, c="black", alpha=0.5, zorder=3)
        ax.axhline(2.5, color="red", lw=1.2)
        ax.set_ylabel("oracle Cα-RMSD to target (Å)")
        ax.set_title("Fixed-backbone oracle RMSD (ESMFold, not AlphaFold)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = out_dir / "fig_oracle_rmsd.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(p)

    return made


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="论文二结果汇总")
    ap.add_argument("--root", type=Path, default=REPRO_DIR / "outputs" / "p2")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    out_dir: Path = args.out or (root / "summary")
    if not root.is_dir():
        print(f"[错误] 找不到产物目录 {root}")
        print("先跑：python run_lm_design_batch.py --task fixedbb --pdb 2N2U --seeds 0 --num-iter 300 --out-dir", root)
        return 2

    runs = load_runs(root)
    novelty = load_novelty(root)
    paper = load_paper_data()
    print(f"读到 {len(runs)} 条跑动、{len(novelty)} 条新颖性记录、{len(paper)} 行论文数据")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(runs, out_dir)
    (out_dir / "P2_SUMMARY.md").write_text(build_report(runs, novelty, paper), encoding="utf-8")
    figs = [] if args.no_plot else make_plots(runs, out_dir)

    print(f"已写出：{out_dir / 'p2_runs.csv'}")
    print(f"已写出：{out_dir / 'p2_by_config.csv'}")
    print(f"已写出：{out_dir / 'P2_SUMMARY.md'}")
    for p in figs:
        print(f"已写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
