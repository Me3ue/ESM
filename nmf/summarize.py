#!/usr/bin/env python
"""把 ``nmf`` 各阶段的产物汇总成一份可直接引用的指标表。

只做 JSON / 检查点的**后处理**：不加载 ESM、不重新评测，因此实验跑完后可以随时重跑，
也可以在只跑了一部分实验时先生成"已完成的那些格"的汇总。

汇总内容：

1. 数据（`data/processed/<名称>/stats.json`）：条数、长度分位、去污染、sha256；
2. **逐层分解质量**（`factor_report_<tag>.json` 或 `factors_<tag>.pt`）：平均/中位/最差相对
   Frobenius 误差、余弦相似度、参数量倍率、耗时；
3. **训练过程**（`history_<tag>.json`）：步数、轮数、耗时、分组学习率、蒸馏损失与重构误差的
   起止值、回滚次数、最后的验证集指标；
4. **模型级能力对比**（`<bench>/benchmark.json`）：直接复用 `nmf.benchmark` 的四张论文表。

产物：`SUMMARY.md`（人读）+ `summary.json`（机器可读）+ `layer_metrics.csv`（逐层明细）。

示例
----
.. code-block:: bash

    python -m nmf.summarize --root nmf/outputs/fullmodel \
        --bench nmf/outputs/fullmodel/bench_matrix \
        --data data/processed/swissprot \
        --out nmf/outputs/fullmodel/SUMMARY.md
"""

import argparse
import csv
import json
import os
import statistics as stats
from typing import Dict, List, Optional


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


# --------------------------------------------------------------------------- #
# 1. 数据
# --------------------------------------------------------------------------- #
def collect_data(data_dir: str) -> Optional[Dict]:
    path = os.path.join(data_dir, "stats.json")
    report = _load_json(path)
    if report is None:
        return None
    out: Dict = {
        "stats_path": os.path.abspath(path),
        "source": report.get("source"),
        "query": report.get("query"),
        "raw_file": report.get("raw_file"),
        "raw_file_sha256": report.get("raw_file_sha256"),
        "quality_control": report.get("quality_control", {}),
        "decontamination": report.get("decontamination", {}),
        "splits": {},
    }
    for name, payload in report.get("splits", {}).items():
        out["splits"][name] = {
            "path": payload.get("path"),
            "sha256": payload.get("sha256"),
            "length": payload.get("length", {}),
        }
    return out


# --------------------------------------------------------------------------- #
# 2. 逐层分解质量
# --------------------------------------------------------------------------- #
def discover_tags(root: str) -> List[str]:
    """从 ``factor_report_<tag>.json`` / ``factors_<tag>.pt`` / ``history_<tag>.json`` 推断 tag。"""
    tags: List[str] = []
    for prefix, suffix in (
        ("factor_report_", ".json"),
        ("factors_", ".pt"),
        ("history_", ".json"),
        ("trained_", ".pt"),
    ):
        if not os.path.isdir(root):
            continue
        for filename in sorted(os.listdir(root)):
            if filename.startswith(prefix) and filename.endswith(suffix):
                tag = filename[len(prefix) : -len(suffix)]
                if tag and tag not in tags:
                    tags.append(tag)
    return tags


def _factorization_layers(root: str, tag: str) -> Optional[Dict[str, Dict]]:
    """优先读轻量的报告 JSON，缺失时退回检查点里的 ``factorization`` 字段。"""
    report = _load_json(os.path.join(root, f"factor_report_{tag}.json"))
    if report and "layers" in report:
        return report["layers"]
    report = _load_json(os.path.join(root, f"factors_{tag}.pt"))
    if report is None:
        return None
    try:  # 检查点需要 torch
        import torch

        ckpt = torch.load(
            os.path.join(root, f"factors_{tag}.pt"), map_location="cpu", weights_only=False
        )
        return ckpt.get("factorization") or None
    except Exception:
        return None


def collect_factorization(root: str, tag: str) -> Optional[Dict]:
    layers = _factorization_layers(root, tag)
    if not layers:
        return None
    rels = [v["relative_frobenius_error"] for v in layers.values()]
    coss = [v["cosine_similarity"] for v in layers.values()]
    params_orig = sum(int(v.get("n_params_original", 0)) for v in layers.values())
    params_fact = sum(int(v.get("n_params_factored", 0)) for v in layers.values())
    ranks = sorted({int(v.get("rank", 0)) for v in layers.values()})
    report = _load_json(os.path.join(root, f"factor_report_{tag}.json")) or {}
    return {
        "tag": tag,
        "n_layers": len(layers),
        "ranks": ranks,
        "mean_relative_frobenius_error": _mean(rels),
        "median_relative_frobenius_error": float(stats.median(rels)),
        "max_relative_frobenius_error": float(max(rels)),
        "mean_cosine_similarity": _mean(coss),
        "min_cosine_similarity": float(min(coss)),
        "n_params_original": params_orig,
        "n_params_factored": params_fact,
        "param_ratio": (params_fact / params_orig) if params_orig else float("nan"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "per_layer": {
            name: {
                "shape": v.get("shape"),
                "rank": v.get("rank"),
                "relative_frobenius_error": v.get("relative_frobenius_error"),
                "cosine_similarity": v.get("cosine_similarity"),
                "n_params_original": v.get("n_params_original"),
                "n_params_factored": v.get("n_params_factored"),
            }
            for name, v in layers.items()
        },
    }


# --------------------------------------------------------------------------- #
# 3. 训练过程
# --------------------------------------------------------------------------- #
def collect_training(root: str, tag: str) -> Optional[Dict]:
    payload = _load_json(os.path.join(root, f"history_{tag}.json"))
    if not payload:
        return None
    history: List[Dict] = payload.get("history", [])
    eval_history: List[Dict] = payload.get("eval_history", [])
    if not history:
        return None

    rels = [r["loss_recon"] for r in history if "loss_recon" in r]
    out: Dict = {
        "tag": tag,
        "steps": len(history),
        "epochs_used": sorted({r.get("epoch") for r in history if r.get("epoch")}),
        "elapsed_seconds": history[-1].get("elapsed_s"),
        "seconds_per_step": (history[-1].get("elapsed_s") or 0) / len(history),
        "rolled_back_steps": sum(1 for r in history if r.get("rolled_back")),
        "lr_halved_steps": sum(1 for r in history if r.get("lr_halved")),
        "group_lrs_first": history[0].get("lrs"),
        "group_lrs_last": history[-1].get("lrs"),
        "distill_loss_first": history[0].get("loss_distill"),
        "distill_loss_last": history[-1].get("loss_distill"),
        "recon_rel_fro_first": rels[0] if rels else None,
        "recon_rel_fro_last": rels[-1] if rels else None,
        "n_masked_total": sum(int(r.get("n_masked", 0)) for r in history),
        "final_eval": eval_history[-1] if eval_history else None,
        "eval_steps": [e.get("step") for e in eval_history],
    }
    if out["recon_rel_fro_first"] and out["recon_rel_fro_last"]:
        out["recon_rel_fro_change"] = (
            out["recon_rel_fro_last"] / out["recon_rel_fro_first"] - 1.0
        )
    return out


# --------------------------------------------------------------------------- #
# 4. 模型级评测（复用 nmf.benchmark 的渲染）
# --------------------------------------------------------------------------- #
def collect_benchmark(bench_dir: str) -> Optional[Dict]:
    path = os.path.join(bench_dir, "benchmark.json")
    report = _load_json(path)
    if not report:
        return None
    variants = report.get("variants", {})
    if not variants:
        return None
    out: Dict = {
        "report_path": os.path.abspath(path),
        "n_eval": report.get("n_eval"),
        "n_residues": report.get("n_residues"),
        "mask_frac": report.get("mask_frac"),
        "mask_rounds": report.get("mask_rounds"),
        "seed": report.get("seed"),
        "fasta": report.get("fasta"),
        "orig_params": report.get("orig_params"),
        "orig_pseudo_perplexity": report.get("orig_pseudo_perplexity"),
        "variants": {},
    }
    for name, entry in variants.items():
        m = entry.get("masked_lm", {})
        d = entry.get("distribution", {})
        r = entry.get("representation", {})
        ls = entry.get("layer_stats", {})
        out["variants"][name] = {
            "checkpoint": entry.get("checkpoint"),
            "stage": entry.get("stage"),
            "masked_top1": m.get("nmf_top1_acc"),
            "masked_top1_std": m.get("nmf_top1_acc_std"),
            "orig_masked_top1": m.get("orig_top1_acc"),
            "masked_top5": m.get("nmf_top5_acc"),
            "orig_masked_top5": m.get("orig_top5_acc"),
            "masked_perplexity": m.get("nmf_masked_perplexity"),
            "orig_masked_perplexity": m.get("orig_masked_perplexity"),
            "pseudo_perplexity": entry.get("pseudo_perplexity"),
            "kl_b_a": d.get("kl_b_a"),
            "js_divergence": d.get("js_divergence"),
            "top1_agreement": d.get("top1_agreement"),
            "top5_agreement": d.get("top5_agreement"),
            "cka": r.get("cka"),
            "repr_cosine": r.get("mean_cosine"),
            "mean_relative_frobenius_error": ls.get("mean_relative_frobenius_error"),
            "max_relative_frobenius_error": ls.get("max_relative_frobenius_error"),
            "mean_cosine_similarity": ls.get("mean_cosine_similarity"),
            "min_cosine_similarity": ls.get("min_cosine_similarity"),
            "n_layers": ls.get("n_layers"),
            "n_params_factored_layers": ls.get("n_params_factored"),
            "n_params_original_layers": ls.get("n_params_original"),
            "n_params_total": entry.get("n_params_total"),
            "linear_mac_ratio": entry.get("linear_mac_ratio"),
            "forward_seconds": entry.get("time"),
        }
    return out


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _num(value, digits: int = 4, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return dash
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _delta(value: Optional[float], base: Optional[float], digits: int = 1) -> str:
    if value is None or base in (None, 0):
        return "—"
    return f"{100.0 * (value / base - 1.0):+.{digits}f}%"


def render_markdown(
    data: Optional[Dict],
    factorizations: Dict[str, Dict],
    trainings: Dict[str, Dict],
    benchmark: Optional[Dict],
    bench_dir: str,
) -> str:
    lines: List[str] = []
    lines.append("# NMF 实验指标汇总")
    lines.append("")
    if benchmark:
        lines.append(
            f"- 模型：`esm2_t6_8M_UR50D`（原模型参数量 {_num(benchmark.get('orig_params'))}）"
        )
        lines.append(
            f"- 评测集：`{benchmark.get('fasta')}`，{benchmark.get('n_eval')} 条"
            f"（{_num(benchmark.get('n_residues'))} 残基），遮挡 {benchmark.get('mask_frac')}"
            f" × {benchmark.get('mask_rounds')} 回合"
        )
        lines.append(f"- 评测目录：`{bench_dir}`")
    lines.append(f"- 汇总覆盖的分解/训练 tag：{'、'.join(sorted(set(factorizations) | set(trainings))) or '（无）'}")
    lines.append("")

    # ---- 1. 数据 ----
    lines.append("## 1. 数据")
    lines.append("")
    if data:
        qc, decon = data.get("quality_control", {}), data.get("decontamination", {})
        lines.append(f"- 来源：`{data.get('source')}`，原始文件 `{data.get('raw_file')}`")
        if data.get("query"):
            lines.append(f"- 查询式：`{data['query']}`")
        lines.append(
            f"- 原始 {_num(qc.get('input'))} → 质控后 {_num(qc.get('kept'))}"
            f"（非常规残基 {_num(qc.get('nonstandard'))}、过短 {_num(qc.get('too_short'))}、"
            f"过长 {_num(qc.get('too_long'))}、重复 {_num(qc.get('duplicate'))}）"
        )
        if decon:
            lines.append(
                f"- 去污染（{decon.get('k')}-mer 覆盖度 ≥ {decon.get('threshold')}）："
                f"train {_num(decon.get('train_before'))} → {_num(decon.get('train_after'))}"
                f"（剔除 {_num(decon.get('removed'))}）"
            )
        lines.append("")
        lines.append("| 划分 | 条数 | min | p50 | p95 | max | 平均长度 | 总残基 | sha256（前 12 位） |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in ("train", "valid", "test"):
            split = data.get("splits", {}).get(name)
            if not split:
                continue
            L = split.get("length", {})
            sha = (split.get("sha256") or "")[:12]
            lines.append(
                f"| {name} | {_num(L.get('n'))} | {_num(L.get('min'))} | {_num(L.get('p50'))} | "
                f"{_num(L.get('p95'))} | {_num(L.get('max'))} | {_num(L.get('mean'), 1)} | "
                f"{_num(L.get('total_residues'))} | `{sha}` |"
            )
    else:
        lines.append("（未找到 `stats.json`）")
    lines.append("")

    # ---- 2. 逐层分解 ----
    lines.append("## 2. 逐层分解质量（等效权重 vs 原权重）")
    lines.append("")
    if factorizations:
        lines.append("| tag | 层数 | 秩 | 平均相对 Frobenius 误差 ↓ | 中位 | 最差 | 平均余弦 ↑ | 最低余弦 | 参数量（原 → 分解） | 倍率 | 耗时 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in sorted(factorizations):
            f = factorizations[tag]
            ranks = f["ranks"]
            rank_str = f"{ranks[0]}~{ranks[-1]}" if len(ranks) > 1 else str(ranks[0] if ranks else "—")
            lines.append(
                f"| {tag} | {f['n_layers']} | {rank_str} | "
                f"**{_num(f['mean_relative_frobenius_error'], 5)}** | "
                f"{_num(f['median_relative_frobenius_error'], 5)} | "
                f"{_num(f['max_relative_frobenius_error'], 5)} | "
                f"{_num(f['mean_cosine_similarity'], 5)} | {_num(f['min_cosine_similarity'], 5)} | "
                f"{_num(f['n_params_original'])} → {_num(f['n_params_factored'])} | "
                f"**{_num(f['param_ratio'], 3)}×** | {_num(f['elapsed_seconds'], 0)} s |"
            )
        worst = []
        for tag in sorted(factorizations):
            for name, v in factorizations[tag]["per_layer"].items():
                worst.append((tag, name, v.get("relative_frobenius_error") or 0.0,
                              v.get("cosine_similarity") or 0.0))
        worst.sort(key=lambda x: -x[2])
        lines.append("")
        lines.append("最差的 5 层：")
        lines.append("")
        lines.append("| tag | 层 | 相对误差 ↓ | 余弦 ↑ |")
        lines.append("| --- | --- | --- | --- |")
        for tag, name, rel, cos in worst[:5]:
            lines.append(f"| {tag} | `{name}` | {_num(rel, 5)} | {_num(cos, 5)} |")
    else:
        lines.append("（未找到任何分解报告；先跑 `nmf.factorize`）")
    lines.append("")

    # ---- 3. 训练 ----
    lines.append("## 3. 训练过程（只更新 A/S/B/offset）")
    lines.append("")
    if trainings:
        lines.append("| tag | 步数 | 轮数 | 耗时 | 秒/步 | 蒸馏 KL（起 → 终） | 重构误差（起 → 终） | 变化 | 回滚步数 | 学习率减半步数 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in sorted(trainings):
            t = trainings[tag]
            lines.append(
                f"| {tag} | {_num(t['steps'])} | {len(t['epochs_used'])} | "
                f"{_num(t['elapsed_seconds'], 0)} s | {_num(t['seconds_per_step'], 2)} | "
                f"{_num(t['distill_loss_first'], 4)} → **{_num(t['distill_loss_last'], 4)}** | "
                f"{_num(t['recon_rel_fro_first'], 5)} → {_num(t['recon_rel_fro_last'], 5)} | "
                f"{_delta(t.get('recon_rel_fro_last'), t.get('recon_rel_fro_first'))} | "
                f"{t['rolled_back_steps']} | {t['lr_halved_steps']} |"
            )
        lines.append("")
        lines.append("分组学习率（按扰动增益标定，`--lr-scale auto`）：")
        lines.append("")
        lines.append("| tag | A | S | B | offset |")
        lines.append("| --- | --- | --- | --- | --- |")
        for tag in sorted(trainings):
            lrs = trainings[tag].get("group_lrs_last") or {}
            row = " | ".join(
                (f"{lrs[k]:.2e}" if k in lrs else "—") for k in ("A", "S", "B", "offset")
            )
            lines.append(f"| {tag} | {row} |")
        lines.append("")
        lines.append("训练中的验证集快评（valid 子集，用于观察趋势）：")
        lines.append("")
        lines.append("| tag | 步数 | 原 top-1 | 分解 top-1 | 原 masked ppl | 分解 masked ppl | KL | top-1 一致率 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag in sorted(trainings):
            e = trainings[tag].get("final_eval")
            if not e:
                continue
            lines.append(
                f"| {tag} | {_num(e.get('step'))} | {_num(e.get('orig_top1_acc'), 4)} | "
                f"{_num(e.get('nmf_top1_acc'), 4)} | {_num(e.get('orig_masked_ppl'), 3)} | "
                f"{_num(e.get('nmf_masked_ppl'), 3)} | {_num(e.get('kl_nmf_ref'), 4)} | "
                f"{_num(e.get('top1_agreement'), 4)} |"
            )
    else:
        lines.append("（未找到任何训练历史；先跑 `nmf.train_nmf`）")
    lines.append("")

    # ---- 4. 模型级能力对比 ----
    lines.append("## 4. 模型级能力对比（论文级协议）")
    lines.append("")
    if benchmark:
        base_name = next(iter(benchmark["variants"]))
        base_m = benchmark["variants"][base_name]
        lines.append(
            "| 变体 | 遮挡 top-1 ↑ | 相对原模型 | masked 困惑度 ↓ | 相对原模型 | 伪困惑度 ↓ | "
            "相对原模型 | KL(分解‖原) ↓ | top-1 一致率 ↑ | top-5 一致率 ↑ | 逐残基 CKA ↑ | "
            "平均层重构误差 ↓ | 参数量倍率 | 线性层乘加比 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append(
            f"| 原模型（基准） | {_num(base_m['orig_masked_top1'], 4)} | — | "
            f"{_num(base_m['orig_masked_perplexity'], 3)} | — | "
            f"{_num(benchmark.get('orig_pseudo_perplexity'), 4)} | — | — | — | — | — | — | — | — |"
        )
        for name in benchmark["variants"]:
            v = benchmark["variants"][name]
            lines.append(
                f"| {name} | {_num(v['masked_top1'], 4)}"
                f"{' ± ' + _num(v['masked_top1_std'], 4) if v.get('masked_top1_std') else ''} | "
                f"{_delta(v['masked_top1'], base_m['orig_masked_top1'])} | "
                f"{_num(v['masked_perplexity'], 3)} | "
                f"{_delta(v['masked_perplexity'], base_m['orig_masked_perplexity'])} | "
                f"{_num(v['pseudo_perplexity'], 4)} | "
                f"{_delta(v['pseudo_perplexity'], benchmark.get('orig_pseudo_perplexity'))} | "
                f"{_num(v['kl_b_a'], 5)} | {_num(v['top1_agreement'], 4)} | "
                f"{_num(v['top5_agreement'], 4)} | {_num(v['cka'], 5)} | "
                f"{_num(v['mean_relative_frobenius_error'], 5)} | "
                f"{_num(v['n_params_total'], 0)} / {_num(benchmark.get('orig_params'), 0)} | "
                f"{_num(v['linear_mac_ratio'], 3)}× |"
            )
        lines.append("")
        lines.append("### 4.1 层级别与效率明细")
        lines.append("")
        lines.append("| 变体 | 分解层数 | 平均相对误差 ↓ | 最差 | 平均余弦 ↑ | 最低余弦 | 分解层参数量（原 → 分解） | 全模型参数量 | 线性层乘加比 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in benchmark["variants"]:
            v = benchmark["variants"][name]
            lines.append(
                f"| {name} | {_num(v.get('n_layers'))} | "
                f"{_num(v['mean_relative_frobenius_error'], 5)} | "
                f"{_num(v['max_relative_frobenius_error'], 5)} | "
                f"{_num(v.get('mean_cosine_similarity'), 5)} | "
                f"{_num(v.get('min_cosine_similarity'), 5)} | "
                f"{_num(v.get('n_params_original_layers'))} → {_num(v.get('n_params_factored_layers'))} | "
                f"{_num(v['n_params_total'], 0)} | {_num(v['linear_mac_ratio'], 3)}× |"
            )
        lines.append("")
        lines.append(f"> 完整的四张论文表见 `{os.path.join(bench_dir, 'benchmark.md')}`；")
        lines.append(f"> 逐层明细见 `{os.path.join(bench_dir, 'layer_stats.csv')}`。")
    else:
        lines.append(f"（未找到 `{os.path.join(bench_dir, 'benchmark.json')}`；先跑 `nmf.benchmark`）")
    lines.append("")
    return "\n".join(lines)


def write_layer_csv(path: str, factorizations: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "layer", "shape", "rank", "relative_frobenius_error",
                         "cosine_similarity", "n_params_original", "n_params_factored"])
        for tag in sorted(factorizations):
            for name, v in factorizations[tag]["per_layer"].items():
                shape = v.get("shape") or []
                writer.writerow([
                    tag, name, "x".join(str(s) for s in shape), v.get("rank"),
                    None if v.get("relative_frobenius_error") is None else f"{v['relative_frobenius_error']:.6f}",
                    None if v.get("cosine_similarity") is None else f"{v['cosine_similarity']:.6f}",
                    v.get("n_params_original"), v.get("n_params_factored"),
                ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="汇总 nmf 各阶段产物，生成可直接引用的指标表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="nmf/outputs/fullmodel",
                        help="分解/训练产物的目录（factor_report_*.json、history_*.json 所在处）")
    parser.add_argument("--bench", default=None,
                        help="nmf.benchmark 的输出目录（含 benchmark.json）；默认 <root>/bench")
    parser.add_argument("--data", default="data/processed/swissprot", help="数据切分目录")
    parser.add_argument("--tags", nargs="*", default=None,
                        help="只汇总这些 tag；默认自动发现全部")
    parser.add_argument("--out", default=None, help="SUMMARY.md 路径，默认 <root>/SUMMARY.md")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = args.root
    bench_dir = args.bench or os.path.join(root, "bench")
    out_md = args.out or os.path.join(root, "SUMMARY.md")
    out_json = os.path.splitext(out_md)[0] + ".json"
    out_csv = os.path.join(os.path.dirname(os.path.abspath(out_md)), "layer_metrics.csv")

    tags = args.tags if args.tags else discover_tags(root)
    factorizations = {}
    trainings = {}
    for tag in tags:
        f = collect_factorization(root, tag)
        if f:
            factorizations[tag] = f
        t = collect_training(root, tag)
        if t:
            trainings[tag] = t

    data = collect_data(args.data)
    benchmark = collect_benchmark(bench_dir)

    markdown = render_markdown(data, factorizations, trainings, benchmark, bench_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as handle:
        handle.write(markdown)
    write_layer_csv(out_csv, factorizations)
    with open(out_json, "w") as handle:
        json.dump(
            {
                "root": os.path.abspath(root),
                "bench_dir": os.path.abspath(bench_dir),
                "data": data,
                "factorizations": factorizations,
                "trainings": trainings,
                "benchmark": benchmark,
            },
            handle, indent=2, ensure_ascii=False,
        )

    if not args.quiet:
        print(f"汇总完成：\n  {out_md}\n  {out_json}\n  {out_csv}")
        print(f"  覆盖 tag：{tags}")
        print(f"  数据：{'有' if data else '无'}；"
              f"分解：{sorted(factorizations) or '无'}；训练：{sorted(trainings) or '无'}；"
              f"模型级评测：{'有' if benchmark else '无'}")


if __name__ == "__main__":
    main()
