#!/usr/bin/env python
"""阶段 3：比较“三因子非负分解后的模型”与“原模型”的能力。

评估维度
--------
1. **层级别重构质量**：等效权重 ``A S B + offset`` 与原始权重的相对 Frobenius
   误差、余弦相似度、非零元素占比、参数量变化；
2. **输出分布一致性**：不遮挡任何残基时，两个模型在所有残基位置上的 KL / JS
   散度、top-1 / top-5 一致率、对真实氨基酸的对数概率相关系数；
3. **语言建模能力**：随机遮挡 15% 残基后，两模型在遮挡位置的 top-1 / top-5
   准确率、masked 困惑度；
4. **效率**：可训练参数量与前向耗时。

支持一次传入多个检查点（例如“未训练的分解”与“训练后的分解”），一并对比。

示例
----
.. code-block:: bash

    python -m nmf.evaluate \
        --checkpoint nmf/outputs/nmf_factors.pt nmf/outputs/nmf_trained.pt \
        --fasta examples/data/some_proteins.fasta --max-records 8 \
        --out-dir nmf/outputs/eval
"""

import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Tuple

import torch

from .esm_common import (
    apply_mlm_mask,
    attach_factors,
    distribution_agreement_metrics,
    freeze_all,
    load_checkpoint,
    load_esm2,
    masked_lm_metrics,
    non_special_mask,
    read_fasta_records,
    resolve_device,
    set_seed,
    timing_forward,
    tokenize_records,
)
from .nmf_layers import (
    cosine_similarity_matrix,
    get_submodule,
    relative_frobenius_error,
    swap_in_empty_nmf_modules,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="原模型 vs 三因子非负分解模型的能力对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--hub-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="一个或多个分解检查点（nmf.factorize / nmf.train_nmf 的产物）",
    )
    parser.add_argument("--names", nargs="*", default=None, help="与 --checkpoint 对应的显示名")
    parser.add_argument("--fasta", default="examples/data/some_proteins.fasta", help="评估序列")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--seq-max-len", type=int, default=512)
    parser.add_argument("--mask-frac", type=float, default=0.15)
    parser.add_argument("--mask-rounds", type=int, default=3, help="随机遮挡重复次数（取平均）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="nmf/outputs/eval")
    parser.add_argument("--timing-repeats", type=int, default=1, help="前向计时重复次数；0 表示跳过")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def make_variant(
    orig_model,
    ckpt_path: str,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict]:
    """从检查点构造一个分解模型（其余层与原模型完全一致）。"""
    ckpt = load_checkpoint(ckpt_path)
    if "layers" not in ckpt:
        raise KeyError(f"{ckpt_path} 不是有效的分解检查点（缺少 'layers' 字段）。")
    factors = ckpt["layers"]
    ranks = {name: int(payload["A"].shape[1]) for name, payload in factors.items()}
    model = copy.deepcopy(orig_model)
    swap_in_empty_nmf_modules(model, ranks)
    attach_factors(model, factors)
    model = model.eval().to(device)
    freeze_all(model)
    if verbose:
        print(f"  构造变体 {os.path.basename(ckpt_path)}：分解层 {list(ranks)}")
    return model, ckpt


def layer_level_metrics(orig_model, variant_model, layer_names) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with torch.no_grad():
        for name in layer_names:
            layer = get_submodule(variant_model, name)
            target = get_submodule(orig_model, name).weight.detach()
            approx = layer.effective_weight()
            out[name] = {
                "shape": list(target.shape),
                "rank": layer.rank,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "relative_frobenius_error": relative_frobenius_error(target, approx),
                "cosine_similarity": cosine_similarity_matrix(target, approx),
                "offset_mean": float(layer.weight_offset.mean().item()),
                "offset_std": float(layer.weight_offset.std(unbiased=False).item()) if layer.weight_offset.numel() > 1 else 0.0,
                "sparsity_A": float((layer.A.abs() < 1e-6).float().mean().item()),
                "sparsity_S": float((layer.S.abs() < 1e-6).float().mean().item()),
                "sparsity_B": float((layer.B.abs() < 1e-6).float().mean().item()),
                "n_params_original": int(target.numel()),
                "n_params_factored": int(layer.A.numel() + layer.S.numel() + layer.B.numel()),
                "nonnegative": bool(
                    (layer.A >= 0).all().item()
                    and (layer.S >= 0).all().item()
                    and (layer.B >= 0).all().item()
                ),
            }
    return out


@torch.no_grad()
def distribution_metrics(orig_model, variant_model, tokens, alphabet) -> Dict:
    positions = non_special_mask(tokens, alphabet)
    logits_ref = orig_model(tokens, repr_layers=[], return_contacts=False)["logits"]
    logits_var = variant_model(tokens, repr_layers=[], return_contacts=False)["logits"]
    metrics = distribution_agreement_metrics(logits_var, logits_ref, positions, targets=tokens)
    # 逐序列平均的 KL，避免长序列主导
    per_seq = []
    for i in range(tokens.shape[0]):
        pos_i = positions[i]
        if int(pos_i.sum()) == 0:
            continue
        m = distribution_agreement_metrics(
            logits_var[i : i + 1], logits_ref[i : i + 1], pos_i.unsqueeze(0), targets=tokens[i : i + 1]
        )
        per_seq.append(m.get("kl_b_a", float("nan")))
    metrics["kl_nmf_ref_per_seq_mean"] = float(torch.tensor(per_seq).mean().item()) if per_seq else float("nan")
    return metrics


@torch.no_grad()
def masked_metrics_over_rounds(
    orig_model, variant_model, tokens, alphabet, mask_frac: float, rounds: int, seed: int
) -> Dict:
    """多次随机遮挡取平均，报告两模型的遮挡预测能力。"""
    agg_orig: List[Dict] = []
    agg_var: List[Dict] = []
    for r in range(max(rounds, 1)):
        generator = torch.Generator().manual_seed(seed * 7919 + r)
        masked, positions = apply_mlm_mask(tokens, alphabet, mask_frac, generator)
        logits_ref = orig_model(masked, repr_layers=[], return_contacts=False)["logits"]
        logits_var = variant_model(masked, repr_layers=[], return_contacts=False)["logits"]
        agg_orig.append(masked_lm_metrics(logits_ref, tokens, positions))
        agg_var.append(masked_lm_metrics(logits_var, tokens, positions, logits_ref=logits_ref))

    def _mean(rows: List[Dict], key: str) -> float:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    return {
        "n_masked_per_round": agg_orig[0].get("n_masked", 0),
        "rounds": max(rounds, 1),
        "orig_top1_acc": _mean(agg_orig, "top1_acc"),
        "orig_top5_acc": _mean(agg_orig, "top5_acc"),
        "orig_masked_ppl": _mean(agg_orig, "masked_perplexity"),
        "nmf_top1_acc": _mean(agg_var, "top1_acc"),
        "nmf_top5_acc": _mean(agg_var, "top5_acc"),
        "nmf_masked_ppl": _mean(agg_var, "masked_perplexity"),
        "top1_agreement_on_masked": _mean(agg_var, "top1_agreement"),
        "kl_on_masked": _mean(agg_var, "kl_vs_reference"),
    }


def render_markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# ESM 线性层三因子非负分解：能力对比报告")
    lines.append("")
    lines.append(f"- 模型：`{report['model_name']}`")
    lines.append(f"- 评估序列：{report['n_records']} 条（来自 `{report['fasta']}`）")
    lines.append(f"- 遮挡比例：{report['mask_frac']}，重复 {report['mask_rounds']} 次取平均")
    lines.append(f"- 随机种子：{report.get('seed', '未记录')}（序列子集与遮挡位置都由它决定）")
    lines.append(f"- 序列长度上限：{report.get('seq_max_len', '未记录')}")
    lines.append(f"- 设备：`{report['device']}`，torch `{report['torch_version']}`")
    lines.append(f"- 原模型参数量：{report['orig_params']:,}")
    lines.append("")

    lines.append("## 1. 层级别重构质量")
    lines.append("")
    lines.append("| 变体 | 层 | 形状 | rank | 相对 Frobenius 误差 | 余弦相似度 | 参数量（原→分解） | 非负 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, variant in report["variants"].items():
        for layer_name, m in variant["layer_metrics"].items():
            shape = "x".join(str(s) for s in m["shape"])
            lines.append(
                f"| {name} | `{layer_name}` | {shape} | {m['rank']} | "
                f"{m['relative_frobenius_error']:.6f} | {m['cosine_similarity']:.6f} | "
                f"{m['n_params_original']:,} → {m['n_params_factored']:,} | "
                f"{'是' if m['nonnegative'] else '否'} |"
            )
    lines.append("")

    lines.append("## 2. 输出分布一致性（未遮挡）")
    lines.append("")
    lines.append("| 变体 | 位置数 | KL(分解‖原) | JS 散度 | top-1 一致率 | top-5 一致率 | 真实 token 对数似然相关 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, variant in report["variants"].items():
        d = variant["distribution"]
        lines.append(
            f"| {name} | {d.get('n_positions', 0)} | {d.get('kl_b_a', float('nan')):.5f} | "
            f"{d.get('js_divergence', float('nan')):.5f} | "
            f"{d.get('top1_agreement', float('nan')):.4f} | {d.get('top5_agreement', float('nan')):.4f} | "
            f"{d.get('pearson_true_token_logprob', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("## 3. 语言建模能力（遮挡 15% 残基）")
    lines.append("")
    lines.append("| 变体 | top-1 准确率 | top-5 准确率 | masked 困惑度 | 遮挡位 top-1 一致率 | 遮挡位 KL |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    orig_m = report["variants"][next(iter(report["variants"]))]["masked"]
    lines.append(
        f"| 原模型（基准） | {orig_m.get('orig_top1_acc', float('nan')):.4f} | "
        f"{orig_m.get('orig_top5_acc', float('nan')):.4f} | "
        f"{orig_m.get('orig_masked_ppl', float('nan')):.3f} | — | — |"
    )
    for name, variant in report["variants"].items():
        m = variant["masked"]
        lines.append(
            f"| {name} | {m.get('nmf_top1_acc', float('nan')):.4f} | "
            f"{m.get('nmf_top5_acc', float('nan')):.4f} | "
            f"{m.get('nmf_masked_ppl', float('nan')):.3f} | "
            f"{m.get('top1_agreement_on_masked', float('nan')):.4f} | "
            f"{m.get('kl_on_masked', float('nan')):.5f} |"
        )
    lines.append("")

    lines.append("## 4. 效率")
    lines.append("")
    lines.append("| 变体 | 分解层参数量 | 全模型参数量 | 相对原模型 | 前向耗时（秒/批） |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| 原模型 | — | {report['orig_params']:,} | 1.000x | "
        f"{report.get('orig_time', float('nan')):.4f} |"
    )
    for name, variant in report["variants"].items():
        lines.append(
            f"| {name} | {variant['n_params_factored_layers']:,} | {variant['n_params_total']:,} | "
            f"{variant['n_params_total'] / report['orig_params']:.3f}x | "
            f"{variant.get('time', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("> 说明：分解层按三个矩阵依次相乘实现（`x B^T S^T A^T`），数学上等价于使用")
    lines.append("> 乘积矩阵，但参数量由 `out×in` 变为 `out×r + r×r + r×in`；当 `r < min(in,out)`")
    lines.append("> 时参数量与乘加次数同时下降，`r = min(in,out)`（满阶方阵）时参数量上升、乘加次数不变。")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    records = read_fasta_records(
        args.fasta,
        max_records=args.max_records,
        max_len=args.seq_max_len,
        shuffle=True,
        seed=args.seed,
    )
    if not records:
        raise SystemExit(f"未能从 {args.fasta} 读到任何序列。")
    if not args.quiet:
        print(f"评估序列：{len(records)} 条（来自 {args.fasta}）")

    orig_model, alphabet = load_esm2(args.model, hub_dir=args.hub_dir)
    orig_model = orig_model.eval().to(device)
    freeze_all(orig_model)
    orig_params = sum(p.numel() for p in orig_model.parameters())

    _, _, tokens = tokenize_records(alphabet, records)
    tokens = tokens.to(device)

    names = args.names or [os.path.splitext(os.path.basename(p))[0] for p in args.checkpoint]
    if len(names) != len(args.checkpoint):
        raise SystemExit("--names 的数量必须与 --checkpoint 一致。")

    report: Dict = {
        "model_name": args.model,
        "fasta": args.fasta,
        "n_records": len(records),
        "mask_frac": args.mask_frac,
        "mask_rounds": args.mask_rounds,
        "seed": args.seed,
        "seq_max_len": args.seq_max_len,
        "config": vars(args),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "orig_params": orig_params,
        "variants": {},
    }

    if args.timing_repeats > 0:
        report["orig_time"] = timing_forward(orig_model, tokens, n_repeats=args.timing_repeats)
        if not args.quiet:
            print(f"原模型前向耗时：{report['orig_time']:.4f}s")

    for name, ckpt_path in zip(names, args.checkpoint):
        if not args.quiet:
            print(f"\n=== 变体 {name}（{ckpt_path}）===")
        variant_model, ckpt = make_variant(orig_model, ckpt_path, device, verbose=not args.quiet)
        layer_names = list(ckpt["layers"].keys())

        layer_metrics = layer_level_metrics(orig_model, variant_model, layer_names)
        dist = distribution_metrics(orig_model, variant_model, tokens, alphabet)
        masked = masked_metrics_over_rounds(
            orig_model, variant_model, tokens, alphabet,
            args.mask_frac, args.mask_rounds, args.seed,
        )
        n_params_factored_layers = sum(
            int(get_submodule(variant_model, ln).A.numel()
                + get_submodule(variant_model, ln).S.numel()
                + get_submodule(variant_model, ln).B.numel())
            for ln in layer_names
        )
        n_params_total = sum(p.numel() for p in variant_model.parameters())
        entry = {
            "checkpoint": ckpt_path,
            "stage": ckpt.get("stage", "unknown"),
            "layer_names": layer_names,
            "layer_metrics": layer_metrics,
            "distribution": dist,
            "masked": masked,
            "n_params_factored_layers": n_params_factored_layers,
            "n_params_total": n_params_total,
        }
        if args.timing_repeats > 0:
            entry["time"] = timing_forward(variant_model, tokens, n_repeats=args.timing_repeats)
        report["variants"][name] = entry

        if not args.quiet:
            for ln, m in layer_metrics.items():
                print(
                    f"  层 {ln}: rel_fro={m['relative_frobenius_error']:.6f} "
                    f"cos={m['cosine_similarity']:.6f} 非负={m['nonnegative']}"
                )
            print(
                f"  分布：KL={dist.get('kl_b_a', float('nan')):.5f} "
                f"JS={dist.get('js_divergence', float('nan')):.5f} "
                f"top1一致={dist.get('top1_agreement', float('nan')):.4f}"
            )
            print(
                f"  遮挡：原 top1={masked['orig_top1_acc']:.4f} / 分解 top1={masked['nmf_top1_acc']:.4f}；"
                f"原 ppl={masked['orig_masked_ppl']:.3f} / 分解 ppl={masked['nmf_masked_ppl']:.3f}"
            )

    # ------------------------------------------------------------- 保存 ----
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "report.json")
    md_path = os.path.join(args.out_dir, "report.md")
    csv_path = os.path.join(args.out_dir, "layer_metrics.csv")

    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with open(md_path, "w") as handle:
        handle.write(render_markdown(report))

    with open(csv_path, "w", newline="") as handle:
        fieldnames = [
            "variant", "layer", "shape", "rank", "relative_frobenius_error",
            "cosine_similarity", "offset_mean", "offset_std", "sparsity_A", "sparsity_S",
            "sparsity_B", "n_params_original", "n_params_factored", "nonnegative",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, variant in report["variants"].items():
            for layer_name, m in variant["layer_metrics"].items():
                row = {"variant": name, "layer": layer_name, "shape": "x".join(str(s) for s in m["shape"])}
                row.update({k: m[k] for k in fieldnames if k in m})
                writer.writerow(row)

    if not args.quiet:
        print(f"\n报告已保存：\n  {md_path}\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
