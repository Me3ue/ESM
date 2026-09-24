#!/usr/bin/env python
"""论文级评测：原模型 vs 三因子非负分解模型，在留出测试集上的严格对比。

与 ``nmf.evaluate``（快速对比、侧重逐层诊断）不同，本脚本按文献常用协议报告指标，
并把结果整理成可直接入表的 Markdown / CSV：

1. **Masked-LM**：随机遮挡 ``mask_frac`` 比例的真实残基，报告 top-1 / top-5 准确率与
   masked 困惑度，``--mask-rounds`` 次重复给 **mean ± std**；
2. **伪困惑度（pseudo-perplexity）**：标准单点遮挡协议——逐位置只遮挡一个残基，
   用 ``-log p(真实残基)`` 的平均值取指数，是最常被引用的自回归/掩码语言模型指标；
3. **分布一致性**：未遮挡输入下逐残基 ``KL(分解‖原)``、JS 散度、top-1/top-5 一致率、
   真实残基对数概率的 Pearson 相关；
4. **表征相似度**：逐残基表示的线性 CKA（Kornblith et al., 2019）与逐位置余弦相似度；
5. **层级别与效率**：等效权重的相对 Frobenius 误差/余弦相似度、参数量、线性层乘加次数。

示例
----
.. code-block:: bash

    python -m nmf.benchmark \
        --checkpoint 满阶=nmf/outputs/fullmodel/factors_fullrank.pt \
        --fasta data/processed/swissprot/test.fasta \
        --n-eval 256 --mask-rounds 5 --ppl-records 16 --ppl-max-len 256 \
        --out-dir nmf/outputs/benchmark_fullrank
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .esm_common import (
    apply_mlm_mask,
    distribution_agreement_metrics,
    freeze_all,
    load_esm2,
    masked_lm_metrics,
    non_special_mask,
    read_fasta_records,
    resolve_device,
    set_seed,
    timing_forward,
    tokenize_records,
)
from .evaluate import make_variant
from .nmf_layers import (
    ThreeFactorNMFLinear,
    cosine_similarity_matrix,
    get_submodule,
    relative_frobenius_error,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="论文级评测：原模型 vs 三因子非负分解模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--hub-dir", default=None)
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="一个或多个 <名称>=<检查点路径>；也可直接给路径（名称取文件名）。"
                             "使用 --from-dir 时不需要")
    parser.add_argument("--from-dir", default=None,
                        help="不加载模型、不重新评测：直接读取该目录下已保存的 base.json + "
                             "variants/*.json 重新渲染 benchmark.md/json/csv（用于报告丢失或改图）")
    parser.add_argument("--reuse-base", action="store_true",
                        help="复用 out-dir/base.json 里已算好的原模型指标（尤其伪困惑度，最耗时），"
                             "指纹不一致时自动重算")
    parser.add_argument("--reuse-variants", action="store_true",
                        help="已存在同名变体结果时直接复用而不重新评测（长评测被中断后接着跑）")
    parser.add_argument("--fasta", default="data/processed/swissprot/test.fasta",
                        help="留出测试集 FASTA")
    parser.add_argument("--n-eval", type=int, default=256, help="用于评测的序列条数")
    parser.add_argument("--seq-max-len", type=int, default=1022)
    parser.add_argument("--mask-frac", type=float, default=0.15)
    parser.add_argument("--mask-rounds", type=int, default=5)
    parser.add_argument("--ppl-records", type=int, default=16,
                        help="计算单点伪困惑度的序列条数（0 表示跳过）")
    parser.add_argument("--ppl-max-len", type=int, default=256,
                        help="伪困惑度的序列截断长度（控制耗时）。注意：只截断，不过滤序列，"
                             "因此长序列会被截到该长度而不是被排除在外")
    parser.add_argument("--ppl-batch", type=int, default=32,
                        help="伪困惑度每次前向并行遮挡的位置数")
    parser.add_argument("--cka-repr-layer", type=int, default=-1,
                        help="计算 CKA 的表示层（-1 表示最后一层；0 表示 embedding）")
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="nmf/outputs/benchmark")
    parser.add_argument("--plot", action="store_true", help="输出对比柱状图 PNG")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# 模型侧统计
# --------------------------------------------------------------------------- #
@torch.no_grad()
def layer_statistics(orig_model, variant_model, layer_names: Sequence[str]) -> Dict:
    rel_fros, cosines, params_orig, params_fact = [], [], 0, 0
    per_layer = []
    for name in layer_names:
        layer = get_submodule(variant_model, name)
        target = get_submodule(orig_model, name).weight.detach()
        approx = layer.effective_weight()
        rel = relative_frobenius_error(target, approx)
        cos = cosine_similarity_matrix(target, approx)
        n_fact = int(layer.A.numel() + layer.S.numel() + layer.B.numel())
        rel_fros.append(rel)
        cosines.append(cos)
        params_orig += int(target.numel())
        params_fact += n_fact
        per_layer.append({"layer": name, "shape": list(target.shape), "rank": layer.rank,
                          "relative_frobenius_error": rel, "cosine_similarity": cos,
                          "n_params_original": int(target.numel()), "n_params_factored": n_fact})
    return {
        "n_layers": len(layer_names),
        "mean_relative_frobenius_error": float(sum(rel_fros) / len(rel_fros)),
        "max_relative_frobenius_error": float(max(rel_fros)),
        "mean_cosine_similarity": float(sum(cosines) / len(cosines)),
        "min_cosine_similarity": float(min(cosines)),
        "n_params_original": params_orig,
        "n_params_factored": params_fact,
        "per_layer": per_layer,
    }


def linear_cca_style_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """线性 CKA（Kornblith et al., 2019）：对样本做特征中心化后的归一化内积。"""
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = (x.t() @ y).norm() ** 2
    xtx = (x.t() @ x).norm()
    yty = (y.t() @ y).norm()
    denom = xtx * yty
    if float(denom) == 0.0:
        return float("nan")
    return float((xty / denom).item())


@torch.no_grad()
def representation_similarity(
    orig_model, variant_model, tokens, alphabet, layer_idx: int
) -> Dict:
    """逐残基表示的 CKA 与逐位置余弦相似度（层号 -1 表示最后一层）。"""
    n_layers = orig_model.num_layers
    idx = n_layers if layer_idx < 0 else layer_idx
    mask = non_special_mask(tokens, alphabet)
    out_ref = orig_model(tokens, repr_layers=[idx], return_contacts=False)
    out_var = variant_model(tokens, repr_layers=[idx], return_contacts=False)
    ref = out_ref["representations"][idx][mask]   # (N_res, d)
    var = out_var["representations"][idx][mask]
    cos = F.cosine_similarity(ref.float(), var.float(), dim=-1)
    # 相对 L2 误差（逐残基表示）
    rel = (ref.float() - var.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp(min=1e-12)
    return {
        "repr_layer": idx,
        "n_residues": int(ref.shape[0]),
        "cka": linear_cca_style_cka(ref, var),
        "mean_cosine": float(cos.mean().item()),
        "mean_relative_l2": float(rel.mean().item()),
    }


# --------------------------------------------------------------------------- #
# 语言建模指标
# --------------------------------------------------------------------------- #
@torch.no_grad()
def masked_lm_rounds(
    orig_model, variant_model, tokens, alphabet, mask_frac: float, rounds: int, seed: int
) -> Dict:
    """多回合随机遮挡，报告 mean ± std。"""
    stats: Dict[str, List[float]] = {
        "orig_top1_acc": [], "orig_top5_acc": [], "orig_masked_perplexity": [],
        "nmf_top1_acc": [], "nmf_top5_acc": [], "nmf_masked_perplexity": [],
        "kl_nmf_ref": [], "top1_agreement": [],
    }
    for r in range(max(rounds, 1)):
        generator = torch.Generator().manual_seed(seed * 1000003 + r)
        masked, positions = apply_mlm_mask(tokens, alphabet, mask_frac, generator)
        logits_ref = orig_model(masked, repr_layers=[], return_contacts=False)["logits"]
        logits_nmf = variant_model(masked, repr_layers=[], return_contacts=False)["logits"]
        m_ref = masked_lm_metrics(logits_ref, tokens, positions)
        m_nmf = masked_lm_metrics(logits_nmf, tokens, positions, logits_ref=logits_ref)
        stats["orig_top1_acc"].append(m_ref["top1_acc"])
        stats["orig_top5_acc"].append(m_ref["top5_acc"])
        stats["orig_masked_perplexity"].append(m_ref["masked_perplexity"])
        stats["nmf_top1_acc"].append(m_nmf["top1_acc"])
        stats["nmf_top5_acc"].append(m_nmf["top5_acc"])
        stats["nmf_masked_perplexity"].append(m_nmf["masked_perplexity"])
        stats["kl_nmf_ref"].append(m_nmf["kl_vs_reference"])
        stats["top1_agreement"].append(m_nmf["top1_agreement"])

    out: Dict[str, float] = {"rounds": max(rounds, 1), "n_masked": m_ref["n_masked"]}
    for key, values in stats.items():
        t = torch.tensor(values, dtype=torch.float64)
        out[key] = float(t.mean().item())
        out[f"{key}_std"] = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
    return out


@torch.no_grad()
def pseudo_perplexity(
    model, records: Sequence[Tuple[str, str]], alphabet, device,
    max_len: int, batch: int,
) -> float:
    """标准单点伪困惑度：逐位置只遮挡一个残基，取 exp(平均 -log p(真实残基))。"""
    total_nll, total_n = 0.0, 0
    for label, seq in records:
        seq = seq[:max_len]
        if len(seq) < 2:
            continue
        _, _, tokens = tokenize_records(alphabet, [(label, seq)])
        tokens = tokens.to(device)
        base = tokens.expand(len(seq), -1).clone()
        for i in range(len(seq)):
            base[i, i + 1] = alphabet.mask_idx
        nll_sum = 0.0
        for start in range(0, len(seq), max(batch, 1)):
            chunk = base[start : start + max(batch, 1)]
            logits = model(chunk, repr_layers=[], return_contacts=False)["logits"]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for k in range(chunk.shape[0]):
                pos = start + k
                nll_sum += -log_probs[k, pos + 1, tokens[0, pos + 1]].item()
        total_nll += nll_sum
        total_n += len(seq)
    if total_n == 0:
        raise ValueError(
            "伪困惑度无可用序列：请检查 --ppl-records 是否大于 0、FASTA 是否为空。"
        )
    return float(torch.exp(torch.tensor(total_nll / total_n)).item())


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def render_markdown(report: Dict) -> str:
    lines: List[str] = []
    lines.append("# 论文级评测报告：原模型 vs 三因子非负分解模型")
    lines.append("")
    lines.append(f"- 模型：`{report['model_name']}`（原模型参数量 {report['orig_params']:,}）")
    lines.append(f"- 测试集：`{report['fasta']}`，取 {report['n_eval']} 条"
                 f"（总残基 {report['n_residues']:,}）")
    lines.append(f"- 遮挡：比例 {report['mask_frac']}，重复 {report['mask_rounds']} 回合（mean ± std）")
    lines.append(f"- 伪困惑度：{report['ppl_records']} 条序列、截断 {report['ppl_max_len']}，单点遮挡协议")
    lines.append(f"- 设备：`{report['device']}`，torch `{report['torch_version']}`，seed {report['seed']}")
    lines.append("")

    names = list(report["variants"].keys())

    lines.append("## 表 1 语言建模能力（Masked-LM，mean ± std）")
    lines.append("")
    lines.append("| 模型 | 遮挡位 top-1 ↑ | 遮挡位 top-5 ↑ | masked 困惑度 ↓ | 与原模型 top-1 一致率 | 与原模型 KL ↓ |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    base = report["variants"][names[0]]["masked_lm"]
    lines.append(
        f"| 原模型（基准） | {base['orig_top1_acc']:.4f} ± {base['orig_top1_acc_std']:.4f} | "
        f"{base['orig_top5_acc']:.4f} ± {base['orig_top5_acc_std']:.4f} | "
        f"{base['orig_masked_perplexity']:.3f} ± {base['orig_masked_perplexity_std']:.3f} | — | — |"
    )
    for name in names:
        m = report["variants"][name]["masked_lm"]
        lines.append(
            f"| {name} | {m['nmf_top1_acc']:.4f} ± {m['nmf_top1_acc_std']:.4f} | "
            f"{m['nmf_top5_acc']:.4f} ± {m['nmf_top5_acc_std']:.4f} | "
            f"{m['nmf_masked_perplexity']:.3f} ± {m['nmf_masked_perplexity_std']:.3f} | "
            f"{m['top1_agreement']:.4f} | {m['kl_nmf_ref']:.5f} |"
        )
    lines.append("")

    if "orig_pseudo_perplexity" in report:
        lines.append("## 表 2 伪困惑度（单点遮挡，越低越好）")
        lines.append("")
        lines.append("| 模型 | 伪困惑度 ↓ |")
        lines.append("| --- | --- |")
        base_ppl = report["orig_pseudo_perplexity"]
        lines.append(f"| 原模型（基准） | {base_ppl:.4f} |")
        for name in names:
            v = report["variants"][name]
            if "pseudo_perplexity" not in v:
                continue
            delta = v["pseudo_perplexity"] - base_ppl
            lines.append(f"| {name} | {v['pseudo_perplexity']:.4f}（{delta:+.4f}） |")
        lines.append("")

    lines.append("## 表 3 输出分布与表征相似度（未遮挡）")
    lines.append("")
    lines.append("| 模型 | KL(分解‖原) ↓ | JS 散度 ↓ | top-1 一致率 ↑ | top-5 一致率 ↑ | 真实残基对数概率相关 ↑ | 逐残基 CKA ↑ | 逐残基表示余弦 ↑ |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in names:
        d = report["variants"][name]["distribution"]
        r = report["variants"][name]["representation"]
        lines.append(
            f"| {name} | {d['kl_b_a']:.5f} | {d['js_divergence']:.5f} | {d['top1_agreement']:.4f} | "
            f"{d['top5_agreement']:.4f} | {d['pearson_true_token_logprob']:.4f} | "
            f"{r['cka']:.5f} | {r['mean_cosine']:.5f} |"
        )
    lines.append("")

    lines.append("## 表 4 层级别重构与效率")
    lines.append("")
    lines.append("| 模型 | 分解层数 | 平均相对 Frobenius 误差 ↓ | 最大相对误差 | 平均余弦 ↑ | 分解层参数量（原 → 分解） | 全模型参数量 | 线性层乘加比 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(f"| 原模型 | — | — | — | — | — | {report['orig_params']:,} | 1.000× |")
    for name in names:
        v = report["variants"][name]
        ls = v["layer_stats"]
        lines.append(
            f"| {name} | {ls['n_layers']} | {ls['mean_relative_frobenius_error']:.5f} | "
            f"{ls['max_relative_frobenius_error']:.5f} | {ls['mean_cosine_similarity']:.5f} | "
            f"{ls['n_params_original']:,} → {ls['n_params_factored']:,} | "
            f"{v['n_params_total']:,} | {v['linear_mac_ratio']:.3f}× |"
        )
    lines.append("")
    lines.append("> 说明：CKA 为 Kornblith et al. (2019) 的线性 CKA，在全部真实残基位置上计算，")
    lines.append("> 1.0 表示表征完全一致。乘加比统计的是所有线性层的乘加次数之比")
    lines.append("> （分解层按 `x·B^T·S^T·A^T` 三次矩阵乘计算，即 `in·r + r·r + r·out`）。")
    lines.append("")
    return "\n".join(lines)


def count_linear_names(model) -> List[str]:
    """模型中所有线性层（原始 nn.Linear 或已替换的 NMF 层）的限定名。"""
    return [
        name
        for name, mod in model.named_modules()
        if name and isinstance(mod, (torch.nn.Linear, ThreeFactorNMFLinear))
    ]


def linear_macs(model, tokens_per_layer: int, layer_names: Sequence[str]) -> float:
    """统计指定线性层在给定 token 数下的乘加次数。"""
    total = 0.0
    for name in layer_names:
        m = get_submodule(model, name)
        if isinstance(m, ThreeFactorNMFLinear):
            total += tokens_per_layer * (m.in_features * m.rank + m.rank * m.rank + m.rank * m.out_features)
        else:
            total += tokens_per_layer * m.in_features * m.out_features
    return total


def parse_checkpoints(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path, name = item, os.path.splitext(os.path.basename(item))[0]
        out.append((name, path))
    return out


# --------------------------------------------------------------------------- #
# 增量落盘（长评测被中断时不丢结果）
# --------------------------------------------------------------------------- #
BASE_NAME = "base.json"
VARIANTS_DIR = "variants"


def _safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name) or "variant"


def variant_json_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, VARIANTS_DIR, _safe_name(name) + ".json")


def dump_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def fingerprint(args) -> Dict:
    """评测口径指纹：只有这些参数全一致，复用已有结果才是合法的。"""
    return {
        "model_name": args.model,
        "fasta": os.path.abspath(args.fasta),
        "n_eval": args.n_eval,
        "seq_max_len": args.seq_max_len,
        "mask_frac": args.mask_frac,
        "mask_rounds": args.mask_rounds,
        "ppl_records": args.ppl_records,
        "ppl_max_len": args.ppl_max_len,
        "cka_repr_layer": args.cka_repr_layer,
        "seed": args.seed,
    }


def _fingerprint_of(out_dir: str, filename: str) -> Optional[Dict]:
    payload = load_json(os.path.join(out_dir, VARIANTS_DIR, filename))
    return None if payload is None else payload.get("fingerprint")


def select_consistent_variants(
    out_dir: str, base_fp: Optional[Dict], quiet: bool = False
) -> Dict[str, Dict]:
    """只挑出与基准口径（``base.json`` 的 fingerprint）一致的变体结果。

    评测目录里可能残留着用**别的参数**跑出来的变体文件（例如先用 ``--n-eval 96`` 跑了一个，
    后来又用 ``--n-eval 64`` 跑另一个）。直接把它们混进同一张表会得到不可比的结果，因此这里
    以 ``base.json`` 的口径为准筛掉不一致的文件，并明确告知。
    """
    directory = os.path.join(out_dir, VARIANTS_DIR)
    if not os.path.isdir(directory):
        return {}
    names = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
    if not names:
        return {}

    fingerprints: Dict[str, Optional[Dict]] = {f: _fingerprint_of(out_dir, f) for f in names}
    if base_fp is None:
        # 没有 base.json 时退化为"要求所有变体口径一致"
        distinct = {json.dumps(fp, sort_keys=True, ensure_ascii=False) for fp in fingerprints.values()}
        if len(distinct) > 1:
            raise SystemExit(
                "该目录下的变体结果口径不一致，无法合成一张可比的表：\n  "
                + "\n  ".join(f"{f}: {fingerprints[f]}" for f in names)
                + "\n请删除口径不符的文件，或用同一组参数重新评测。"
            )

    kept: Dict[str, Dict] = {}
    skipped: List[str] = []
    for filename in names:
        if base_fp is not None and fingerprints[filename] != base_fp:
            skipped.append(filename)
            continue
        payload = load_json(os.path.join(directory, filename))
        entry = payload.get("entry", payload)
        name = payload.get("name") or os.path.splitext(filename)[0]
        kept[name] = entry
    if skipped and not quiet:
        print(f"[警告] 以下变体与 base.json 的评测口径不一致，已跳过：{skipped}")
        print("        （如需对比，请用同一组参数重新评测或另建目录）")
    return kept


def save_variant(out_dir: str, name: str, fp: Dict, entry: Dict) -> str:
    """立即把单个变体的评测结果落盘（`<out_dir>/variants/<名称>.json`）。

    长评测往往要跑几十分钟，一旦中途被中断，只在最后统一写盘的做法会让所有结果丢失；
    这里每个变体算完就写一次，配合 ``--reuse-variants`` 可以接着跑。
    """
    path = variant_json_path(out_dir, name)
    dump_json(path, {
        "name": name,
        "fingerprint": fp,
        "checkpoint": entry.get("checkpoint"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry": entry,
    })
    return path


def load_variant(out_dir: str, name: str, fp: Dict, ckpt_path: str) -> Optional[Dict]:
    """读取可复用的变体结果；口径（fingerprint）或检查点不一致时返回 None。"""
    wrapper = load_json(variant_json_path(out_dir, name))
    if not wrapper:
        return None
    if wrapper.get("fingerprint") != fp:
        return None
    if wrapper.get("checkpoint") != ckpt_path:
        return None
    return wrapper.get("entry")


def print_variant_summary(name: str, entry: Dict, orig_params: int, report: Dict) -> None:
    m, d, r = entry["masked_lm"], entry["distribution"], entry["representation"]
    print(
        f"  Masked-LM：原 top-1 {m['orig_top1_acc']:.4f} / 分解 {m['nmf_top1_acc']:.4f}；"
        f"原 ppl {m['orig_masked_perplexity']:.3f} / 分解 {m['nmf_masked_perplexity']:.3f}"
    )
    if "pseudo_perplexity" in entry:
        print(f"  伪困惑度：{entry['pseudo_perplexity']:.4f} "
              f"（原 {report.get('orig_pseudo_perplexity', float('nan')):.4f}）")
    print(f"  分布一致：KL {d['kl_b_a']:.5f}，top-1 一致 {d['top1_agreement']:.4f}，"
          f"CKA {r['cka']:.5f}，表示余弦 {r['mean_cosine']:.5f}")
    print(f"  层级别：平均 rel_fro {entry['layer_stats']['mean_relative_frobenius_error']:.5f}，"
          f"最大 {entry['layer_stats']['max_relative_frobenius_error']:.5f}；"
          f"参数 {orig_params:,} -> {entry['n_params_total']:,}；"
          f"乘加比 {entry['linear_mac_ratio']:.3f}×")


def render_report(report: Dict, out_dir: str, plot: bool = False, quiet: bool = False) -> None:
    """把内存中的 report 写成 benchmark.json / benchmark.md / layer_stats.csv（+ 可选 PNG）。

    该函数是纯渲染，不接触模型，因此可以拿 ``--from-dir`` 从已落盘的中间结果重新生成报告。
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "benchmark.json")
    md_path = os.path.join(out_dir, "benchmark.md")
    csv_path = os.path.join(out_dir, "layer_stats.csv")
    if not report.get("variants"):
        raise SystemExit(f"没有任何变体结果，无法生成报告（目录：{out_dir}）")

    dump_json(json_path, report)
    with open(md_path, "w") as handle:
        handle.write(render_markdown(report))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "layer", "shape", "rank", "relative_frobenius_error",
                         "cosine_similarity", "n_params_original", "n_params_factored"])
        for name, v in report["variants"].items():
            for row in v["layer_stats"]["per_layer"]:
                writer.writerow([name, row["layer"], "x".join(map(str, row["shape"])), row["rank"],
                                 f"{row['relative_frobenius_error']:.6f}",
                                 f"{row['cosine_similarity']:.6f}",
                                 row["n_params_original"], row["n_params_factored"]])

    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = ["原模型"] + list(report["variants"].keys())
            base = report["variants"][list(report["variants"].keys())[0]]["masked_lm"]
            panels = [
                ("Masked top-1 acc.", [base["orig_top1_acc"]] + [
                    report["variants"][n]["masked_lm"]["nmf_top1_acc"] for n in report["variants"]]),
                ("Masked perplexity", [base["orig_masked_perplexity"]] + [
                    report["variants"][n]["masked_lm"]["nmf_masked_perplexity"] for n in report["variants"]]),
                ("Pseudo-perplexity", [report.get("orig_pseudo_perplexity", float("nan"))] + [
                    report["variants"][n].get("pseudo_perplexity", float("nan")) for n in report["variants"]]),
                ("Per-residue CKA", [1.0] + [
                    report["variants"][n]["representation"]["cka"] for n in report["variants"]]),
            ]
            fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.8))
            for ax, (title, values) in zip(axes, panels):
                ax.bar(range(len(values)), values, color=["#888888"] + ["#c0392b"] * (len(values) - 1))
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(names, rotation=20, fontsize=8)
                ax.set_title(title, fontsize=10)
                ax.grid(axis="y", alpha=0.3)
            fig.suptitle(f"NMF vs original - {report['model_name']} (test n={report['n_eval']})", fontsize=11)
            fig.tight_layout()
            plot_path = os.path.join(out_dir, "benchmark.png")
            fig.savefig(plot_path, dpi=150)
            if not quiet:
                print(f"对比图已保存：{plot_path}")
        except ImportError:
            print("[提示] 未安装 matplotlib，跳过绘图。")

    if not quiet:
        print(f"\n报告已保存：\n  {md_path}\n  {json_path}\n  {csv_path}")


def rerender_from_dir(from_dir: str, plot: bool = False, quiet: bool = False) -> None:
    """从 ``--from-dir`` 目录里的 base.json + variants/*.json 重新渲染报告。"""
    base = load_json(os.path.join(from_dir, BASE_NAME))
    if base is None:
        raise SystemExit(
            f"{from_dir} 下没有 {BASE_NAME}；该目录不是本脚本 --out-dir 产生的评测目录。"
        )
    variants = select_consistent_variants(from_dir, base.get("fingerprint"), quiet=quiet)
    if not variants:
        raise SystemExit(f"{from_dir}/{VARIANTS_DIR}/ 下没有与 base.json 口径一致的变体结果。")
    report = dict(base)
    report["variants"] = variants
    if not quiet:
        print(f"从 {from_dir} 读取 {len(variants)} 个变体：{list(variants)}")
    render_report(report, from_dir, plot, quiet)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    # ---- 纯渲染模式：不加载模型，从已落盘的中间结果重建报告 ----
    if args.from_dir:
        rerender_from_dir(args.from_dir, plot=args.plot, quiet=args.quiet)
        return

    if not args.checkpoint:
        raise SystemExit("必须提供 --checkpoint（或用 --from-dir 重新渲染已有结果）。")

    device = resolve_device(args.device)

    records = read_fasta_records(
        args.fasta, max_records=args.n_eval, max_len=args.seq_max_len, shuffle=True, seed=args.seed
    )
    if not records:
        raise SystemExit(f"未能从 {args.fasta} 读到任何序列。")
    n_residues = sum(len(s) for _, s in records)
    if not args.quiet:
        print(f"测试集：{len(records)} 条，共 {n_residues:,} 个残基（来自 {args.fasta}）")

    orig_model, alphabet = load_esm2(args.model, hub_dir=args.hub_dir)
    orig_model = orig_model.eval().to(device)
    freeze_all(orig_model)
    orig_params = sum(p.numel() for p in orig_model.parameters())

    _, _, tokens = tokenize_records(alphabet, records)
    tokens = tokens.to(device)

    checkpoints = parse_checkpoints(args.checkpoint)
    print_fingerprint = fingerprint(args)
    report: Dict = {
        "model_name": args.model,
        "fasta": args.fasta,
        "n_eval": len(records),
        "n_residues": n_residues,
        "seq_max_len": args.seq_max_len,
        "mask_frac": args.mask_frac,
        "mask_rounds": args.mask_rounds,
        "ppl_records": args.ppl_records,
        "ppl_max_len": args.ppl_max_len,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "seed": args.seed,
        "orig_params": orig_params,
        "fingerprint": print_fingerprint,
        "config": vars(args),
        "variants": {},
    }

    ppl_records: List[Tuple[str, str]] = []
    if args.ppl_records > 0:
        # 注意：这里只按条数抽样，**不按长度过滤**——长序列在 pseudo_perplexity 内部被截断到
        # --ppl-max-len。早期版本把 ppl_max_len 同时当成过滤条件，会让伪困惑度只统计短序列
        # （若测试集最短序列都长于该值，还会静默变成 nan）。
        ppl_records = read_fasta_records(
            args.fasta, max_records=args.ppl_records, shuffle=True, seed=args.seed + 1,
        )

    # ---- 复用已算好的原模型指标（伪困惑度最耗时，中断续跑时非常有用）----
    cached_base = load_json(os.path.join(args.out_dir, BASE_NAME)) if args.reuse_base else None
    base_reused = False
    if cached_base is not None:
        if cached_base.get("fingerprint") == print_fingerprint:
            for key, value in cached_base.items():
                if key not in ("variants", "config", "fingerprint"):
                    report[key] = value
            base_reused = True
            if not args.quiet:
                print(f"复用 {os.path.join(args.out_dir, BASE_NAME)} 中已算好的原模型指标"
                      f"（伪困惑度 {report.get('orig_pseudo_perplexity', float('nan')):.4f}）")
        elif not args.quiet:
            print(f"[警告] {os.path.join(args.out_dir, BASE_NAME)} 的评测口径与本次不一致，忽略缓存重算。")

    if not base_reused:  # 原模型的伪困惑度
        if args.ppl_records > 0:
            if not args.quiet:
                print(f"计算原模型伪困惑度（{len(ppl_records)} 条 × 单点遮挡）…")
            report["orig_pseudo_perplexity"] = pseudo_perplexity(
                orig_model, ppl_records, alphabet, device, args.ppl_max_len, args.ppl_batch
            )
            if not args.quiet:
                print(f"  原模型伪困惑度 = {report['orig_pseudo_perplexity']:.4f}")

        if args.timing_repeats > 0:
            report["orig_time"] = timing_forward(orig_model, tokens, n_repeats=args.timing_repeats)
        n_tokens_eval = int(tokens.numel())
        report["n_tokens_eval"] = n_tokens_eval
        report["orig_linear_macs"] = linear_macs(
            orig_model, n_tokens_eval, count_linear_names(orig_model)
        )
        # 立即落盘：后续变体评测即使被中断，原模型基准也不用重算
        dump_json(os.path.join(args.out_dir, BASE_NAME), report)
    n_tokens_eval = report["n_tokens_eval"]

    for name, ckpt_path in checkpoints:
        if args.reuse_variants:
            cached = load_variant(args.out_dir, name, print_fingerprint, ckpt_path)
            if cached is not None:
                report["variants"][name] = cached
                if not args.quiet:
                    print(f"\n=== 复用变体 {name}（{ckpt_path}）的已有结果 ===")
                    print_variant_summary(name, cached, orig_params, report)
                continue

        if not args.quiet:
            print(f"\n=== 评测 {name}（{ckpt_path}）===")
        variant_model, ckpt = make_variant(orig_model, ckpt_path, device, verbose=not args.quiet)
        layer_names = list(ckpt["layers"].keys())

        entry: Dict = {"checkpoint": ckpt_path, "stage": ckpt.get("stage", "unknown")}
        entry["layer_stats"] = layer_statistics(orig_model, variant_model, layer_names)
        entry["masked_lm"] = masked_lm_rounds(
            orig_model, variant_model, tokens, alphabet, args.mask_frac, args.mask_rounds, args.seed
        )
        entry["distribution"] = distribution_agreement_metrics(
            variant_model(tokens, repr_layers=[], return_contacts=False)["logits"],
            orig_model(tokens, repr_layers=[], return_contacts=False)["logits"],
            non_special_mask(tokens, alphabet),
            targets=tokens,
        )
        entry["representation"] = representation_similarity(
            orig_model, variant_model, tokens, alphabet, args.cka_repr_layer
        )
        entry["n_params_total"] = sum(p.numel() for p in variant_model.parameters())
        if args.ppl_records > 0:
            entry["pseudo_perplexity"] = pseudo_perplexity(
                variant_model, ppl_records, alphabet, device, args.ppl_max_len, args.ppl_batch
            )
        if args.timing_repeats > 0:
            entry["time"] = timing_forward(variant_model, tokens, n_repeats=args.timing_repeats)
        entry["linear_macs"] = linear_macs(variant_model, n_tokens_eval, count_linear_names(variant_model))
        entry["linear_mac_ratio"] = entry["linear_macs"] / max(report["orig_linear_macs"], 1e-9)
        report["variants"][name] = entry
        variant_path = save_variant(args.out_dir, name, print_fingerprint, entry)

        if not args.quiet:
            print_variant_summary(name, entry, orig_params, report)
            print(f"  [已保存 {variant_path}]")

    render_report(report, args.out_dir, args.plot, args.quiet)


if __name__ == "__main__":
    main()
