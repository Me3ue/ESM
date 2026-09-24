#!/usr/bin/env python
"""阶段 2：在保留非负约束的前提下，对分解出的 A/S/B 做若干批次的训练。

训练目标由三部分组成：

1. **蒸馏损失**（``--distill-weight``）：让分解模型在遮挡位置上的输出分布
   逼近原模型（``KL(softmax(logits_ref/T) || softmax(logits_nmf/T)) * T^2``）；
2. **重构损失**（``--recon-weight``）：保持 ``A S B + offset`` 与原始权重矩阵一致
   （相对 Frobenius 误差），防止分解因子在训练中漂移；
3. **交叉熵损失**（``--ce-weight``）：直接以真实氨基酸为标签，衡量模型自身的语言建模能力。

每一步梯度更新后，A/S/B 都会被投影回非负象限（clamp 到 >= 0），因此训练全程
保持三因子非负、中间方阵的约束。

示例
----
.. code-block:: bash

    python -m nmf.train_nmf \
        --layers "layers.5.fc1" \
        --fasta examples/data/some_proteins.fasta --max-records 64 \
        --batch-size 4 --max-batches 30 \
        --out nmf/outputs/nmf_trained.pt --history nmf/outputs/nmf_history.json
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .esm_common import (
    apply_mlm_mask,
    build_factorized_model,
    distribution_agreement_metrics,
    masked_lm_metrics,
    non_special_mask,
    read_fasta_records,
    resolve_device,
    set_seed,
    set_trainable_nmf_params,
    tokenize_records,
)
from .nmf_layers import ThreeFactorNMFLinear, get_submodule, relative_frobenius_error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="对 ESM 线性层的三因子非负分解做批次训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="esm2_t6_8M_UR50D", help="ESM 模型名")
    parser.add_argument("--hub-dir", default=None)
    parser.add_argument("--layers", default="layers.5.fc1", help="层选择表达式")
    parser.add_argument("--include-all-linear", action="store_true")
    parser.add_argument("--rank", type=int, default=None, help="中间方阵阶数（默认 min(in,out)）")
    parser.add_argument("--rank-ratio", type=float, default=None, help="秩比例，如 0.5")
    parser.add_argument("--init-checkpoint", default=None,
                        help="由 nmf.factorize 产出的分解检查点；不提供则现场分解")
    parser.add_argument("--solver", choices=["hals", "pgd", "mu"], default="hals",
                        help="离线分解阶段的求解器（hals 精度最高）")
    parser.add_argument("--iters", type=int, default=500, help="离线分解迭代次数")
    parser.add_argument("--n-inner", type=int, default=20, help="hals 中 S 的内层迭代次数")
    parser.add_argument("--init-lr", type=float, default=2e-2, help="离线分解的学习率（pgd）")
    parser.add_argument("--shift", choices=["min", "row", "zero", "none"], default="min")
    parser.add_argument("--init", choices=["nndsvd", "random"], default="nndsvd")

    parser.add_argument("--fasta", default="examples/data/some_proteins.fasta", help="训练序列")
    parser.add_argument("--max-records", type=int, default=64, help="最多使用的序列条数")
    parser.add_argument("--seq-max-len", type=int, default=512, help="过滤过长的序列")
    parser.add_argument("--seq-min-len", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4, help="每个批次的序列条数")
    parser.add_argument("--max-batches", type=int, default=30, help="最多训练多少个批次（优化步）")
    parser.add_argument("--epochs", type=int, default=1, help="按语料重复的轮数上限")
    parser.add_argument("--sort-by-length", action="store_true", default=True,
                        help="按长度排序分桶以减少 padding")
    parser.add_argument("--no-sort-by-length", dest="sort_by_length", action="store_false")
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096,
                        help="每批的最大 token 数（含 BOS/EOS）；变长序列下比固定条数高效。"
                             "设为 0 则退回按 --batch-size 条数分批")
    parser.add_argument("--kl-scope", choices=["masked", "all"], default="masked",
                        help="蒸馏损失统计范围：masked=仅遮挡位置；all=所有真实残基位置"
                             "（更强的保真目标，适合“性能不下降”实验）")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="每步允许的【等效权重相对移动量】上界（默认 1e-3 即 0.1%%）；"
                             "各因子组的 Adam 学习率 = lr / 该因子的扰动增益（见 --lr-scale）")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scale", choices=["auto", "none"], default="auto",
                        help="auto=按各因子的扰动增益自动缩放学习率（A/S/B 分组，"
                             "避免中间矩阵 S 的扰动被放大而破坏分解）；none=三组同 lr")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度 T")
    parser.add_argument("--mask-frac", type=float, default=0.15, help="MLM 遮挡比例")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--train-bias", action="store_true", help="同时训练线性层的 bias")
    parser.add_argument("--freeze-offset", action="store_true", help="不训练整体平移标量 offset")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值，0 表示关闭")
    parser.add_argument("--max-recon-degradation", type=float, default=0.3,
                        help="信任域：允许“相对 Frobenius 重构误差”相对初始值最多劣化的比例；"
                             "超出则回滚该步并把学习率减半；设为负数可关闭保护")
    parser.add_argument("--rollback-tries", type=int, default=3,
                        help="每步最多回滚重试次数（每次学习率减半）")

    parser.add_argument("--eval-every", type=int, default=5, help="每多少步做一次小评估；0 表示关闭")
    parser.add_argument("--save-every", type=int, default=0,
                        help="每多少步覆盖保存一次检查点（长时间训练建议设置，0 表示只在结束时保存）")
    parser.add_argument("--eval-fasta", default=None, help="评估用 FASTA；默认复用训练集尾部")
    parser.add_argument("--eval-records", type=int, default=8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu / cuda / auto")
    parser.add_argument("--out", default="nmf/outputs/nmf_trained.pt", help="训练后检查点")
    parser.add_argument("--history", default="nmf/outputs/nmf_history.json", help="训练历史 JSON")
    parser.add_argument("--csv", default=None, help="可选的训练历史 CSV")
    parser.add_argument("--plot", action="store_true", help="保存训练曲线 PNG")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def make_batches(
    records: List[Tuple[str, str]],
    batch_size: int,
    sort_by_length: bool = True,
    max_tokens_per_batch: int = 0,
    shuffle_seed: Optional[int] = None,
) -> List[List[Tuple[str, str]]]:
    """把序列切成批次。

    - ``max_tokens_per_batch > 0`` 时按 token 预算打包（变长序列下比固定条数高效得多，
      与 ``esm-extract`` 的 ``--toks_per_batch`` 思路一致），同时受 ``batch_size`` 上限约束；
    - ``sort_by_length`` 让同批长度相近，显著减少 padding；
    - ``shuffle_seed`` 用于每个 epoch 重新打乱，避免固定顺序带来的偏置。
    """
    records = list(records)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    if sort_by_length:
        records.sort(key=lambda kv: len(kv[1]))

    if not max_tokens_per_batch:
        return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

    batches: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    current_tokens = 0
    for record in records:
        tokens = len(record[1]) + 2  # BOS + EOS
        if current and (len(current) >= batch_size or current_tokens + tokens > max_tokens_per_batch):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(record)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def reconstruction_loss(
    fact_model,
    orig_model,
    layer_names,
) -> torch.Tensor:
    """所有被分解层的加权相对 Frobenius 误差（权重矩阵级别的重构损失）。"""
    total = None
    for name in layer_names:
        layer = get_submodule(fact_model, name)
        target = get_submodule(orig_model, name).weight.detach()
        approx = layer.effective_weight()
        err = torch.norm(target - approx) / torch.norm(target).clamp(min=1e-12)
        total = err if total is None else total + err
    if total is None:
        return torch.zeros((), requires_grad=True)
    return total / len(layer_names)


def factor_perturbation_gains(fact_model, layer_names) -> Dict[str, float]:
    """估计 A/S/B/offset 各自"逐元素扰动 δ 时层等效权重的最大变化"。

    三个因子都是非负矩阵，因此对**同号**扰动，等效权重 `A S B + offset` 的变化可以直接
    用求和上界刻画：

        Δ(A S B) <- ΔA · S B     ⇒ 增益 ≈ max_j Σ_k (S B)_{kj}
        Δ(A S B) <- A · ΔS · B   ⇒ 增益 ≈ (max_i Σ_k A_{ik}) × (max_l Σ_j B_{lj})
        Δ(A S B) <- A S · ΔB     ⇒ 增益 ≈ max_i Σ_l (A S)_{il}

    实测（``layers.0.self_attn.k_proj``，δ=1e-4 同号扰动）：S 的增益比 B 大 3 个数量级，
    即 S 的逐元素扰动被夹在中间的两个矩阵放大。**Adam 的步长与梯度幅值无关（只取符号），
    因此会把这种放大原样传给等效权重，一步就能破坏分解**——必须按增益给每个因子分组
    设置不同的学习率（见 ``build_parameter_groups``）。
    """
    gA = gS = gB = gO = 0.0
    for name in layer_names:
        layer = get_submodule(fact_model, name)
        A = layer.A.detach().float()
        S = layer.S.detach().float()
        B = layer.B.detach().float()
        gA = max(gA, float((S @ B).sum(dim=0).max().item()))
        gS = max(gS, float(A.sum(dim=1).max().item()) * float(B.sum(dim=0).max().item()))
        gB = max(gB, float((A @ S).sum(dim=1).max().item()))
        # offset 的第 i 个元素给整行同一个增量，Frobenius 意义下增益 ~ sqrt(in_features)
        gO = max(gO, float(layer.in_features) ** 0.5)
    return {"A": max(gA, 1e-8), "S": max(gS, 1e-8), "B": max(gB, 1e-8), "offset": max(gO, 1e-8)}


def build_parameter_groups(
    fact_model,
    layer_names,
    lr: float,
    train_offset: bool = True,
    train_bias: bool = False,
    lr_scale: str = "auto",
    verbose: bool = True,
) -> Tuple[List[Dict], Dict[str, float]]:
    """按因子分组构造优化器参数组，并对敏感因子按增益缩小学习率。

    ``lr_scale="auto"``（默认）时取 ``lr_factor = lr / gain_factor``：``lr`` 的含义是
    **每步允许的"等效权重相对移动量"上界**，除以该因子的扰动增益后，得到使各因子单步
    对 ``A S B`` 的扰动量级一致的组内学习率。``lr_scale="none"`` 时三组共用 ``lr``
    （即早期会训坏模型的做法，仅用于对照）。
    """
    groups: Dict[str, List] = {"A": [], "S": [], "B": [], "offset": [], "bias": []}
    for name in layer_names:
        layer = get_submodule(fact_model, name)
        groups["A"].append(layer.A)
        groups["S"].append(layer.S)
        groups["B"].append(layer.B)
        if train_offset and isinstance(layer.weight_offset, torch.nn.Parameter):
            groups["offset"].append(layer.weight_offset)
        if train_bias and layer.bias is not None:
            groups["bias"].append(layer.bias)

    gains = factor_perturbation_gains(fact_model, layer_names)
    param_groups: List[Dict] = []
    used_lrs: Dict[str, float] = {}
    for key, params in groups.items():
        if not params:
            continue
        if lr_scale == "auto" and key in gains:
            # lr 的含义：每步允许的"等效权重相对移动量"上界；除以增益得到该因子的步长
            group_lr = lr / gains[key]
        else:
            group_lr = lr
        used_lrs[key] = group_lr
        param_groups.append({"params": params, "lr": group_lr, "name": key})
    if verbose:
        print("因子分组学习率（按扰动增益标定，使各因子单步的等效权重移动量一致）：")
        for key in ("A", "S", "B", "offset", "bias"):
            if key in used_lrs:
                g = gains.get(key, float("nan"))
                print(f"  {key:7s} 增益≈{g:12.2f}   组内元素 {sum(p.numel() for p in groups[key]):>9,}   lr={used_lrs[key]:.3e}")
    return param_groups, used_lrs


def snapshot_factors(fact_model, layer_names) -> Dict[str, Tuple]:
    """保存 A/S/B/offset 的快照，用于信任域回滚。"""
    snap = {}
    for name in layer_names:
        layer = get_submodule(fact_model, name)
        snap[name] = (
            layer.A.detach().clone(),
            layer.S.detach().clone(),
            layer.B.detach().clone(),
            layer.weight_offset.detach().clone(),
        )
    return snap


@torch.no_grad()
def restore_factors(fact_model, snap: Dict[str, Tuple]) -> None:
    """把 A/S/B/offset 恢复到快照状态。"""
    for name, (a, s, b, off) in snap.items():
        layer = get_submodule(fact_model, name)
        layer.A.copy_(a)
        layer.S.copy_(s)
        layer.B.copy_(b)
        layer.weight_offset.copy_(off)


def scale_optimizer_lr(optimizer, factor: float) -> float:
    """把优化器的学习率乘以 ``factor``，返回新的学习率。"""
    new_lr = None
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * factor
        new_lr = group["lr"]
    return float(new_lr if new_lr is not None else 0.0)


@torch.no_grad()
def quick_eval(orig_model, fact_model, alphabet, records, mask_frac, seed, device) -> Dict:
    """在一个固定的小评估集上比较两模型的遮挡预测能力。"""
    if not records:
        return {}
    _, _, tokens = tokenize_records(alphabet, records)
    tokens = tokens.to(device)
    generator = torch.Generator().manual_seed(seed)
    masked, positions = apply_mlm_mask(tokens, alphabet, mask_frac, generator)

    logits_ref = orig_model(masked, repr_layers=[], return_contacts=False)["logits"]
    logits_nmf = fact_model(masked, repr_layers=[], return_contacts=False)["logits"]

    orig_metrics = masked_lm_metrics(logits_ref, tokens, positions)
    nmf_metrics = masked_lm_metrics(logits_nmf, tokens, positions)
    agree = distribution_agreement_metrics(logits_nmf, logits_ref, positions, targets=tokens)

    return {
        "orig_top1_acc": orig_metrics.get("top1_acc", float("nan")),
        "orig_top5_acc": orig_metrics.get("top5_acc", float("nan")),
        "orig_masked_ppl": orig_metrics.get("masked_perplexity", float("nan")),
        "nmf_top1_acc": nmf_metrics.get("top1_acc", float("nan")),
        "nmf_top5_acc": nmf_metrics.get("top5_acc", float("nan")),
        "nmf_masked_ppl": nmf_metrics.get("masked_perplexity", float("nan")),
        "kl_nmf_ref": agree.get("kl_b_a", float("nan")),
        "top1_agreement": agree.get("top1_agreement", float("nan")),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.rank is not None and args.rank_ratio is not None:
        raise SystemExit("--rank 与 --rank-ratio 只能指定一个。")

    # ---------------------------------------------------------------- 数据 --
    records = read_fasta_records(
        args.fasta,
        max_records=args.max_records,
        min_len=args.seq_min_len,
        max_len=args.seq_max_len,
        shuffle=True,
        seed=args.seed,
    )
    if not records:
        raise SystemExit(f"未能从 {args.fasta} 读到任何序列。")
    if not args.quiet:
        lengths = [len(s) for _, s in records]
        print(f"训练序列：{len(records)} 条，长度 {min(lengths)}~{max(lengths)}（来自 {args.fasta}）")

    eval_records: List[Tuple[str, str]] = []
    if args.eval_fasta:
        eval_records = read_fasta_records(
            args.eval_fasta, max_records=args.eval_records, max_len=args.seq_max_len, seed=args.seed
        )
    elif args.eval_every > 0:
        split = max(1, len(records) // 5)
        eval_records = records[-split:][: args.eval_records]

    # -------------------------------------------------------------- 模型 ---
    orig_model, fact_model, alphabet, layer_names, reports = build_factorized_model(
        model_name=args.model,
        layer_spec=args.layers,
        rank=args.rank,
        rank_ratio=args.rank_ratio,
        factor_ckpt=args.init_checkpoint,
        device=device,
        hub_dir=args.hub_dir,
        solver=args.solver,
        iters=args.iters,
        shift=args.shift,
        init=args.init,
        lr=args.init_lr,
        n_inner=args.n_inner,
        tol=1e-9,
        seed=args.seed,
        verbose=not args.quiet,
        only_transformer_blocks=not args.include_all_linear,
        trainable=False,
    )
    # 只训练 NMF 参数（+ 可选 offset / bias）
    trainable = set_trainable_nmf_params(
        fact_model,
        layer_names,
        train_offset=not args.freeze_offset,
        train_bias=args.train_bias,
    )
    n_trainable = sum(p.numel() for p in trainable)
    if not args.quiet:
        print(f"可训练参数：{n_trainable:,} 个（{len(layer_names)} 个分解层）")

    param_groups, group_lrs = build_parameter_groups(
        fact_model,
        layer_names,
        lr=args.lr,
        train_offset=not args.freeze_offset,
        train_bias=args.train_bias,
        lr_scale=args.lr_scale,
        verbose=not args.quiet,
    )
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    probe_batches = make_batches(
        records, args.batch_size, args.sort_by_length, args.max_tokens_per_batch, shuffle_seed=args.seed
    )
    if not args.quiet:
        avg_tokens = sum(len(s) + 2 for _, s in records) / max(len(records), 1)
        print(
            f"共 {len(probe_batches)} 个批次/轮（平均每批 "
            f"{sum(len(b) for b in probe_batches) / max(len(probe_batches),1):.1f} 条、"
            f"约 {avg_tokens:.0f} 残基/条），最多训练 {args.max_batches} 步（{args.epochs} 轮上限）"
        )

# --------------------------------------------------- 检查点保存（闭包） --
    def dump_checkpoint(steps_done: int, elapsed_s: float, is_final: bool) -> Dict:
        """把当前因子与训练历史写盘（周期性调用，保证中断也能拿到可用结果）。"""
        factors: Dict[str, Dict] = {}
        for name in layer_names:
            layer = get_submodule(fact_model, name)
            assert isinstance(layer, ThreeFactorNMFLinear)
            factors[name] = {
                "A": layer.A.detach().float().cpu(),
                "S": layer.S.detach().float().cpu(),
                "B": layer.B.detach().float().cpu(),
                "offset": layer.weight_offset.detach().float().cpu(),
                "bias": None if layer.bias is None else layer.bias.detach().float().cpu(),
                "rank": layer.rank,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "trainable_bias": args.train_bias,
                "trainable_offset": not args.freeze_offset,
            }

        stats = {}
        with torch.no_grad():
            for name in layer_names:
                layer = get_submodule(fact_model, name)
                target = get_submodule(orig_model, name).weight.detach()
                approx = layer.effective_weight()
                stats[name] = {
                    "relative_frobenius_error": relative_frobenius_error(target, approx),
                    "offset_mean": float(layer.weight_offset.mean().detach().cpu()),
                    "offset_std": float(layer.weight_offset.std(unbiased=False).detach().cpu())
                    if layer.weight_offset.numel() > 1 else 0.0,
                    "sparsity": layer.factor_stats()["sparsity"],
                    "n_params_original": int(target.numel()),
                    "n_params_factored": int(layer.A.numel() + layer.S.numel() + layer.B.numel()),
                }

        checkpoint = {
            "stage": "train",
            "model_name": args.model,
            "layers": factors,
            "factorization": reports,
            "final_stats": stats,
            "history": history,
            "eval_history": eval_history,
            "config": vars(args),
            "elapsed_seconds": elapsed_s,
            "steps_done": steps_done,
            "is_final": is_final,
            "torch_version": str(torch.__version__),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(checkpoint, args.out)
        if args.history:
            os.makedirs(os.path.dirname(os.path.abspath(args.history)), exist_ok=True)
            with open(args.history, "w") as handle:
                json.dump(
                    {"history": history, "eval_history": eval_history, "final_stats": stats},
                    handle, indent=2, ensure_ascii=False,
                )
        if args.csv and history:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            with open(args.csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
        return stats


    # ------------------------------------------------------------ 训练循环 --
    step = 0
    history: List[Dict] = []
    eval_history: List[Dict] = []
    start_time = time.time()

    with torch.no_grad():
        initial_recon = float(reconstruction_loss(fact_model, orig_model, layer_names).item())
    if args.max_recon_degradation >= 0:
        guard_limit = initial_recon * (1.0 + args.max_recon_degradation)
    else:
        guard_limit = None
    if not args.quiet:
        if guard_limit:
            print(
                f"初始相对重构误差 {initial_recon:.6f}；信任域上限 {guard_limit:.6f}"
                f"（允许劣化 {args.max_recon_degradation:.0%}，超出则回滚并降低学习率）"
            )
        else:
            print(f"初始相对重构误差 {initial_recon:.6f}；信任域保护已关闭")

    for epoch in range(args.epochs):
        epoch_batches = make_batches(
            records,
            args.batch_size,
            args.sort_by_length,
            args.max_tokens_per_batch,
            shuffle_seed=args.seed + epoch,   # 每轮重新打乱
        )
        for batch in epoch_batches:
            if step >= args.max_batches:
                break

            labels, seqs, tokens = tokenize_records(alphabet, batch)
            tokens = tokens.to(device)
            generator = torch.Generator().manual_seed(args.seed * 100003 + step)
            masked, positions = apply_mlm_mask(tokens, alphabet, args.mask_frac, generator)

            with torch.no_grad():
                logits_ref = orig_model(masked, repr_layers=[], return_contacts=False)["logits"]
            logits_nmf = fact_model(masked, repr_layers=[], return_contacts=False)["logits"]

            if args.kl_scope == "all":
                kl_positions = non_special_mask(tokens, alphabet)
            else:
                kl_positions = positions
            sel_nmf = logits_nmf[kl_positions].float()
            sel_ref = logits_ref[kl_positions].float()
            sel_true = tokens[positions]

            temperature = max(args.temperature, 1e-6)
            log_p_nmf_t = F.log_softmax(sel_nmf / temperature, dim=-1)
            with torch.no_grad():
                p_ref_t = F.softmax(sel_ref / temperature, dim=-1)
            loss_distill = F.kl_div(log_p_nmf_t, p_ref_t, reduction="batchmean") * (temperature ** 2)

            loss_ce = F.cross_entropy(logits_nmf[positions].float(), tokens[positions])
            loss_recon = reconstruction_loss(fact_model, orig_model, layer_names)

            loss = (
                args.distill_weight * loss_distill
                + args.ce_weight * loss_ce
                + args.recon_weight * loss_recon
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

            # ---- 信任域保护：限制“权重重构误差”的累计劣化 ----
            # Adam 的步长与参数尺度无关，学习率偏大时一步就可能把分解破坏掉。
            # 这里在每步更新后检查重构误差，超出上限就回滚并把学习率减半重试。
            snap = snapshot_factors(fact_model, layer_names) if guard_limit else None
            optimizer.step()
            for name in layer_names:
                get_submodule(fact_model, name).project_to_nonnegative_()

            rolled_back = False
            tries = 0
            with torch.no_grad():
                rel_fro = float(reconstruction_loss(fact_model, orig_model, layer_names).item())
            while guard_limit and rel_fro > guard_limit and tries < args.rollback_tries:
                restore_factors(fact_model, snap)
                scale_optimizer_lr(optimizer, 0.5)
                optimizer.step()
                for name in layer_names:
                    get_submodule(fact_model, name).project_to_nonnegative_()
                with torch.no_grad():
                    rel_fro = float(reconstruction_loss(fact_model, orig_model, layer_names).item())
                tries += 1
            if guard_limit and rel_fro > guard_limit:
                # 仍然劣化：直接放弃这一步
                restore_factors(fact_model, snap)
                with torch.no_grad():
                    rel_fro = float(reconstruction_loss(fact_model, orig_model, layer_names).item())
                rolled_back = True

            current_lr = float(optimizer.param_groups[0]["lr"])
            current_lrs = {g.get("name", f"g{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
            record = {
                "step": step + 1,
                "epoch": epoch + 1,
                "n_seq": len(batch),
                "n_tokens": int(tokens.numel()),
                "n_masked": int(positions.sum().item()),
                "n_kl": int(kl_positions.sum().item()),
                "loss": float(loss.item()),
                "loss_distill": float(loss_distill.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_recon": float(rel_fro),
                "lr": current_lr,
                "lrs": current_lrs,
                "lr_halved": int(tries),
                "rolled_back": rolled_back,
                "elapsed_s": time.time() - start_time,
            }
            history.append(record)

            if not args.quiet:
                note = ""
                if record["rolled_back"]:
                    note = "  [回滚]"
                elif record["lr_halved"]:
                    note = f"  [学习率减半 x{record['lr_halved']}]"
                print(
                    f"[step {record['step']:3d}] loss={record['loss']:.4f}  "
                    f"distill={record['loss_distill']:.4f}  ce={record['loss_ce']:.4f}  "
                    f"recon(rel_fro)={record['loss_recon']:.6f}  "
                    f"lr={record['lr']:.2e}  masked={record['n_masked']}{note}"
                )

            if args.save_every and (step + 1) % args.save_every == 0:
                dump_checkpoint(step + 1, time.time() - start_time, is_final=False)
                if not args.quiet:
                    print(f"          [已保存检查点：{args.out}]", flush=True)

            if args.eval_every and (step + 1) % args.eval_every == 0:
                metrics = quick_eval(
                    orig_model, fact_model, alphabet, eval_records,
                    args.mask_frac, args.seed, device,
                )
                if metrics:
                    metrics["step"] = step + 1
                    eval_history.append(metrics)
                    if not args.quiet:
                        print(
                            f"          评估：原始 top1={metrics['orig_top1_acc']:.4f} "
                            f"ppl={metrics['orig_masked_ppl']:.2f} | "
                            f"分解 top1={metrics['nmf_top1_acc']:.4f} "
                            f"ppl={metrics['nmf_masked_ppl']:.2f} | "
                            f"KL={metrics['kl_nmf_ref']:.4f} "
                            f"一致率={metrics['top1_agreement']:.4f}"
                        )
            step += 1
        if step >= args.max_batches:
            break

    elapsed = time.time() - start_time

    # ------------------------------------------------------------- 保存 ----

    final_stats = dump_checkpoint(len(history), elapsed, is_final=True)
    if not args.quiet:
        print(f"\n训练后检查点已保存：{args.out}（{len(history)} 个批次，耗时 {elapsed:.1f}s）")
        if args.history:
            print(f"训练历史已保存：{args.history}")
        if args.csv:
            print(f"训练历史 CSV 已保存：{args.csv}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            steps = [r["step"] for r in history]
            axes[0].plot(steps, [r["loss"] for r in history], label="total")
            axes[0].plot(steps, [r["loss_distill"] for r in history], label="distill")
            if any(r["loss_ce"] for r in history):
                axes[0].plot(steps, [r["loss_ce"] for r in history], label="ce")
            axes[0].plot(steps, [r["loss_recon"] for r in history], label="recon (rel_fro)")
            axes[0].set_xlabel("batch step")
            axes[0].set_ylabel("loss")
            axes[0].set_title("Training loss")
            axes[0].grid(alpha=0.3)
            axes[0].legend(fontsize=8)

            if eval_history:
                es = [e["step"] for e in eval_history]
                axes[1].plot(es, [e["orig_top1_acc"] for e in eval_history], marker="o", label="original top-1")
                axes[1].plot(es, [e["nmf_top1_acc"] for e in eval_history], marker="s", label="NMF top-1")
                axes[1].set_xlabel("batch step")
                axes[1].set_ylabel("masked top-1 accuracy")
                axes[1].set_title("Masked-residue top-1 accuracy")
                axes[1].grid(alpha=0.3)
                axes[1].legend(fontsize=8)
            fig.tight_layout()
            plot_path = os.path.splitext(args.out)[0] + "_training.png"
            fig.savefig(plot_path, dpi=150)
            if not args.quiet:
                print(f"训练曲线已保存：{plot_path}")
        except ImportError:
            print("[提示] 未安装 matplotlib，跳过训练曲线绘制。")


if __name__ == "__main__":
    main()
