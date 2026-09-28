#!/usr/bin/env python
"""阶段 1：对 ESM 中指定的线性层做三因子非负矩阵分解。

    W ≈ A @ S @ B + offset        A/S/B >= 0，S 为 rank x rank 方阵

示例
----
.. code-block:: bash

    # 列出可选的线性层
    python -m nmf.factorize --list-layers

    # 对第 5 层 FFN 的第一个线性层做满阶（方阵）非负分解
    python -m nmf.factorize --layers "layers.5.fc1" --out nmf/outputs/esm2_8M_fc1.pt

    # 压缩到一半秩，用投影梯度求解，并保存报告
    python -m nmf.factorize --layers "layers.5.fc1,layers.5.self_attn.out_proj" \
        --rank-ratio 0.5 --solver pgd --iters 800 --report nmf/outputs/factor_report.json
"""

import argparse
import json
import os
import time
from typing import Dict

import torch

from .esm_common import freeze_all, load_esm2, resolve_device, set_seed
from .nmf_layers import list_linear_layers, replace_linears_with_nmf


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ESM 线性层的三因子非负矩阵分解（W ≈ A S B，S 为方阵）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="esm2_t6_8M_UR50D", help="ESM 模型名")
    parser.add_argument("--hub-dir", default=None, help="torch.hub 缓存目录（可选）")
    parser.add_argument(
        "--layers",
        default="layers.5.fc1",
        help="层选择表达式：all/first/middle/last、精确名、通配符或逗号分隔组合",
    )
    parser.add_argument("--include-all-linear", action="store_true",
                        help="也允许选择 Transformer block 之外的线性层")
    parser.add_argument("--rank", type=int, default=None,
                        help="中间方阵的阶数 r；默认 min(in, out)")
    parser.add_argument("--rank-ratio", type=float, default=None,
                        help="秩比例（相对 min(in, out)），如 0.5；与 --rank 二选一")
    parser.add_argument("--solver", choices=["hals", "pgd", "mu"], default="hals",
                        help="hals=逐列/逐行精确 NNLS（默认，精度最高）；"
                             "pgd=Adam 投影梯度；mu=经典 Lee-Seung 乘性更新")
    parser.add_argument("--iters", type=int, default=500, help="求解迭代次数")
    parser.add_argument("--n-inner", type=int, default=20,
                        help="hals 中 S 的 FISTA 内层迭代次数")
    parser.add_argument("--lr", type=float, default=2e-2, help="pgd 学习率")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="pgd 梯度裁剪阈值")
    parser.add_argument("--tol", type=float, default=1e-8, help="收敛阈值（相邻误差相对变化）")
    parser.add_argument("--shift", choices=["min", "row", "zero", "none"], default="min",
                        help="让目标矩阵非负的方式：整体平移（默认）/逐行平移/截断负值/不处理")
    parser.add_argument("--init", choices=["nndsvd", "random"], default="nndsvd",
                        help="非负初始化方式")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                        help="cpu / cuda / auto。注意：HALS 求解器对每个因子逐列/逐行更新，"
                             "内含 Python 循环与频繁 CPU 同步，在 GPU 上未必比 CPU 快；"
                             "这是一次性离线成本，用 cpu 也完全可以")
    parser.add_argument("--out", default="nmf/outputs/nmf_factors.pt", help="分解结果保存路径")
    parser.add_argument("--report", default=None, help="分解报告 JSON 保存路径")
    parser.add_argument("--plot", action="store_true", help="保存乘性更新收敛曲线")
    parser.add_argument("--list-layers", action="store_true", help="只打印可选线性层后退出")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.list_layers:
        model, _ = load_esm2(args.model, hub_dir=args.hub_dir)
        print(f"{args.model} 中的线性层：")
        for name in list_linear_layers(model):
            module = dict(model.named_modules())[name]
            print(f"  {name:45s} {module.out_features:5d} x {module.in_features:5d}")
        return

    if args.rank is not None and args.rank_ratio is not None:
        raise SystemExit("--rank 与 --rank-ratio 只能指定一个。")

    # 先加载模型以解析出具体的层名（并搬到目标设备：--device 会被真正使用）
    model, alphabet = load_esm2(args.model, hub_dir=args.hub_dir)
    model = model.eval().to(device)
    freeze_all(model)
    from .esm_common import resolve_layer_names

    layer_names = resolve_layer_names(model, args.layers, not args.include_all_linear)
    if not args.quiet:
        print(f"模型：{args.model}（{sum(p.numel() for p in model.parameters()):,} 参数）")
        print(f"待分解层：{layer_names}")

    # 计算每层的目标秩
    ranks: Dict[str, int] = {}
    for name in layer_names:
        linear = dict(model.named_modules())[name]
        full_rank = min(linear.in_features, linear.out_features)
        if args.rank is not None:
            rank = args.rank
        elif args.rank_ratio is not None:
            rank = max(1, int(round(full_rank * args.rank_ratio)))
        else:
            rank = full_rank
        ranks[name] = rank

    start = time.time()
    reports = replace_linears_with_nmf(
        model,
        layer_names,
        rank=ranks,
        solver=args.solver,
        n_iter=args.iters,
        shift=args.shift,
        init=args.init,
        lr=args.lr,
        grad_clip=args.grad_clip,
        n_inner=args.n_inner,
        tol=args.tol,
        seed=args.seed,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start

    # 逐层保存因子
    factors: Dict[str, Dict] = {}
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        factors[name] = {
            "A": layer.A.detach().float().cpu(),
            "S": layer.S.detach().float().cpu(),
            "B": layer.B.detach().float().cpu(),
            "offset": layer.weight_offset.detach().float().cpu(),
            "bias": None if layer.bias is None else layer.bias.detach().float().cpu(),
            "rank": layer.rank,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
        }

    payload = {
        "stage": "factorize",
        "model_name": args.model,
        "layers": factors,
        "factorization": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in reports.items()},
        "history": {k: v.get("history", []) for k, v in reports.items()},
        "args": vars(args),
        "elapsed_seconds": elapsed,
        "torch_version": str(torch.__version__),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    if not args.quiet:
        print(f"\n分解结果已保存：{args.out}（耗时 {elapsed:.1f}s）")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(
                {
                    "model_name": args.model,
                    "layers": {
                        k: {kk: vv for kk, vv in v.items() if kk != "history"}
                        for k, v in reports.items()
                    },
                    "elapsed_seconds": elapsed,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        if not args.quiet:
            print(f"分解报告已保存：{args.report}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.2))
            for name, info in reports.items():
                history = info.get("history", [])
                if history:
                    ax.plot(range(1, len(history) + 1), history, label=name)
            ax.set_xlabel("iteration")
            ax.set_ylabel("relative Frobenius error")
            ax.set_yscale("log")
            ax.set_title("NMF (A S B) convergence")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_path = os.path.splitext(args.out)[0] + "_convergence.png"
            fig.savefig(plot_path, dpi=150)
            if not args.quiet:
                print(f"收敛曲线已保存：{plot_path}")
        except ImportError:
            print("[提示] 未安装 matplotlib，跳过收敛曲线绘制。")


if __name__ == "__main__":
    main()
