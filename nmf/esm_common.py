"""ESM 模型加载、FASTA 读取、MLM 掩码与评估指标等公共工具。

供 ``nmf/factorize.py``、``nmf/train_nmf.py``、``nmf/evaluate.py`` 共用。
"""

import copy
import fnmatch
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nmf_layers import (
    ThreeFactorNMFLinear,
    get_submodule,
    list_linear_layers,
    replace_linears_with_nmf,
    swap_in_empty_nmf_modules,
)


# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """解析设备字符串：``auto``/``cpu``/``cuda``/``cuda:0``。"""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求了 CUDA 但当前不可用，回退到 CPU。")
        return torch.device("cpu")
    return device


def load_esm2(model_name: str = "esm2_t6_8M_UR50D", hub_dir: Optional[str] = None):
    """加载 ESM-2 模型与字母表（权重缺失时自动下载并缓存）。"""
    import esm

    if hub_dir:
        torch.hub.set_dir(hub_dir)
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    return model, alphabet


def save_checkpoint(path: str, payload: Dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict:
    """加载分解检查点，兼容 PyTorch >= 2.6 默认 ``weights_only=True`` 的行为。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# FASTA
# --------------------------------------------------------------------------- #
def read_fasta_records(
    path: str,
    max_records: Optional[int] = None,
    min_len: int = 0,
    max_len: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """读取 FASTA，返回 ``[(label, sequence), ...]``。

    ``min_len``/``max_len`` 按序列长度过滤；``shuffle`` 用于构造训练/评估子集。
    """
    records: List[Tuple[str, str]] = []
    label: Optional[str] = None
    chunks: List[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    records.append((label, "".join(chunks)))
                label = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if label is not None:
            records.append((label, "".join(chunks)))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    kept: List[Tuple[str, str]] = []
    for lab, seq in records:
        if len(seq) < min_len:
            continue
        if max_len is not None and len(seq) > max_len:
            continue
        kept.append((lab, seq))
        if max_records is not None and len(kept) >= max_records:
            break
    return kept


# --------------------------------------------------------------------------- #
# 层选择
# --------------------------------------------------------------------------- #
def resolve_layer_names(
    model: torch.nn.Module,
    spec: str,
    only_transformer_blocks: bool = True,
) -> List[str]:
    """把用户给出的层选择表达式展开成具体的 ``nn.Linear`` 限定名列表。

    支持：

    - ``all``：所有（Transformer block 内的）线性层；
    - ``first`` / ``middle`` / ``last``：第一个、中间、最后一个 block 内的线性层，
      也可带子模式，如 ``first.fc1``、``last.self_attn.*``；
    - ``layers.5.fc1``：精确名字；
    - ``layers.*.fc1``：通配符；
    - ``fc1``：后缀匹配；
    - 逗号分隔组合，如 ``"layers.0.fc1,layers.0.self_attn.out_proj"``。
    """
    names = list_linear_layers(model)
    block_names = [n for n in names if n.startswith("layers.")]
    pool = block_names if (only_transformer_blocks and block_names) else names

    def _block_indices(name: str) -> Optional[int]:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            return int(parts[1])
        return None

    selected: List[str] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "all":
            selected.extend(pool)
            continue
        # first / middle / last，也支持带子模式的写法：first.fc1、last.self_attn.* 等
        keyword = next(
            (k for k in ("first", "middle", "last") if token == k or token.startswith(k + ".")),
            None,
        )
        if keyword is not None:
            sub = token[len(keyword) + 1 :]
            indices = sorted({i for i in (_block_indices(n) for n in pool) if i is not None})
            if not indices:
                raise ValueError("模型中未找到名为 layers.N.* 的层，无法使用 first/middle/last。")
            target_index = {
                "first": indices[0],
                "middle": indices[len(indices) // 2],
                "last": indices[-1],
            }[keyword]
            base = [n for n in pool if _block_indices(n) == target_index]
            if sub:
                block_prefix = f"layers.{target_index}."
                base = [
                    n
                    for n in base
                    if fnmatch.fnmatch(n[len(block_prefix) :], sub)
                    or fnmatch.fnmatch(n, sub)
                    or n.endswith("." + sub)
                ]
                if not base:
                    raise ValueError(
                        f"层选择 '{token}' 没有匹配到任何线性层。"
                        f"layers.{target_index} 下的线性层：{sorted(base) or '无'}"
                    )
            selected.extend(base)
            continue
        if any(ch in token for ch in "*?["):
            matches = [n for n in pool if fnmatch.fnmatch(n, token)]
        elif token in pool:
            matches = [token]
        else:
            matches = [n for n in pool if n.endswith("." + token) or n == token]
        if not matches:
            raise ValueError(
                f"层选择 '{token}' 没有匹配到任何线性层。可用层示例：{pool[:6]}"
            )
        selected.extend(matches)

    # 去重并保持 model.named_modules() 的顺序
    ordered: List[str] = []
    for name in names:
        if name in set(selected) and name not in ordered:
            ordered.append(name)
    return ordered


# --------------------------------------------------------------------------- #
# token 化与掩码
# --------------------------------------------------------------------------- #
def tokenize_records(alphabet, records: Sequence[Tuple[str, str]], truncation_seq_length: int = 1022):
    """``[(label, sequence)]`` -> ``(labels, strings, tokens)``。"""
    converter = alphabet.get_batch_converter(truncation_seq_length=truncation_seq_length)
    return converter(list(records))


def non_special_mask(tokens: torch.Tensor, alphabet) -> torch.Tensor:
    """标记真实残基位置（排除 BOS/EOS/PAD）。"""
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for idx in filter(lambda x: x is not None, [alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx]):
        mask &= tokens != idx
    return mask


def apply_mlm_mask(
    tokens: torch.Tensor,
    alphabet,
    mask_frac: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 ESM 预训练的方式随机遮挡残基，返回 ``(masked_tokens, mask_positions)``。

    ``mask_positions`` 为布尔张量，标记被遮挡（需要预测）的位置。
    """
    eligible = non_special_mask(tokens, alphabet)
    rand = torch.rand(tokens.shape, generator=generator)
    mask_positions = eligible & (rand < mask_frac)
    # 保证每条序列至少遮挡一个位置（长度 > 1 时）
    empty_rows = mask_positions.sum(dim=1) == 0
    if empty_rows.any():
        for row in torch.nonzero(empty_rows, as_tuple=False).flatten().tolist():
            valid = torch.nonzero(eligible[row], as_tuple=False).flatten()
            if valid.numel() > 0:
                mask_positions[row, valid[torch.randint(valid.numel(), (1,), generator=generator).item()]] = True
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = alphabet.mask_idx
    return masked_tokens, mask_positions


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
@torch.no_grad()
def masked_lm_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask_positions: torch.Tensor,
    logits_ref: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """在遮挡位置计算 top-1/top-5 准确率、平均负对数似然与 masked 困惑度。

    仅统计属于标准氨基酸词表的 logits（ESM-2 的词表本身就是 33 类，
    其中 20 种标准氨基酸 + 特殊 token），因此这里直接在整个词表上取 argmax。
    """
    sel_logits = logits[mask_positions]           # (N, vocab)
    sel_targets = targets[mask_positions]         # (N,)
    n = int(sel_targets.numel())
    if n == 0:
        return {"n_masked": 0}

    log_probs = F.log_softmax(sel_logits.float(), dim=-1)
    nll = -log_probs.gather(1, sel_targets.view(-1, 1)).squeeze(1)
    top1 = (sel_logits.argmax(dim=-1) == sel_targets).float().mean().item()
    k5 = min(5, sel_logits.shape[-1])
    top5 = (
        sel_logits.topk(k5, dim=-1).indices == sel_targets.view(-1, 1)
    ).any(dim=-1).float().mean().item()

    out = {
        "n_masked": n,
        "top1_acc": float(top1),
        "top5_acc": float(top5),
        "mean_nll": float(nll.mean().item()),
        "masked_perplexity": float(torch.exp(nll.mean()).item()),
    }

    if logits_ref is not None:
        ref_log_probs = F.log_softmax(logits_ref[mask_positions].float(), dim=-1)
        out["kl_vs_reference"] = float(
            F.kl_div(log_probs, ref_log_probs.exp(), reduction="batchmean", log_target=False).item()
        )
        out["top1_agreement"] = float(
            (sel_logits.argmax(dim=-1) == logits_ref[mask_positions].argmax(dim=-1)).float().mean().item()
        )
    return out


@torch.no_grad()
def distribution_agreement_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    positions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """比较两个模型在给定位置上的输出分布。

    - ``kl_a_b``：``KL(p_a || p_b)``，衡量 b 相对 a 的信息损失；
    - ``kl_b_a``：``KL(p_b || p_a)``；
    - ``js_divergence``：对称化的 JS 散度；
    - ``top1_agreement`` / ``top5_agreement``：贪心与 top-5 预测一致率；
    - 若给定 ``targets``，还给出两者对真实 token 的 log 概率的 Pearson 相关系数。
    """
    a = logits_a[positions].float()
    b = logits_b[positions].float()
    if a.numel() == 0:
        return {"n_positions": 0}

    log_p = F.log_softmax(a, dim=-1)
    log_q = F.log_softmax(b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-12).log()

    kl_ab = F.kl_div(log_q, p, reduction="batchmean").item()
    kl_ba = F.kl_div(log_p, q, reduction="batchmean").item()
    js = 0.5 * (
        F.kl_div(log_m, p, reduction="batchmean").item()
        + F.kl_div(log_m, q, reduction="batchmean").item()
    )

    k5 = min(5, a.shape[-1])
    top1_a = a.argmax(dim=-1)
    top1_b = b.argmax(dim=-1)
    top5_a = a.topk(k5, dim=-1).indices
    top5_b = b.topk(k5, dim=-1).indices

    out = {
        "n_positions": int(a.shape[0]),
        "kl_a_b": float(kl_ab),
        "kl_b_a": float(kl_ba),
        "js_divergence": float(js),
        "top1_agreement": float((top1_a == top1_b).float().mean().item()),
        "top5_agreement": float(
            (top5_b.unsqueeze(1) == top5_a.unsqueeze(2)).any(dim=2).float().mean().item()
        ),
    }
    if targets is not None:
        t = targets[positions]
        lp_a = log_p.gather(1, t.view(-1, 1)).squeeze(1)
        lp_b = log_q.gather(1, t.view(-1, 1)).squeeze(1)
        out["pearson_true_token_logprob"] = float(
            torch.corrcoef(torch.stack([lp_a, lp_b]))[0, 1].item()
        )
    return out


@torch.no_grad()
def timing_forward(model, tokens: torch.Tensor, n_repeats: int = 1, warmup: int = 1) -> float:
    """返回单次前向的平均耗时（秒）。"""
    for _ in range(warmup):
        model(tokens, repr_layers=[], return_contacts=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_repeats):
        model(tokens, repr_layers=[], return_contacts=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / max(n_repeats, 1)


# --------------------------------------------------------------------------- #
# 模型构建 / 装载
# --------------------------------------------------------------------------- #
def attach_factors(model: nn.Module, factors: Dict[str, Dict]) -> List[str]:
    """把检查点里的 A/S/B/offset/bias 装载到已替换好的 NMF 层中。返回层名列表。"""
    names = []
    for name, payload in factors.items():
        layer = get_submodule(model, name)
        if not isinstance(layer, ThreeFactorNMFLinear):
            raise TypeError(f"{name} 不是 ThreeFactorNMFLinear，无法装载分解参数。")
        with torch.no_grad():
            layer.A.copy_(payload["A"].float())
            layer.S.copy_(payload["S"].float())
            layer.B.copy_(payload["B"].float())
            layer.set_offset(payload.get("offset", 0.0))
            if layer.bias is not None and payload.get("bias") is not None:
                layer.bias.copy_(payload["bias"].float())
        names.append(name)
    return names


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def set_trainable_nmf_params(
    model: nn.Module,
    layer_names: Sequence[str],
    train_offset: bool = True,
    train_bias: bool = False,
) -> List[nn.Parameter]:
    """冻结其他所有参数，只把指定 NMF 层的 A/S/B（+offset/bias）设为可训练。"""
    freeze_all(model)
    params: List[nn.Parameter] = []
    for name in layer_names:
        layer = get_submodule(model, name)
        if not isinstance(layer, ThreeFactorNMFLinear):
            raise TypeError(f"{name} 不是 ThreeFactorNMFLinear。")
        for p in layer.nmf_parameters(include_offset=train_offset):
            p.requires_grad_(True)
            params.append(p)
        if train_bias and layer.bias is not None:
            layer.bias.requires_grad_(True)
            params.append(layer.bias)
    return params


def build_factorized_model(
    model_name: str = "esm2_t6_8M_UR50D",
    layer_spec: str = "layers.5.fc1",
    rank: Optional[int] = None,
    rank_ratio: Optional[float] = None,
    factor_ckpt: Optional[str] = None,
    device: str = "cpu",
    hub_dir: Optional[str] = None,
    solver: str = "hals",
    iters: int = 500,
    shift: str = "min",
    init: str = "nndsvd",
    lr: float = 2e-2,
    grad_clip: float = 1.0,
    n_inner: int = 20,
    tol: float = 1e-9,
    seed: int = 0,
    verbose: bool = True,
    only_transformer_blocks: bool = True,
    trainable: bool = False,
    train_offset: bool = True,
    train_bias: bool = False,
):
    """构建 ``(原模型, 分解模型, alphabet, 层名列表, 分解报告)``。

    - 原模型始终处于 ``eval()`` 且 ``requires_grad=False``，作为蒸馏/对照的参考；
    - 分解模型是原模型的深拷贝，其中指定线性层被替换为三因子非负分解层；
    - ``factor_ckpt`` 为 ``None`` 时现场做 NMF，否则从检查点装载已算好的因子；
    - ``trainable=True`` 时只放开 NMF 层参数（及可选的 offset/bias）。
    """
    orig_model, alphabet = load_esm2(model_name, hub_dir=hub_dir)
    orig_model = orig_model.eval().to(device)
    freeze_all(orig_model)

    layer_names = resolve_layer_names(orig_model, layer_spec, only_transformer_blocks)
    fact_model = copy.deepcopy(orig_model)

    reports: Dict[str, Dict] = {}
    if factor_ckpt:
        ckpt = load_checkpoint(factor_ckpt)
        factors = ckpt["layers"]
        ranks = {name: int(payload["A"].shape[1]) for name, payload in factors.items()}
        missing = [n for n in ranks if n not in layer_names]
        if missing:
            raise ValueError(
                f"检查点中的层 {missing} 不在 --layers 指定的范围 {layer_names} 内。"
            )
        swap_in_empty_nmf_modules(fact_model, ranks)
        attach_factors(fact_model, factors)
        reports = ckpt.get("factorization", {})
        if verbose:
            for name in ranks:
                info = reports.get(name, {})
                print(
                    f"  装载 {name}: rank={ranks[name]}  "
                    f"rel_fro={info.get('relative_frobenius_error', float('nan')):.6f}"
                )
    else:
        ranks: Dict[str, int] = {}
        for name in layer_names:
            linear = get_submodule(orig_model, name)
            full_rank = min(linear.in_features, linear.out_features)
            if rank is not None:
                ranks[name] = int(rank)
            elif rank_ratio is not None:
                ranks[name] = max(1, int(round(full_rank * rank_ratio)))
            else:
                ranks[name] = full_rank
        reports = replace_linears_with_nmf(
            fact_model,
            layer_names,
            rank=ranks,
            solver=solver,
            n_iter=iters,
            shift=shift,
            init=init,
            lr=lr,
            grad_clip=grad_clip,
            n_inner=n_inner,
            tol=tol,
            seed=seed,
            verbose=verbose,
        )

    fact_model = fact_model.to(device)
    if trainable:
        set_trainable_nmf_params(
            fact_model, layer_names, train_offset=train_offset, train_bias=train_bias
        )
    else:
        freeze_all(fact_model)
    return orig_model, fact_model, alphabet, layer_names, reports

