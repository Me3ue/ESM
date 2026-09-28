# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""``nmf`` 包的单元测试：三因子非负分解的正确性、前向等价性与检查点往返。

这些测试只使用随机张量和极小的 ``nn.Linear``，不会下载预训练权重，可以在 CPU 上秒级跑完：

    pytest tests/test_nmf.py -v
"""

import copy
import json
import os

import pytest
import torch
import torch.nn as nn

from nmf.esm_common import (
    attach_factors,
    load_checkpoint,
    resolve_layer_names,
    set_trainable_nmf_params,
)
from nmf.nmf_layers import (
    ThreeFactorNMFLinear,
    factorize_matrix,
    init_nndsvd,
    init_random_nonnegative,
    make_shift,
    relative_frobenius_error,
    replace_linears_with_nmf,
    swap_in_empty_nmf_modules,
)


def _random_weight(out_features=48, in_features=32, seed=0):
    """生成带正负元素、量级与真实线性层接近的随机权重。"""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(out_features, in_features, generator=generator) * 0.25


def _tiny_model():
    """构造一个含 layers.N.fc1 / layers.N.self_attn.out_proj 命名结构的小模型。"""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.out_proj = nn.Linear(32, 32)
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 32)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(self.self_attn.out_proj(x))))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return Tiny()


# --------------------------------------------------------------------------- #
# 基本面：形状、非负、方阵
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", ["mu", "pgd"])
def test_factorize_shapes_and_nonneg(solver):
    W = _random_weight()
    rank = 20
    A, S, B, offset, info = factorize_matrix(W, rank=rank, solver=solver, n_iter=200)

    assert A.shape == (W.shape[0], rank)
    assert S.shape == (rank, rank)          # 中间矩阵必须是方阵
    assert B.shape == (rank, W.shape[1])
    assert offset.shape == (W.shape[0],)

    assert (A >= 0).all() and (S >= 0).all() and (B >= 0).all(), "A/S/B 必须非负"
    assert info["rank"] == rank
    assert info["n_params_factored"] == A.numel() + S.numel() + B.numel()
    assert 0.0 <= info["relative_frobenius_error"] < 1.0


def test_rank_defaults_to_min_dimension():
    W = _random_weight(out_features=48, in_features=32)
    A, S, B, _, info = factorize_matrix(W, n_iter=100)
    assert info["rank"] == 32
    assert A.shape == (48, 32) and S.shape == (32, 32) and B.shape == (32, 32)


def test_iterations_reduce_error():
    W = _random_weight()
    _, _, _, _, info_few = factorize_matrix(W, rank=24, solver="mu", n_iter=20)
    _, _, _, _, info_many = factorize_matrix(W, rank=24, solver="mu", n_iter=600)
    assert info_many["relative_frobenius_error"] < info_few["relative_frobenius_error"]


def test_lower_rank_is_worse():
    """降秩应当带来更大的重构误差（NMF 是有损压缩）。"""
    W = _random_weight(out_features=64, in_features=64)
    _, _, _, _, full = factorize_matrix(W, rank=64, solver="pgd", n_iter=800)
    _, _, _, _, half = factorize_matrix(W, rank=32, solver="pgd", n_iter=800)
    assert half["relative_frobenius_error"] > full["relative_frobenius_error"]


# --------------------------------------------------------------------------- #
# 平移（offset）
# --------------------------------------------------------------------------- #
def test_shift_modes_produce_nonnegative_target():
    W = _random_weight()
    for mode in ("min", "row"):
        target, offset = make_shift(W, mode)
        assert target.min() >= -1e-6, f"shift={mode} 的目标矩阵必须非负"
        assert offset.shape == (W.shape[0],)
        if mode == "min":
            assert torch.allclose(offset, torch.full_like(offset, float(W.min())))

    row_target, row_offset = make_shift(W, "row")
    min_target, min_offset = make_shift(W, "min")
    # 逐行平移的目标范数不会大于整体平移
    assert row_target.norm() <= min_target.norm() + 1e-6
    assert not torch.allclose(row_offset, min_offset)


def test_shift_none_rejects_signed_input():
    W = _random_weight()
    with pytest.raises(ValueError):
        make_shift(W, "none")


def test_shift_zero_clips_negatives():
    W = _random_weight()
    target, offset = make_shift(W, "zero")
    assert target.min() >= 0
    assert torch.all(offset == 0)


# --------------------------------------------------------------------------- #
# 前向等价性
# --------------------------------------------------------------------------- #
def test_forward_matches_materialized_weight():
    linear = nn.Linear(32, 48)
    nmf_layer, _ = ThreeFactorNMFLinear.from_linear(linear, rank=16, n_iter=200)
    nmf_layer.eval()

    x = torch.randn(3, 7, 32)
    with torch.no_grad():
        y_seq = nmf_layer(x)
        y_mat = nmf_layer.forward_materialized(x)
    assert torch.allclose(y_seq, y_mat, atol=1e-5), "三矩阵顺序相乘必须等价于乘积矩阵"

    # effective_weight 应等于 A @ S @ B + offset[:, None]
    expected = nmf_layer.A @ nmf_layer.S @ nmf_layer.B + nmf_layer.weight_offset.unsqueeze(1)
    assert torch.allclose(nmf_layer.effective_weight(), expected)


def test_bias_is_preserved():
    linear = nn.Linear(32, 48)
    with torch.no_grad():
        linear.bias.copy_(torch.randn(48))
    nmf_layer, _ = ThreeFactorNMFLinear.from_linear(linear, rank=16, n_iter=100)
    x = torch.zeros(1, 4, 32)
    with torch.no_grad():
        assert torch.allclose(nmf_layer(x), linear.bias.expand(1, 4, 48), atol=1e-5)


def test_module_without_bias():
    linear = nn.Linear(32, 48, bias=False)
    nmf_layer, _ = ThreeFactorNMFLinear.from_linear(linear, rank=16, n_iter=100)
    assert nmf_layer.bias is None
    with torch.no_grad():
        nmf_layer(torch.randn(2, 5, 32))


# --------------------------------------------------------------------------- #
# 模型改造与检查点往返
# --------------------------------------------------------------------------- #
def test_resolve_layer_names():
    model = _tiny_model()
    assert resolve_layer_names(model, "layers.1.fc1") == ["layers.1.fc1"]
    assert resolve_layer_names(model, "layers.*.fc1") == [
        "layers.0.fc1",
        "layers.1.fc1",
        "layers.2.fc1",
    ]
    assert resolve_layer_names(model, "first") == ["layers.0.self_attn.out_proj", "layers.0.fc1", "layers.0.fc2"]
    assert resolve_layer_names(model, "last.fc1") == ["layers.2.fc1"]
    assert len(resolve_layer_names(model, "all")) == 9
    with pytest.raises(ValueError):
        resolve_layer_names(model, "layers.9.fc1")


def test_replace_and_trainable_params():
    model = _tiny_model()
    reports = replace_linears_with_nmf(
        model, ["layers.1.fc1"], rank=16, n_iter=100, verbose=False
    )
    assert "layers.1.fc1" in reports
    layer = model.layers[1].fc1
    assert isinstance(layer, ThreeFactorNMFLinear)
    assert layer.S.shape == (16, 16)
    assert model.layers[1].fc1.out_features == 64

    # 只有 NMF 参数可训练
    params = set_trainable_nmf_params(model, ["layers.1.fc1"])
    assert len(params) == 4  # A、S、B、offset
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {
        "layers.1.fc1.A",
        "layers.1.fc1.S",
        "layers.1.fc1.B",
        "layers.1.fc1.weight_offset",
    }

    # 前向可用
    with torch.no_grad():
        assert model(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_project_to_nonnegative():
    model = _tiny_model()
    replace_linears_with_nmf(model, ["layers.1.fc1"], rank=16, n_iter=100, verbose=False)
    layer = model.layers[1].fc1
    with torch.no_grad():
        layer.A.sub_(1.0)
        layer.S.sub_(1.0)
        layer.B.sub_(1.0)
    layer.project_to_nonnegative_()
    assert (layer.A >= 0).all() and (layer.S >= 0).all() and (layer.B >= 0).all()


def test_checkpoint_roundtrip(tmp_path):
    """分解 → 保存 → 用 swap_in_empty_nmf_modules + attach_factors 还原，前向应完全一致。"""
    model = _tiny_model()
    reports = replace_linears_with_nmf(
        model, ["layers.1.fc1", "layers.1.self_attn.out_proj"], rank=16, n_iter=100, verbose=False
    )
    factors = {}
    for name in reports:
        module = dict(model.named_modules())[name]
        factors[name] = {
            "A": module.A.detach().clone(),
            "S": module.S.detach().clone(),
            "B": module.B.detach().clone(),
            "offset": module.weight_offset.detach().clone(),
            "bias": None if module.bias is None else module.bias.detach().clone(),
        }
    ckpt_path = tmp_path / "factors.pt"
    torch.save({"stage": "factorize", "layers": factors, "factorization": {}}, ckpt_path)

    target = copy.deepcopy(model)
    ranks = {name: payload["A"].shape[1] for name, payload in factors.items()}
    # 先把新模型里的对应层换回 nn.Linear，再走一遍装载流程
    for name in ranks:
        parent = target
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Linear(32, 32 if "out_proj" in name else 64))

    swap_in_empty_nmf_modules(target, ranks)
    attach_factors(target, factors)

    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), target(x), atol=1e-6)


def test_saved_checkpoint_loads_with_load_checkpoint(tmp_path):
    """检查点中不含自定义对象，torch.load 的 weights_only 默认值也应当能加载。"""
    from nmf.esm_common import load_checkpoint

    W = _random_weight(out_features=16, in_features=16)
    A, S, B, offset, info = factorize_matrix(W, rank=8, n_iter=50)
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "stage": "factorize",
            "model_name": "dummy",
            "layers": {
                "layers.0.fc1": {
                    "A": A, "S": S, "B": B, "offset": offset, "bias": None,
                    "rank": 8, "in_features": 16, "out_features": 16,
                }
            },
            "factorization": {},
            "torch_version": str(torch.__version__),
        },
        path,
    )
    loaded = load_checkpoint(str(path))
    assert loaded["stage"] == "factorize"
    assert torch.allclose(loaded["layers"]["layers.0.fc1"]["A"], A)


# --------------------------------------------------------------------------- #
# 指标与工具
# --------------------------------------------------------------------------- #
def test_relative_frobenius_error():
    W = torch.ones(4, 4)
    assert relative_frobenius_error(W, W) == 0.0
    assert relative_frobenius_error(W, torch.zeros(4, 4)) == pytest.approx(1.0)


def test_factor_stats_reports_nonnegative():
    linear = nn.Linear(32, 48)
    nmf_layer, _ = ThreeFactorNMFLinear.from_linear(linear, rank=16, n_iter=100)
    stats = nmf_layer.factor_stats()
    assert stats["rank"] == 16
    assert stats["nonnegative"] is True
    assert set(stats["sparsity"]) == {"A", "S", "B"}


# --------------------------------------------------------------------------- #
# HALS 求解器
# --------------------------------------------------------------------------- #
def test_hals_shapes_and_nonneg():
    W = _random_weight(out_features=40, in_features=24)
    A, S, B, offset, info = factorize_matrix(W, rank=12, solver="hals", n_iter=100)
    assert info["solver"] == "hals"
    assert A.shape == (40, 12) and S.shape == (12, 12) and B.shape == (12, 24)
    assert (A >= 0).all() and (S >= 0).all() and (B >= 0).all()


def test_hals_reaches_near_exact_at_full_rank():
    """满阶时存在精确解 A = W-offset, S = I, B = I，HALS 应逼近它。"""
    W = _random_weight(out_features=48, in_features=32, seed=3)
    _, _, _, _, info = factorize_matrix(W, solver="hals", n_iter=300, shift="min")
    assert info["rank"] == 32
    assert info["relative_frobenius_error"] < 0.1, info["relative_frobenius_error"]
    assert info["cosine_similarity"] > 0.99


def test_hals_beats_multiplicative_updates():
    """相同迭代次数下 HALS 应显著优于乘性更新（这是选它做默认求解器的依据）。"""
    W = _random_weight(out_features=40, in_features=40, seed=5)
    _, _, _, _, hals = factorize_matrix(W, rank=40, solver="hals", n_iter=200)
    _, _, _, _, mu = factorize_matrix(W, rank=40, solver="mu", n_iter=200)
    assert hals["relative_frobenius_error"] < mu["relative_frobenius_error"]


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def test_quality_control_filters_nonstandard_length_and_duplicates():
    from nmf.prepare_data import quality_control

    records = [
        ("ok1", "ACDEFGHIKLMNPQRSTVWY" * 3),      # 合格（60 aa）
        ("bad_aa", "ACDEFGHIKLX" * 6),             # X 非常规
        ("short", "ACDEFGHIKL"),                   # 过短
        ("long", "ACDEFGHIKLMNPQRSTVWY" * 10),     # 过长
        ("dup", "ACDEFGHIKLMNPQRSTVWY" * 3),       # 与 ok1 重复
    ]
    kept, stats = quality_control(records, min_len=50, max_len=100)
    labels = [lab for lab, _ in kept]
    assert labels == ["ok1"]
    assert stats["nonstandard"] == 1 and stats["too_short"] == 1
    assert stats["too_long"] == 1 and stats["duplicate"] == 1


def test_kmer_hash_is_stable():
    from nmf.prepare_data import hash_kmer, kmers

    assert hash_kmer("ACDEFG") == hash_kmer("ACDEFG")
    assert hash_kmer("ACDEFG") != hash_kmer("ACDEFH")
    assert len(kmers("ACDEFGHIKL", 3)) == 8


def test_decontaminate_removes_high_similarity_sequences():
    from nmf.prepare_data import decontaminate

    guard = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    train = [
        ("clean", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQMKTVRQERLKSIV"),
        ("contaminated", guard[:38] + "AA"),   # 与 guard 高度重合
        ("contaminated_exact", guard),
    ]
    kept, stats = decontaminate(train, [("g", guard)], threshold=0.6, k=6, verbose=False)
    labels = [lab for lab, _ in kept]
    assert "contaminated" not in labels and "contaminated_exact" not in labels
    assert "clean" in labels
    assert stats["removed"] == 2


# --------------------------------------------------------------------------- #
# 训练侧工具
# --------------------------------------------------------------------------- #
def test_make_batches_respects_token_budget():
    from nmf.train_nmf import make_batches

    records = [(f"s{i}", "A" * length) for i, length in enumerate([50, 60, 70, 80, 90, 100])]
    batches = make_batches(records, batch_size=100, max_tokens_per_batch=200, sort_by_length=True)
    assert sum(len(b) for b in batches) == len(records), "不能丢样本"
    for batch in batches:
        assert sum(len(s) + 2 for _, s in batch) <= 200


def test_make_batches_falls_back_to_fixed_size():
    from nmf.train_nmf import make_batches

    records = [(f"s{i}", "A" * 50) for i in range(7)]
    batches = make_batches(records, batch_size=3, max_tokens_per_batch=0)
    assert [len(b) for b in batches] == [3, 3, 1]


def test_snapshot_and_restore_factors():
    from nmf.train_nmf import restore_factors, snapshot_factors

    model = _tiny_model()
    replace_linears_with_nmf(model, ["layers.1.fc1"], rank=16, n_iter=50, verbose=False)
    layer = model.layers[1].fc1
    snap = snapshot_factors(model, ["layers.1.fc1"])
    with torch.no_grad():
        layer.A.add_(0.5)
        layer.weight_offset.add_(0.1)
    restore_factors(model, snap)
    assert torch.allclose(layer.A, snap["layers.1.fc1"][0])
    assert torch.allclose(layer.weight_offset, snap["layers.1.fc1"][3])


# --------------------------------------------------------------------------- #
# 论文级评测工具
# --------------------------------------------------------------------------- #
def test_linear_cka_properties():
    from nmf.benchmark import linear_cca_style_cka

    torch.manual_seed(0)
    x = torch.randn(64, 12)
    assert linear_cca_style_cka(x, x) == pytest.approx(1.0, abs=1e-5)
    # 对正交变换不变（CKA 的核心性质）
    q, _ = torch.linalg.qr(torch.randn(12, 12))
    assert linear_cca_style_cka(x, x @ q) == pytest.approx(1.0, abs=1e-4)
    # 对无关表征显著小于 1
    y = torch.randn(64, 12)
    assert linear_cca_style_cka(x, y) < 0.5


def test_linear_macs_ratio():
    from nmf.benchmark import count_linear_names, linear_macs

    model = _tiny_model()
    names = count_linear_names(model)
    base = linear_macs(model, 100, names)
    replace_linears_with_nmf(model, ["layers.1.fc1"], rank=16, n_iter=50, verbose=False)
    after = linear_macs(model, 100, count_linear_names(model))
    # fc1: 32->64 变成 32*16 + 16*16 + 16*64 = 1792，比原来的 2048 小
    assert after < base


def test_parse_checkpoints():
    from nmf.benchmark import parse_checkpoints

    parsed = parse_checkpoints(["未训练=a.pt", "b.pt"])
    assert parsed == [("未训练", "a.pt"), ("b", "b.pt")]


# --------------------------------------------------------------------------- #
# 分组学习率（三因子尺度不平衡导致的训练稳定性修复）
# --------------------------------------------------------------------------- #
def _biased_scale_model(seed=0):
    """构造一个 A/S/B 尺度刻意不平衡的分解模型（模拟真实 NMF 解）。"""
    torch.manual_seed(seed)
    model = _tiny_model()
    replace_linears_with_nmf(model, ["layers.1.fc1"], rank=16, n_iter=50, verbose=False)
    layer = model.layers[1].fc1
    with torch.no_grad():
        # A 很小、B 很大：乘积仍与原权重同量级，但两因子尺度相差 3 个数量级
        layer.A.mul_(1e-3)
        layer.B.mul_(1e3)
    return model, layer


def test_perturbation_gains_match_definitions():
    """增益值应与定义式一致，且四个分组都为正。"""
    from nmf.train_nmf import factor_perturbation_gains

    model, layer = _biased_scale_model()
    gains = factor_perturbation_gains(model, ["layers.1.fc1"])
    assert set(gains) == {"A", "S", "B", "offset"}
    assert all(v > 0 for v in gains.values())

    A, S, B = layer.A.detach(), layer.S.detach(), layer.B.detach()
    expect = {
        "A": float((S @ B).sum(dim=0).max()),
        "S": float(A.sum(dim=1).max()) * float(B.sum(dim=0).max()),
        "B": float((A @ S).sum(dim=1).max()),
        "offset": float(layer.in_features) ** 0.5,
    }
    for key, value in expect.items():
        assert gains[key] == pytest.approx(value, rel=1e-6), key


def test_group_lr_times_gain_is_constant():
    """lr_scale=auto 时，各因子「单步等效权重移动量」lr × gain 应一致。"""
    from nmf.train_nmf import build_parameter_groups, factor_perturbation_gains

    model, _ = _biased_scale_model()
    gains = factor_perturbation_gains(model, ["layers.1.fc1"])
    groups, lrs = build_parameter_groups(
        model, ["layers.1.fc1"], lr=1e-3, lr_scale="auto", verbose=False
    )
    assert set(lrs) == {"A", "S", "B", "offset"}
    products = [lrs[k] * gains[k] for k in lrs]
    assert max(products) == pytest.approx(min(products), rel=1e-6)
    assert max(products) == pytest.approx(1e-3, rel=1e-6)
    # 增益最大的因子必须拿到最小的学习率
    most_sensitive = max(gains, key=lambda k: gains[k])
    assert lrs[most_sensitive] == min(lrs.values())
    assert sum(len(g["params"]) for g in groups) == 4  # 每层 A/S/B/offset


def test_empirical_sensitivity_ordering_matches_gains():
    """核心性质：经验灵敏度的大小顺序应与解析增益的顺序一致（验证增益公式的正确性）。"""
    from nmf.train_nmf import factor_perturbation_gains

    model, layer = _biased_scale_model()
    gains = factor_perturbation_gains(model, ["layers.1.fc1"])
    base = layer.effective_weight().detach().clone()
    delta = 1e-3
    measured = {}
    for name in ("A", "S", "B"):
        backup = getattr(layer, name).detach().clone()
        with torch.no_grad():
            getattr(layer, name).add_(delta)
            measured[name] = float(
                (layer.effective_weight().detach() - base).norm() / base.norm()
            )
            getattr(layer, name).copy_(backup)
    order_measured = sorted(measured, key=lambda k: -measured[k])
    order_gains = sorted(("A", "S", "B"), key=lambda k: -gains[k])
    assert order_measured == order_gains, (measured, gains)
    # 且灵敏度最高的因子确实显著高于最低的
    assert measured[order_measured[0]] > 10 * max(measured[order_measured[-1]], 1e-12)


def test_group_lr_scale_none_gives_uniform_lr():
    from nmf.train_nmf import build_parameter_groups

    model, _ = _biased_scale_model()
    _, lrs = build_parameter_groups(
        model, ["layers.1.fc1"], lr=1e-3, lr_scale="none", verbose=False
    )
    assert all(v == 1e-3 for v in lrs.values())




# --------------------------------------------------------------------------- #
# 论文级评测的增量落盘 / 复用（长评测被中断时不丢结果）
# --------------------------------------------------------------------------- #
def _bench_args(tmp_path, **overrides):
    from nmf.benchmark import parse_args

    args = parse_args([
        "--checkpoint", "dummy.pt",
        "--fasta", str(tmp_path / "toy.fasta"),
        "--out-dir", str(tmp_path / "out"),
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_fingerprint_covers_only_protocol_fields(tmp_path):
    from nmf.benchmark import fingerprint

    base = _bench_args(tmp_path)
    cosmetic = _bench_args(tmp_path, out_dir=str(tmp_path / "other"), plot=True, quiet=True)
    assert fingerprint(base) == fingerprint(cosmetic)
    # 评测口径变了就必须重算
    assert fingerprint(base) != fingerprint(_bench_args(tmp_path, n_eval=7))
    assert fingerprint(base) != fingerprint(_bench_args(tmp_path, mask_rounds=9))
    assert fingerprint(base) != fingerprint(_bench_args(tmp_path, seed=1))


def test_variant_save_load_and_reuse_guard(tmp_path):
    from nmf.benchmark import load_variant, save_variant, variant_json_path

    out_dir = str(tmp_path / "bench")
    fp = {"n_eval": 8}
    entry = {"checkpoint": "a.pt", "n_params_total": 10, "layer_stats": {"per_layer": []}}
    path = save_variant(out_dir, "满阶 训练后", fp, entry)

    assert os.path.exists(path)
    assert os.path.basename(path) == "满阶_训练后.json", "文件名必须做安全化处理"
    assert variant_json_path(out_dir, "满阶 训练后") == path

    assert load_variant(out_dir, "满阶 训练后", fp, "a.pt")["n_params_total"] == 10
    assert load_variant(out_dir, "满阶 训练后", fp, "b.pt") is None, "检查点变了不能复用"
    assert load_variant(out_dir, "满阶 训练后", {"n_eval": 9}, "a.pt") is None, "口径变了不能复用"
    assert load_variant(out_dir, "不存在", fp, "a.pt") is None


def test_select_consistent_variants_skips_other_protocol(tmp_path):
    from nmf.benchmark import dump_json, select_consistent_variants, variant_json_path

    out_dir = str(tmp_path / "bench")
    base_fp = {"n_eval": 8}
    for name, fp in (("同口径", {"n_eval": 8}), ("异口径", {"n_eval": 99})):
        dump_json(
            variant_json_path(out_dir, name),
            {"name": name, "fingerprint": fp, "checkpoint": name + ".pt", "entry": {"checkpoint": name + ".pt"}},
        )
    kept = select_consistent_variants(out_dir, base_fp, quiet=True)
    assert list(kept) == ["同口径"], "口径不一致的变体不能混进同一张表"


def test_select_consistent_variants_requires_uniform_without_base(tmp_path):
    from nmf.benchmark import dump_json, select_consistent_variants, variant_json_path

    out_dir = str(tmp_path / "bench")
    for name, fp in (("a", {"n_eval": 8}), ("b", {"n_eval": 99})):
        dump_json(variant_json_path(out_dir, name), {"name": name, "fingerprint": fp, "entry": {}})
    with pytest.raises(SystemExit):
        select_consistent_variants(out_dir, None, quiet=True)
    # 缺 variants 目录时返回空而不是报错
    assert select_consistent_variants(str(tmp_path / "nothing"), {"n_eval": 8}, quiet=True) == {}


def test_variant_entry_is_renderable_without_model(tmp_path):
    """--from-dir 依赖"变体结果自带渲染所需字段"，缺字段应在渲染阶段就暴露。"""
    from nmf.benchmark import dump_json, render_report, select_consistent_variants, variant_json_path

    out_dir = str(tmp_path / "bench")
    entry = {
        "checkpoint": "a.pt",
        "layer_stats": {
            "mean_relative_frobenius_error": 0.04, "max_relative_frobenius_error": 0.1,
            "mean_cosine_similarity": 0.99, "n_layers": 1, "n_params_original": 10,
            "n_params_factored": 12, "per_layer": [
                {"layer": "layers.0.fc1", "shape": [2, 5], "rank": 2,
                 "relative_frobenius_error": 0.04, "cosine_similarity": 0.99,
                 "n_params_original": 10, "n_params_factored": 12},
            ],
        },
        "masked_lm": {
            "orig_top1_acc": 0.2, "orig_top1_acc_std": 0.0, "orig_top5_acc": 0.5,
            "orig_top5_acc_std": 0.0, "orig_masked_perplexity": 10.0,
            "orig_masked_perplexity_std": 0.0, "nmf_top1_acc": 0.19,
            "nmf_top1_acc_std": 0.0, "nmf_top5_acc": 0.49, "nmf_top5_acc_std": 0.0,
            "nmf_masked_perplexity": 10.2, "nmf_masked_perplexity_std": 0.0,
            "top1_agreement": 0.98, "kl_nmf_ref": 0.01,
        },
        "distribution": {
            "kl_b_a": 0.01, "js_divergence": 0.003, "top1_agreement": 0.99,
            "top5_agreement": 0.9, "pearson_true_token_logprob": 0.95,
        },
        "representation": {"cka": 0.97, "mean_cosine": 0.98},
        "n_params_total": 22, "linear_mac_ratio": 1.01, "pseudo_perplexity": 10.1,
    }
    dump_json(variant_json_path(out_dir, "v"), {"name": "v", "fingerprint": {"n_eval": 1}, "entry": entry})
    base = {
        "model_name": "tiny", "fasta": "toy.fasta", "n_eval": 1, "n_residues": 5,
        "mask_frac": 0.15, "mask_rounds": 1, "ppl_records": 1, "ppl_max_len": 16,
        "device": "cpu", "torch_version": "x", "seed": 0, "orig_params": 20,
        "orig_pseudo_perplexity": 10.0, "orig_linear_macs": 100.0,
        "fingerprint": {"n_eval": 1},
    }
    dump_json(os.path.join(out_dir, "base.json"), base)
    report = dict(base)
    report["variants"] = select_consistent_variants(out_dir, base["fingerprint"], quiet=True)
    render_report(report, out_dir, plot=False, quiet=True)

    with open(os.path.join(out_dir, "benchmark.md")) as handle:
        text = handle.read()
    assert "论文级评测报告" in text and "0.97000" in text and "10.1000" in text
    assert os.path.exists(os.path.join(out_dir, "layer_stats.csv"))
    assert os.path.exists(os.path.join(out_dir, "benchmark.json"))


def test_pseudo_perplexity_rejects_empty_record_list():
    """空记录/全部过短时必须显式报错，而不是静默返回 nan（旧版本会悄悄给 nan）。"""
    from nmf.benchmark import pseudo_perplexity

    class _Model:
        def __call__(self, *a, **k):  # pragma: no cover - 不应被调用
            raise AssertionError("空记录时不应做前向")

    with pytest.raises(ValueError):
        pseudo_perplexity(_Model(), [("a", "A")], alphabet=None, device="cpu", max_len=16, batch=4)


# --------------------------------------------------------------------------- #
# 指标汇总（nmf.summarize，纯后处理）
# --------------------------------------------------------------------------- #
def _fake_factor_report(n_layers=3, rel=0.04, cos=0.99):
    layers = {}
    for i in range(n_layers):
        layers[f"layers.{i}.fc1"] = {
            "shape": [8, 5], "rank": 5,
            "relative_frobenius_error": rel + 0.01 * i,
            "cosine_similarity": cos - 0.001 * i,
            "n_params_original": 40, "n_params_factored": 96,
        }
    return {"model_name": "tiny", "layers": layers, "elapsed_seconds": 12.5}


def _fake_history(steps=4, rolled_back=0):
    history = [
        {"step": i + 1, "epoch": 1, "loss_recon": 0.04 + 0.001 * i,
         "loss_distill": 0.06 - 0.01 * i, "n_masked": 10,
         "lrs": {"A": 5e-8, "S": 3e-9, "B": 2e-5, "offset": 2e-5},
         "lr_halved": 0, "rolled_back": i < rolled_back, "elapsed_s": 2.5 * (i + 1)}
        for i in range(steps)
    ]
    eval_history = [{"step": steps, "orig_top1_acc": 0.22, "nmf_top1_acc": 0.21,
                     "orig_masked_ppl": 12.0, "nmf_masked_ppl": 12.2,
                     "kl_nmf_ref": 0.02, "top1_agreement": 0.84}]
    return {"history": history, "eval_history": eval_history, "final_stats": {}}


def _fake_benchmark():
    masked = {"orig_top1_acc": 0.20, "orig_top1_acc_std": 0.002, "orig_top5_acc": 0.55,
              "orig_top5_acc_std": 0.0, "orig_masked_perplexity": 13.0,
              "orig_masked_perplexity_std": 0.1, "nmf_top1_acc": 0.199,
              "nmf_top1_acc_std": 0.003, "nmf_top5_acc": 0.54, "nmf_top5_acc_std": 0.0,
              "nmf_masked_perplexity": 13.1, "nmf_masked_perplexity_std": 0.1,
              "top1_agreement": 0.99, "kl_nmf_ref": 0.007}
    return {
        "model_name": "tiny", "fasta": "test.fasta", "n_eval": 8, "n_residues": 400,
        "mask_frac": 0.15, "mask_rounds": 5, "seed": 0, "orig_params": 7_512_474,
        "orig_pseudo_perplexity": 12.4,
        "variants": {
            "满阶训练后": {
                "checkpoint": "a.pt", "stage": "train", "masked_lm": masked,
                "distribution": {"kl_b_a": 0.0068, "js_divergence": 0.002,
                                 "top1_agreement": 0.9935, "top5_agreement": 0.9,
                                 "pearson_true_token_logprob": 0.98},
                "representation": {"cka": 0.9732, "mean_cosine": 0.9823},
                "layer_stats": {"n_layers": 38, "mean_relative_frobenius_error": 0.04635,
                                "max_relative_frobenius_error": 0.1422,
                                "mean_cosine_similarity": 0.9985,
                                "min_cosine_similarity": 0.991,
                                "n_params_original": 7_475_320,
                                "n_params_factored": 15_052_922},
                "n_params_total": 15_107_677, "linear_mac_ratio": 2.014,
                "pseudo_perplexity": 12.5438,
            }
        },
    }


def test_summarize_discover_tags(tmp_path):
    from nmf.summarize import discover_tags

    root = str(tmp_path / "out")
    os.makedirs(root)
    for name in ("factor_report_fullrank.json", "factors_fullrank.pt",
                 "history_fullrank.json", "factors_halfrank.pt",
                 "benchmark.json", "SUMMARY.md"):
        open(os.path.join(root, name), "w").close()
    tags = discover_tags(root)
    assert tags == ["fullrank", "halfrank"], tags  # 去重且不误收 benchmark.json / SUMMARY.md


def test_summarize_collects_factorization_and_training(tmp_path):
    from nmf.summarize import collect_factorization, collect_training

    root = str(tmp_path / "out")
    os.makedirs(root)
    with open(os.path.join(root, "factor_report_demo.json"), "w") as handle:
        json.dump(_fake_factor_report(), handle)
    with open(os.path.join(root, "history_demo.json"), "w") as handle:
        json.dump(_fake_history(), handle)

    f = collect_factorization(root, "demo")
    assert f["n_layers"] == 3
    assert f["mean_relative_frobenius_error"] == pytest.approx(0.05)
    assert f["param_ratio"] == pytest.approx(96 / 40)
    assert f["elapsed_seconds"] == pytest.approx(12.5)

    t = collect_training(root, "demo")
    assert t["steps"] == 4 and t["epochs_used"] == [1]
    assert t["recon_rel_fro_change"] == pytest.approx(0.043 / 0.04 - 1.0, rel=1e-6)
    assert t["rolled_back_steps"] == 0 and t["lr_halved_steps"] == 0
    assert collect_factorization(root, "不存在") is None
    assert collect_training(root, "不存在") is None


def test_summarize_renders_all_sections(tmp_path):
    from nmf.summarize import collect_benchmark, render_markdown

    bench_dir = str(tmp_path / "bench")
    os.makedirs(bench_dir)
    with open(os.path.join(bench_dir, "benchmark.json"), "w") as handle:
        json.dump(_fake_benchmark(), handle)

    data = {
        "source": "swissprot", "query": "q", "raw_file": "/x.fasta",
        "quality_control": {"input": 100, "kept": 90, "nonstandard": 1, "too_short": 0,
                            "too_long": 0, "duplicate": 9},
        "decontamination": {"k": 6, "threshold": 0.6, "train_before": 50,
                            "train_after": 49, "removed": 1},
        "splits": {"train": {"sha256": "a" * 64, "length": {"n": 49, "min": 50, "p50": 300,
                                                            "p95": 700, "max": 1000,
                                                            "mean": 340.0,
                                                            "total_residues": 16660}}},
    }
    factorizations = {"demo": {
        "tag": "demo", "n_layers": 3, "ranks": [5],
        "mean_relative_frobenius_error": 0.05, "median_relative_frobenius_error": 0.05,
        "max_relative_frobenius_error": 0.07, "mean_cosine_similarity": 0.998,
        "min_cosine_similarity": 0.996, "n_params_original": 120, "n_params_factored": 288,
        "param_ratio": 2.4, "elapsed_seconds": 12.5,
        "per_layer": {"layers.0.fc1": {"shape": [8, 5], "rank": 5,
                                       "relative_frobenius_error": 0.05,
                                       "cosine_similarity": 0.998,
                                       "n_params_original": 40, "n_params_factored": 96}},
    }}
    trainings = {"demo": {
        "tag": "demo", "steps": 4, "epochs_used": [1], "elapsed_seconds": 10.0,
        "seconds_per_step": 2.5, "rolled_back_steps": 0, "lr_halved_steps": 0,
        "group_lrs_first": None, "group_lrs_last": {"A": 5e-8, "S": 3e-9, "B": 2e-5, "offset": 2e-5},
        "distill_loss_first": 0.06, "distill_loss_last": 0.03,
        "recon_rel_fro_first": 0.04, "recon_rel_fro_last": 0.043,
        "recon_rel_fro_change": 0.075, "n_masked_total": 40,
        "final_eval": _fake_history()["eval_history"][0], "eval_steps": [4],
    }}
    report = collect_benchmark(bench_dir)
    assert report["variants"]["满阶训练后"]["cka"] == pytest.approx(0.9732)

    text = render_markdown(data, factorizations, trainings, report, bench_dir)
    for needle in ("# NMF 实验指标汇总", "## 1. 数据", "## 2. 逐层分解质量",
                   "## 3. 训练过程", "## 4. 模型级能力对比",
                   "aaaaaaaaaaaa",  # 数据切分的 sha256 前 12 位
                   "满阶训练后", "0.05000", "0.97320"):
        assert needle in text, needle
    assert "test.fasta" in text


def test_summarize_render_without_benchmark(tmp_path):
    """没跑评测时也要能出汇总（只含数据/分解/训练），并在缺失处给提示。"""
    from nmf.summarize import render_markdown

    text = render_markdown(None, {}, {}, None, str(tmp_path / "bench"))
    assert "未找到 `stats.json`" in text
    assert "未找到任何分解报告" in text
    assert "未找到任何训练历史" in text
    assert "未找到" in text and "benchmark.json" in text


# --------------------------------------------------------------------------- #
# 设备安全（GPU 支持）
#
# 这几条断言在纯 CPU 的机器上也能通过，但真正的价值是在装了 CUDA 的机器上跑同一套测试：
# 任何把中间张量硬编码成 CPU 的改动都会立刻暴露 —— 那正是 `--device cuda` 曾经直接崩溃的原因。
# --------------------------------------------------------------------------- #
def test_all_intermediate_tensors_follow_input_device():
    W = _random_weight(out_features=12, in_features=8)   # float32
    assert W.dtype == torch.float32

    for mode in ("min", "row", "zero"):
        W_target, offset = make_shift(W, mode)
        assert W_target.device == W.device and offset.device == W.device
        assert W_target.dtype == W.dtype and offset.dtype == W.dtype

    _, offset = make_shift(W.abs(), "none")               # none 要求非负
    assert offset.device == W.device and offset.dtype == W.dtype

    A, S, B = init_nndsvd(W, rank=8)
    for t in (A, S, B):
        assert t.device == W.device and t.dtype == W.dtype

    A, S, B = init_random_nonnegative(
        12, 8, 8, 0.1,
        generator=torch.Generator().manual_seed(0),
        device=W.device, dtype=W.dtype,
    )
    for t in (A, S, B):
        assert t.device == W.device and t.dtype == W.dtype


def test_factorize_matrix_returns_tensors_on_input_device():
    W = _random_weight(out_features=12, in_features=8)
    A, S, B, offset, _ = factorize_matrix(W, rank=6, solver="hals", n_iter=2)
    for t in (A, S, B, offset):
        assert t.device == W.device
    # 满阶 + min 平移时 rank 取 min(in,out)，S 必须是方阵
    A, S, B, offset, info = factorize_matrix(W, solver="hals", n_iter=2)
    assert S.shape == (info["rank"], info["rank"])


def test_apply_mlm_mask_is_device_agnostic_and_reproducible():
    """遮挡位置由 CPU 生成器决定再搬到 tokens 所在设备 → 换设备不改变结果、同 seed 可复现。"""
    from nmf.esm_common import apply_mlm_mask

    class _Alphabet:
        cls_idx, eos_idx, padding_idx, mask_idx = 0, 1, 2, 3

    tokens = torch.randint(4, 30, (4, 16))
    masked_a, pos_a = apply_mlm_mask(tokens, _Alphabet(), 0.3, torch.Generator().manual_seed(7))
    masked_b, pos_b = apply_mlm_mask(tokens, _Alphabet(), 0.3, torch.Generator().manual_seed(7))

    assert torch.equal(pos_a, pos_b) and torch.equal(masked_a, masked_b)
    assert masked_a.device == tokens.device and pos_a.device == tokens.device
    assert int(pos_a.sum()) > 0
    # 每条序列至少遮挡一个位置
    assert (pos_a.sum(dim=1) > 0).all()
    # 特殊 token 不会被遮挡
    assert not bool(pos_a[0, 0])
