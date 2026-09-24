"""ESM 线性层的三因子非负矩阵分解（NMF）实验工具包。

把 ESM 中任意 ``nn.Linear`` 的权重分解为三个非负矩阵，中间为方阵：

    W ≈ A @ S @ B + offset,      A/S/B >= 0

三个阶段：

- :mod:`nmf.factorize`  对指定层做离线非负分解；
- :mod:`nmf.train_nmf`  在保持非负约束下对 A/S/B 做批次训练（蒸馏 + 重构 + 交叉熵）；
- :mod:`nmf.evaluate`   比较分解模型与原模型的能力（分布一致性、遮挡预测、效率）。
"""

from .nmf_layers import (  # noqa: F401
    ThreeFactorNMFLinear,
    factorize_matrix,
    get_submodule,
    list_linear_layers,
    replace_linears_with_nmf,
)

__all__ = [
    "ThreeFactorNMFLinear",
    "factorize_matrix",
    "get_submodule",
    "list_linear_layers",
    "replace_linears_with_nmf",
]
