#!/usr/bin/env python3
"""
论文一《A high-level programming language for generative protein design》
全部设计任务的统一驱动脚本。

背景
----
仓库 examples/protein-programming-language/ 只提供了「program 构造函数」
（programs/*.py）+ 语言运行时（language/），**没有任何命令行入口**：
论文正文里的每一次优化跑动（free hallucination 200 seed、固定骨架 ≥50 seed、
对称单体 180 跑动 ……）都需要自己写驱动。本脚本就是那个驱动。

它做四件事：
  1. 按论文正文的网格（grid）枚举 spec × seed；
  2. 用仓库自带的构造函数（或本文件内等价实现）构造 ProgramNode；
  3. 用 language.run_simulated_annealing 跑模拟退火（论文参数：30000 步、Tmax=1、Tmin=1e-4）；
  4. 把结果（序列 / PDB / 各项能量 / pLDDT / pTM / 耗时）逐条落盘，
     已存在的运行自动跳过，便于断点续跑。

用法
----
  # 看有哪些任务、每个任务对应论文哪张图
  python design_programs.py --list-tasks

  # 环境体检（缺什么依赖会明确告诉你）
  python design_programs.py --check

  # 只打印计划，不跑（强烈建议先来一遍）
  python design_programs.py --task free_hallucination --grid paper --dry-run

  # 冒烟：每个 spec 2 个 seed、200 步
  python design_programs.py --task free_hallucination --grid smoke --out-dir /tmp/p1

  # 论文规模
  python design_programs.py --task free_hallucination --grid paper \
      --out-dir reproduce/outputs/p1 --device cuda:0

落盘结构
--------
  <out-dir>/<task>/<spec>/seed<k>/result.json     ← 一条跑的完整记录
  <out-dir>/<task>/<spec>/seed<k>/design.pdb      ← 设计出的结构（ESMFold 预测）
  <out-dir>/<task>/<spec>/seed<k>/design.fasta    ← 设计出的序列
  <out-dir>/<task>/<spec>/seed<k>/status.json     ← 状态（running/done/failed），用于续跑

注意
----
* 论文正文里每一次「跑动」= 一次 30,000 步的模拟退火，每一步都要过一次 ESMFold。
  单条跑动在 32GB V100 上论文未给具体耗时，但 CPU 上是「小时~天」量级。
  因此 --grid paper 会真的提交数千条跑动，请先 --dry-run 并估算资源。
* 本脚本把论文里没有给出程序文件的实验（固定骨架的 6 个 target、homo-oligomer、
  非对称层级对称等）按论文 Methods A.3 的权重与长度约束「等价实现」，
  并在 result.json 的 "weights_profile" 字段标明用了哪一套权重。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# 路径：把 examples/protein-programming-language 加进 sys.path
# --------------------------------------------------------------------------
REPRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_DIR.parent
PPL_DIR = REPO_ROOT / "examples" / "protein-programming-language"

if not PPL_DIR.is_dir():
    sys.stderr.write(
        f"[致命] 找不到 {PPL_DIR}\n"
        "请确认本脚本位于 <repo>/reproduce/paper1_programming/ 下。\n"
    )
    raise SystemExit(2)
sys.path.insert(0, str(PPL_DIR))

# 论文 Methods A.1.4 的退火参数
TMAX = 1.0
TMIN = 1e-4

# 仓库自带程序中所有用到的「单一目标」PDB
TARGET_PDBS = ["1qys", "5l33", "6d0t", "6mrs", "6w3w", "6wvs"]

# 论文 Methods A.3.4 的五个功能位点（PDB、链、残基区间，按论文正文）
PAPER_SITES = {
    # name: (pdb_id, chain, start, end_inclusive)
    "il10": ("1y6k", "L", 31, 40),
    "ace2": ("6m0j", "A", 5, 23),
    "c3d": ("1ghq", "A", 104, 184),   # 论文是两个不连续区间 104-126 与 170-184
    "ha2": ("5jw3", "B", 14, 49),     # 论文是 14-21、33-42、45-49
    "rbd": ("7mmo", "C", 439, 506),   # 论文是 439-450、498-506
}


# ==========================================================================
# 结果记录
# ==========================================================================
@dataclass
class RunResult:
    task: str
    spec: str
    seed: int
    steps: int
    length: Optional[int]
    final_sequence: str
    energy: float
    energy_terms: List[Dict[str, Any]] = field(default_factory=list)
    plddt: Optional[float] = None
    ptm: Optional[float] = None
    seconds: float = 0.0
    weights_profile: str = "paper"
    annealing_rate: float = 0.0
    tmax: float = TMAX
    tmin: float = TMIN
    device: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "result_format": 1,
            "task": self.task,
            "spec": self.spec,
            "seed": self.seed,
            "steps": self.steps,
            "length": self.length,
            "final_sequence": self.final_sequence,
            "energy": self.energy,
            "energy_terms": self.energy_terms,
            "plddt": self.plddt,
            "ptm": self.ptm,
            "seconds": self.seconds,
            "weights_profile": self.weights_profile,
            "annealing_rate": self.annealing_rate,
            "tmax": self.tmax,
            "tmin": self.tmin,
            "device": self.device,
            "extra": self.extra,
        }


def annealing_rate_for(steps: int, tmax: float = TMAX, tmin: float = TMIN) -> float:
    """论文 A.1.4：T_i = Tmax * (Tmin/Tmax)^(i/M)。返回每步的衰减因子。"""
    if steps <= 1:
        return 1.0
    return (tmin / tmax) ** (1.0 / steps)


# ==========================================================================
# 任务注册表：每个任务给出 (论文位置, 说明, spec 枚举函数, 构造函数)
# ==========================================================================
@dataclass
class TaskSpec:
    """一个 spec = 一条配置（例如「3 重对称、总长 200」），下面挂若干 seed。"""

    name: str
    kwargs: Dict[str, Any]
    seeds: int
    label: str = ""

    @property
    def spec_id(self) -> str:
        return self.name


def _len_from_kwargs(kw: Dict[str, Any]) -> Optional[int]:
    for k in ("length", "sequence_length", "total_length", "protomer_length"):
        if k in kw:
            return int(kw[k])
    if "num_protomers" in kw and "protomer_length" in kw:
        return int(kw["num_protomers"]) * int(kw["protomer_length"])
    return None


# ---------------------------------------------------------------- builders
def build_free_hallucination(kw, weights, **_) :
    from programs.free_hallucination import free_hallucination

    return free_hallucination(int(kw["length"]))


def build_fixed_backbone(kw, weights, **_):
    """论文 A.3.2：w=2 on dRMSD；w=1 on cRMSD/pTM/pLDDT；w=0.5 on hydrophobics。

    仓库的 programs/fixed_backbone.py 只硬编码了 6MRS 且没有设权重（全 1）；
    这里按论文正文的权重与任意 target 实现等价版本。
    """
    from biotite.database.rcsb import fetch

    from language import (
        FixedLengthSequenceSegment,
        MaximizePLDDT,
        MaximizePTM,
        MinimizeCRmsd,
        MinimizeDRmsd,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
        pdb_file_to_atomarray,
        sequence_from_atomarray,
    )

    pdb_id = str(kw["pdb_id"])
    template = pdb_file_to_atomarray(fetch(pdb_id, format="pdb"))
    seq_len = len(sequence_from_atomarray(template))

    if weights == "repo":
        w_drmsd, w_crmsd, w_ptm, w_plddt, w_hydro = 1.0, 1.0, 1.0, 1.0, 1.0
    else:  # paper
        w_drmsd, w_crmsd, w_ptm, w_plddt, w_hydro = 2.0, 1.0, 1.0, 1.0, 0.5

    return ProgramNode(
        sequence_segment=FixedLengthSequenceSegment(seq_len),
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MinimizeCRmsd(template=template, backbone_only=True),
            MinimizeDRmsd(template=template, backbone_only=True),
        ],
        energy_function_weights=[w_ptm, w_plddt, w_hydro, w_crmsd, w_drmsd],
    )


def build_secondary_structure(kw, weights, **_):
    """论文 A.3.3：w=10 on SS 约束，w=1 on pTM/pLDDT/hydrophobics。

    两个子节点各自带一个 MatchSecondaryStructure，权重挂在子节点上；
    仓库版（programs/secondary_structure.py）没设权重，默认全 1。
    """
    from language import (
        FixedLengthSequenceSegment,
        MatchSecondaryStructure,
        MaximizePLDDT,
        MaximizePTM,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
    )

    node1_sse, node2_sse = str(kw["node1_sse"]), str(kw["node2_sse"])
    seg = int(kw.get("segment_length", 50))
    w_ss = 1.0 if weights == "repo" else 10.0

    def _ss_node(sse: str) -> ProgramNode:
        return ProgramNode(
            sequence_segment=FixedLengthSequenceSegment(seg),
            energy_function_terms=[MatchSecondaryStructure(sse)],
            energy_function_weights=[w_ss],
        )

    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
        ],
        energy_function_weights=[1.0, 1.0, 1.0],
        children=[_ss_node(node1_sse), _ss_node(node2_sse)],
    )


def build_scaffolding_repo(kw, weights, **_):
    """仓库自带：programs/functional_site_scaffolding.py::scaffolding_ace2()"""
    from programs.functional_site_scaffolding import scaffolding_ace2

    return scaffolding_ace2()


def build_scaffolding_generic(kw, weights, **_):
    """论文 A.3.4：单功能位点脚手架（通用版，可指定任意 PDB 与残基段）。

    论文权重：w=2 on cRMSD 与 dRMSD；w=1 on pTM/pLDDT/hydrophobics；
    位点子树额外加 w=1 的 surface exposure。
    仓库自带版本(ACE2)对 (surface_exposure, cRMSD, dRMSD) 用的是 [1,10,10]。
    """
    from biotite.database.rcsb import fetch

    from language import (
        ConstantSequenceSegment,
        MaximizePLDDT,
        MaximizePTM,
        MaximizeSurfaceExposure,
        MinimizeCRmsd,
        MinimizeDRmsd,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
        VariableLengthSequenceSegment,
        get_atomarray_in_residue_range,
        pdb_file_to_atomarray,
        sequence_from_atomarray,
    )

    pdb_id = str(kw["pdb_id"])
    # 注意：get_atomarray_in_residue_range(atoms, start, end) 语义是 res_id ∈ [start, end)
    start, end = int(kw["start"]), int(kw["end"])
    atoms = get_atomarray_in_residue_range(
        pdb_file_to_atomarray(fetch(pdb_id, format="pdb")), start=start, end=end
    )
    site_seq = sequence_from_atomarray(atoms)

    flank = int(kw.get("flank_length", 50))
    leader = VariableLengthSequenceSegment(flank)
    follower = VariableLengthSequenceSegment(flank)
    site = ConstantSequenceSegment(site_seq)

    if weights == "repo":
        site_w = [1.0, 10.0, 10.0]
        top_w = None
    else:
        site_w = [1.0, 2.0, 2.0]
        top_w = [1.0, 1.0, 1.0]

    top_terms = [
        MaximizePTM(),
        MaximizePLDDT(),
        MinimizeSurfaceHydrophobics(),
    ]
    top = ProgramNode(
        energy_function_terms=top_terms,
        children=[
            ProgramNode(sequence_segment=leader),
            ProgramNode(
                sequence_segment=site,
                energy_function_terms=[
                    MaximizeSurfaceExposure(),
                    MinimizeCRmsd(template=atoms),
                    MinimizeDRmsd(template=atoms),
                ],
                energy_function_weights=site_w,
            ),
            ProgramNode(sequence_segment=follower),
        ],
    )
    if top_w is not None:
        top.energy_function_weights = top_w
    return top


def build_symmetric_monomer(kw, weights, **_):
    """论文 A.3.5：单链 K 重旋转对称。

    论文：K∈{3..8} × 总长∈{200,300,400} × 10 seed = 180 条跑动；
    对称约束放在顶层节点，K 个 child 共享同一段序列（sequence tying）。
    """
    from language import (
        FixedLengthSequenceSegment,
        MaximizePLDDT,
        MaximizePTM,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
        SymmetryRing,
    )

    K = int(kw["num_protomers"])
    total = int(kw["total_length"])
    protomer_len = max(1, round(total / K))

    protomer = FixedLengthSequenceSegment(protomer_len)

    def _child():
        return ProgramNode(sequence_segment=protomer)

    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            SymmetryRing(),
            MinimizeSurfaceHydrophobics(),
        ],
        energy_function_weights=[1.0, 1.0, 1.0, 1.0],
        children=[_child() for _ in range(K)],
    )


def build_homo_oligomer(kw, weights, **_):
    """论文 A.3.5 后半：4/6/8 聚体，顶层用「球状对称」约束（globular symmetry），
    去掉 single-chain 约束，每个 terminal 上加 w=0.1 的 globularity 约束，
    整个复合物总长约束为 720 残基。"""
    from language import (
        FixedLengthSequenceSegment,
        MaximizeGlobularity,
        MaximizePLDDT,
        MaximizePTM,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
        SymmetryRing,
    )

    K = int(kw["num_protomers"])
    total = int(kw.get("total_length", 720))
    protomer_len = max(1, round(total / K))
    protomer = FixedLengthSequenceSegment(protomer_len)

    def _child():
        return ProgramNode(
            sequence_segment=protomer,
            energy_function_terms=[MaximizeGlobularity()],
            energy_function_weights=[0.1],
        )

    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            SymmetryRing(all_to_all_protomer_symmetry=True),
            MinimizeSurfaceHydrophobics(),
        ],
        energy_function_weights=[1.0, 1.0, 1.0, 1.0],
        children=[_child() for _ in range(K)],
        # 去掉「单链」约束 → 各 protomer 是独立链（多聚体）
        children_are_different_chains=True,
    )


def build_two_level_repo(kw, weights, **_):
    """仓库自带：programs/symmetric_two_level_multimer.py"""
    from programs.symmetric_two_level_multimer import symmetric_two_level_multimer

    return symmetric_two_level_multimer(
        num_chains=int(kw["num_chains"]),
        num_protomers_per_chain=int(kw["num_protomers_per_chain"]),
        protomer_sequence_length=int(kw["protomer_length"]),
    )


def build_symmetric_binding_repo(kw, weights, **_):
    """仓库自带：programs/symmetric_binding.py::symmetric_binding_il10(n)"""
    from programs.symmetric_binding import symmetric_binding_il10

    return symmetric_binding_il10(num_binding_sites=int(kw.get("num_binding_sites", 3)))


def build_hierarchical_asymmetric(kw, weights, **_):
    """论文 A.3.10 / 图 5C-F：三层约束。

    顶层 x1 有两个「非对称」子单元；每个子单元自身又是一个双层对称体
    （x2 → 若干 chain，chain → 若干 protomer，两层都用旋转对称约束）。
    kw: sub_a_num_chains, sub_a_num_protomers, sub_b_num_chains,
        sub_b_num_protomers, protomer_length
    """
    from language import (
        FixedLengthSequenceSegment,
        MaximizeGlobularity,
        MaximizePLDDT,
        MaximizePTM,
        MinimizeSurfaceHydrophobics,
        ProgramNode,
        SymmetryRing,
    )

    plen = int(kw.get("protomer_length", 50))
    protomer = FixedLengthSequenceSegment(plen)

    def _subunit(num_chains: int, num_protomers: int) -> ProgramNode:
        def _chain():
            return ProgramNode(
                energy_function_terms=[SymmetryRing(), MaximizeGlobularity()],
                energy_function_weights=[1.0, 1.0],
                children=[ProgramNode(sequence_segment=protomer) for _ in range(num_protomers)],
            )

        return ProgramNode(
            children=[_chain() for _ in range(num_chains)],
            children_are_different_chains=True,   # 每个 chain 是独立链
        )

    sub_a = int(kw.get("sub_a_num_chains", 2))
    sub_a_p = int(kw.get("sub_a_num_protomers", 2))
    sub_b = int(kw.get("sub_b_num_chains", 2))
    sub_b_p = int(kw.get("sub_b_num_protomers", 2))

    # 顶层：两个子单元之间不再加对称约束（"非对称地组合"），但它们是不同的链
    return ProgramNode(
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
        ],
        energy_function_weights=[1.0, 1.0, 1.0],
        children=[_subunit(sub_a, sub_a_p), _subunit(sub_b, sub_b_p)],
        children_are_different_chains=True,
    )


# ==========================================================================
# 任务定义
# ==========================================================================
def spec_grid(task: str, grid: str, args) -> List[TaskSpec]:
    """按论文正文枚举 spec × seed 数。grid='paper' 用论文规模，'smoke' 用极小规模。"""
    paper = grid == "paper"
    seeds_paper_default = 10
    specs: List[TaskSpec] = []

    def nseeds(paper_n: int) -> int:
        if args.seeds is not None:
            return int(args.seeds)
        return paper_n if paper else 2

    if task == "free_hallucination":
        # 图 2A-C：200 条跑动；论文未在正文给出长度，这里默认 100（可 --length 覆盖）
        specs.append(
            TaskSpec(
                name=f"len{args.length}",
                kwargs={"length": args.length},
                seeds=nseeds(200),
                label="图2A-C 自由幻觉（论文 200 seeds × 30000 步）",
            )
        )

    elif task == "fixed_backbone":
        # 图 2D-F：6 个 target × ≥50 seeds
        for pdb in TARGET_PDBS:
            specs.append(
                TaskSpec(
                    name=pdb,
                    kwargs={"pdb_id": pdb},
                    seeds=nseeds(50),
                    label=f"图2D-F 固定骨架设计 target={pdb.upper()}（论文 ≥50 seeds）",
                )
            )

    elif task == "secondary_structure":
        # 图 2G：3 个程序 × 10 seeds
        for n1, n2, tag in (("a", "a", "all_alpha"), ("b", "b", "all_beta"), ("a", "b", "mixed_ab")):
            specs.append(
                TaskSpec(
                    name=tag,
                    kwargs={"node1_sse": n1, "node2_sse": n2, "segment_length": args.segment_length},
                    seeds=nseeds(10),
                    label=f"图2G 二级结构设计 {tag}（论文 10 seeds）",
                )
            )

    elif task == "functional_site_scaffolding":
        # 图 2H：5 个位点 × 1000 seeds；仓库只自带 ACE2
        specs.append(
            TaskSpec(
                name="ace2_repo",
                kwargs={},
                seeds=nseeds(1000),
                label="图2H ACE2 脚手架（用仓库自带的 scaffolding_ace2）",
            )
        )
        for name, (pdb_id, chain, start, end) in PAPER_SITES.items():
            # 论文给的是闭区间残基号；语言运行时用的是 [start, end) 半开区间
            specs.append(
                TaskSpec(
                    name=f"{name}_generic",
                    kwargs={
                        "pdb_id": pdb_id,
                        "chain": chain,
                        "start": start,
                        "end": end + 1,
                        "flank_length": args.flank_length,
                    },
                    seeds=nseeds(1000),
                    label=f"图2H {name.upper()} 脚手架通用实现（论文 1000 seeds）",
                )
            )

    elif task == "symmetric_monomer":
        # 图 3B / S2A-C：6 种对称 × 3 种总长 × 10 seeds = 180
        for K in range(3, 9):
            for total in (200, 300, 400):
                specs.append(
                    TaskSpec(
                        name=f"K{K}_len{total}",
                        kwargs={"num_protomers": K, "total_length": total},
                        seeds=nseeds(10),
                        label=f"图3B {K} 重对称 / 总长 {total}（论文 10 seeds，共 180 跑动）",
                    )
                )

    elif task == "homo_oligomer":
        # 图 3E：4/6/8 聚体 × 10 seeds（论文各 oligomerization level 10 seeds）
        for K in (4, 6, 8):
            specs.append(
                TaskSpec(
                    name=f"oligo{K}",
                    kwargs={"num_protomers": K, "total_length": args.total_length},
                    seeds=nseeds(10),
                    label=f"图3E {K} 聚体（球状对称，论文 10 seeds，总长 {args.total_length}）",
                )
            )

    elif task == "two_level_multimer":
        # 图 4B / S3A-C：top×bottom ∈ [2,4]² = 9 个程序 × 10 seeds = 90
        def total_for(nc: int, npc: int) -> int:
            if (nc, npc) == (2, 2):
                return 200
            if (nc, npc) == (2, 3):
                return 250
            if (nc, npc) in ((2, 4), (3, 2), (3, 3), (4, 2)):
                return 400
            if (nc, npc) in ((3, 4), (4, 3)):
                return 450
            return 500

        for nc in (2, 3, 4):
            for npc in (2, 3, 4):
                prot_len = max(1, total_for(nc, npc) // (nc * npc))
                specs.append(
                    TaskSpec(
                        name=f"top{nc}_bot{npc}",
                        kwargs={
                            "num_chains": nc,
                            "num_protomers_per_chain": npc,
                            "protomer_length": prot_len,
                        },
                        seeds=nseeds(10),
                        label=f"图4B 顶层{nc} × 底层{npc}（论文 10 seeds，共 90 跑动）",
                    )
                )

    elif task == "symmetric_binding":
        # 图 5B / S4A-C：3 个程序 × 20 seeds = 60
        specs.append(
            TaskSpec(
                name="il10_3fold",
                kwargs={"num_binding_sites": 3},
                seeds=nseeds(20),
                label="图5B IL10 位点 3 重对称脚手架（论文 20 seeds）",
            )
        )
        specs.append(
            TaskSpec(
                name="ace2_3fold",
                kwargs={"num_binding_sites": 3},
                seeds=nseeds(20),
                label="图S4C ACE2 位点 3 重对称脚手架（论文 20 seeds）"
                     "—— 注：仓库只自带 IL10，ACE2 需自行换模板",
            )
        )
        specs.append(
            TaskSpec(
                name="ace2_5fold",
                kwargs={"num_binding_sites": 5},
                seeds=nseeds(20),
                label="图S4C ACE2 位点 5 重对称脚手架（论文 20 seeds）",
            )
        )

    elif task == "hierarchical_asymmetric":
        # 图 5C-F / S4D-E：3 个程序 × 10 seeds = 30
        combos = [
            ("asym_2x2__2x2", 2, 2, 2, 2),
            ("asym_2x2__3x2", 2, 2, 3, 2),
            ("asym_2x2__2x3", 2, 2, 2, 3),
        ]
        for tag, ca, pa, cb, pb in combos:
            specs.append(
                TaskSpec(
                    name=tag,
                    kwargs={
                        "sub_a_num_chains": ca,
                        "sub_a_num_protomers": pa,
                        "sub_b_num_chains": cb,
                        "sub_b_num_protomers": pb,
                        "protomer_length": args.protomer_length,
                    },
                    seeds=nseeds(10),
                    label=f"图5C-F 非对称组合 {tag}（论文 10 seeds，共 30 跑动）",
                )
            )

    else:
        raise KeyError(task)

    return specs


TASKS: Dict[str, Dict[str, Any]] = {
    "free_hallucination": {
        "figure": "图 2A-C",
        "builder": build_free_hallucination,
        "desc": "自由幻觉：只约束 pTM/pLDDT/表面疏水",
    },
    "fixed_backbone": {
        "figure": "图 2D-F",
        "builder": build_fixed_backbone,
        "desc": "固定骨架设计：6 个 de novo target 骨架 + RMSD 约束",
    },
    "secondary_structure": {
        "figure": "图 2G + S1A-C",
        "builder": build_secondary_structure,
        "desc": "二级结构设计：两段子序列分别指定 α/β",
    },
    "functional_site_scaffolding": {
        "figure": "图 2H + S1D",
        "builder": None,  # 由 _pick_builder 分派
        "desc": "单功能位点脚手架：5 个天然结合位点",
    },
    "symmetric_monomer": {
        "figure": "图 3A-D + S2A-C",
        "builder": build_symmetric_monomer,
        "desc": "单链 K 重旋转对称（K=3..8，总长 200/300/400）",
    },
    "homo_oligomer": {
        "figure": "图 3E",
        "builder": build_homo_oligomer,
        "desc": "同源寡聚体 4/6/8 聚体（球状对称）",
    },
    "two_level_multimer": {
        "figure": "图 4A-D + S3A-C",
        "builder": build_two_level_repo,
        "desc": "双层对称同源寡聚体（顶层 2-4 × 底层 2-4）",
    },
    "symmetric_binding": {
        "figure": "图 5A-B + S4A-C",
        "builder": build_symmetric_binding_repo,
        "desc": "对称功能位点脚手架（IL10 / ACE2，3 或 5 重对称）",
    },
    "hierarchical_asymmetric": {
        "figure": "图 5C-F + S4D-E",
        "builder": build_hierarchical_asymmetric,
        "desc": "非对称地组合两个双层对称单元（三层约束）",
    },
}


def _pick_builder(task: str, spec: TaskSpec) -> Callable:
    if task == "functional_site_scaffolding":
        return build_scaffolding_repo if spec.name == "ace2_repo" else build_scaffolding_generic
    return TASKS[task]["builder"]


# ==========================================================================
# 跑一条
# ==========================================================================
def run_one(
    task: str,
    spec: TaskSpec,
    seed: int,
    out_dir: Path,
    args,
    callback,
    steps: int,
):
    builder = _pick_builder(task, spec)
    program = builder(spec.kwargs, args.weights)

    from language import run_simulated_annealing

    rate = annealing_rate_for(steps)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    t0 = time.time()
    optimized = run_simulated_annealing(
        program=program,
        initial_temperature=TMAX,
        annealing_rate=rate,
        total_num_steps=steps,
        folding_callback=callback,
        display_progress=False,
    )
    seconds = time.time() - t0

    # 终态评估：重新 fold 一次拿 pLDDT/pTM/各项约束值
    sequence, residue_indices = optimized.get_sequence_and_set_residue_index_ranges()
    folding_output = callback.fold(sequence, residue_indices)
    term_rows = []
    total = 0.0
    for name, weight, fn in optimized.get_energy_term_functions():
        value = float(fn(folding_output))
        term_rows.append({"name": name, "weight": float(weight), "value": value})
        total += float(weight) * value

    res = RunResult(
        task=task,
        spec=spec.name,
        seed=seed,
        steps=steps,
        length=len(sequence),
        final_sequence=sequence,
        energy=total,
        energy_terms=term_rows,
        plddt=float(folding_output.plddt),
        ptm=float(folding_output.ptm),
        seconds=seconds,
        weights_profile=args.weights,
        annealing_rate=rate,
        device=args.device,
    )
    return res, folding_output.atoms


def save_result(res: RunResult, out_dir: Path, atoms=None) -> None:
    d = out_dir / res.task / res.spec / f"seed{res.seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(
        json.dumps(res.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (d / "design.fasta").write_text(
        f">p1_{res.task}_{res.spec}_seed{res.seed}\n{res.final_sequence}\n", encoding="utf-8"
    )
    if atoms is not None:
        try:
            from biotite.structure.io.pdb import PDBFile

            f = PDBFile()
            f.set_structure(atoms)
            f.write(str(d / "design.pdb"))
        except Exception as exc:  # pragma: no cover
            (d / "design.pdb.error.txt").write_text(repr(exc), encoding="utf-8")
    (d / "status.json").write_text(
        json.dumps({"state": "done", "seconds": res.seconds}), encoding="utf-8"
    )


# ==========================================================================
# 依赖体检 / 任务列表
# ==========================================================================
def do_check() -> int:
    import importlib.util

    print("=" * 72)
    print("论文一复现：依赖体检")
    print("=" * 72)
    ok = True
    for mod, why in (
        ("torch", "模型运行时"),
        ("esm", "fair-esm 本体（本仓库）"),
        ("biotite", "P-SEA 二级结构 / SASA / Kabsch 叠合 / RCSB 下载"),
        ("rich", "language/optimize.py 的进度表"),
        ("openfold", "ESMFold 必需；缺它则无法做任何设计实验"),
        ("numpy", "数值"),
    ):
        hit = importlib.util.find_spec(mod) is not None
        print(f"  {'✓' if hit else '✗'} {mod:<10} {why}")
        ok = ok and hit

    import shutil

    print("-" * 72)
    for cmd, why in (("nvcc", "OpenFold CUDA kernel 编译"), ("wget", "权重下载")):
        hit = shutil.which(cmd) is not None
        print(f"  {'✓' if hit else '✗'} {cmd:<10} {why}")
    print("-" * 72)
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  {'✓' if cuda else '✗'} CUDA 可用 = {cuda}  torch={torch.__version__}")
        if not cuda:
            print("     → 论文正文规模的实验（3 万步 × 数百 seed）在 CPU 上不可完成。")
    except Exception as exc:
        print(f"  ✗ torch 导入失败：{exc}")
    print("=" * 72)
    return 0 if ok else 1


def list_tasks() -> None:
    print("=" * 96)
    print(f"{'task':<28}{'论文位置':<20}{'paper 规模':<34}说明")
    print("-" * 96)
    scale = {
        "free_hallucination": "1 程序 × 200 seeds",
        "fixed_backbone": "6 targets × ≥50 seeds = 300",
        "secondary_structure": "3 程序 × 10 seeds = 30",
        "functional_site_scaffolding": "5 位点 × 1000 seeds = 5000",
        "symmetric_monomer": "6 对称 × 3 长度 × 10 = 180",
        "homo_oligomer": "3 聚体水平 × 10 = 30",
        "two_level_multimer": "9 程序 × 10 seeds = 90",
        "symmetric_binding": "3 程序 × 20 seeds = 60",
        "hierarchical_asymmetric": "3 程序 × 10 seeds = 30",
    }
    for name, meta in TASKS.items():
        print(f"{name:<28}{meta['figure']:<20}{scale[name]:<34}{meta['desc']}")
    print("=" * 96)
    print("另需单独运行（见 roundtrip.py / novelty 脚本）：")
    print("  逆折叠 roundtrip            图 3C-D、图 4C-D、S2D  （对 1000 个设计结构 × 10 条 ESM-IF1 采样）")
    print("  结构新颖性 TM-score         图 S2C、S3C        （TM-align vs PDB 2022-08 全库）")
    print("  湿实验验证                  Discussion         （体外表达 + SEC，不可计算复现）")


# ==========================================================================
# main
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="论文一全部设计任务的统一驱动（模拟退火 + ESMFold）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", choices=sorted(TASKS.keys()), help="要跑的设计任务")
    p.add_argument("--grid", choices=["paper", "smoke"], default="smoke",
                   help="paper=论文正文规模；smoke=极小规模打通流程（默认 smoke）")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "outputs" / "p1")
    p.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    p.add_argument("--seeds", type=int, default=None, help="覆盖每个 spec 的 seed 数")
    p.add_argument("--steps", type=int, default=None,
                   help="覆盖模拟退火步数（论文 30000；smoke 默认 200）")
    p.add_argument("--length", type=int, default=100, help="自由幻觉序列长度（论文未给，默认 100）")
    p.add_argument("--segment-length", type=int, default=50, help="二级结构设计每段长度")
    p.add_argument("--flank-length", type=int, default=50, help="脚手架两侧自由段长度")
    p.add_argument("--protomer-length", type=int, default=50, help="层级非对称设计的 protomer 长度")
    p.add_argument("--total-length", type=int, default=720, help="寡聚体复合物总长（论文 720）")
    p.add_argument("--weights", choices=["paper", "repo"], default="paper",
                   help="paper=按论文 Methods 的权重；repo=按仓库自带程序（多为全 1）")
    p.add_argument("--no-skip-done", action="store_true", help="不跳过已完成的跑动")
    p.add_argument("--dry-run", action="store_true", help="只打印计划")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--check", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_tasks:
        list_tasks()
        return 0
    if args.check:
        return do_check()
    if not args.task:
        parse_args(["--help"])
        return 2

    steps = args.steps if args.steps is not None else (30000 if args.grid == "paper" else 200)
    specs = spec_grid(args.task, args.grid, args)
    total_runs = sum(s.seeds for s in specs)

    print("=" * 78)
    print(f"任务      : {args.task}  ({TASKS[args.task]['figure']})")
    print(f"论文位置  : {TASKS[args.task]['desc']}")
    print(f"网格      : {args.grid}   步数/跑动: {steps}   seed 合计: {total_runs}")
    print(f"权重口径  : {args.weights}")
    print(f"输出目录  : {args.out_dir}")
    print("=" * 78)
    for s in specs[:12]:
        print(f"  spec={s.name:<20} seeds={s.seeds:<5} kwargs={s.kwargs}")
    if len(specs) > 12:
        print(f"  ... 其余 {len(specs) - 12} 个 spec 略")
    print("-" * 78)
    if args.grid == "paper" and steps >= 30000:
        print("提示：论文规模下总折叠次数 ≈ seed 数 × 30000，请在 GPU 上分段跑。")

    if args.dry_run:
        print("[dry-run] 不执行。去掉 --dry-run 即开始。")
        return 0

    # ---- 依赖检查 + 加载 ESMFold
    from language import EsmFoldv1

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[设备] {device}")
    if device == "cpu":
        print("[警告] CPU 模式下 ESMFold 极慢；MODE=smoke 也会很慢。")

    print("[1/2] 加载 ESMFold v1 ...")
    callback = EsmFoldv1()
    callback.load(device=device)

    # ---- 逐条跑
    done = skipped = failed = 0
    for spec in specs:
        for seed in range(spec.seeds):
            d = args.out_dir / args.task / spec.name / f"seed{seed}"
            if not args.no_skip_done and (d / "design.fasta").exists():
                skipped += 1
                continue
            try:
                print(f"[2/2] {args.task}/{spec.name}/seed{seed} ...", flush=True)
                res, atoms = run_one(args.task, spec, seed, args.out_dir, args, callback, steps)
                save_result(res, args.out_dir, atoms=atoms)
                done += 1
                print(f"      → pLDDT={res.plddt:.3f} pTM={res.ptm:.3f} "
                      f"E={res.energy:.3f}  ({res.seconds:.1f}s)")
            except KeyboardInterrupt:
                print("\n[中断] 已完成的跑动都已落盘，重跑本命令即可续跑。")
                return 130
            except Exception as exc:
                failed += 1
                d.mkdir(parents=True, exist_ok=True)
                (d / "status.json").write_text(
                    json.dumps({"state": "failed", "error": repr(exc)}, ensure_ascii=False),
                    encoding="utf-8",
                )
                (d / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"      ! 失败：{exc!r}（已记录到 {d}/traceback.txt）")

    print("=" * 78)
    print(f"完成 {done} 条，跳过 {skipped} 条（已存在），失败 {failed} 条")
    print(f"产物：{args.out_dir / args.task}")
    print("下一步：python aggregate_p1.py --root", args.out_dir)
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
