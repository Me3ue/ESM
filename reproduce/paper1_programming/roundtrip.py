#!/usr/bin/env python3
"""
论文一：逆折叠 roundtrip 实验（图 3C-D、图 4C-D、图 S2D、图 S2/3/4 的“可设计性”证据）。

论文做法（Methods A.3.8）
------------------------
给定优化过程产出的一个预测结构：
  1. 用 **ESM-IF1**（独立训练的逆折叠模型）在温度 0.1 下采样 **10 条序列**；
  2. 把每条序列再喂给 **ESMFold** 预测结构；
  3. 计算「起始骨架 vs roundtrip 骨架」的 cRMSD，取 10 条里最小的那个；
  4. 同时报告 ESM-IF1 对该样本的 **perplexity**。
论文对 1000 个结构重复此流程，得到「pLDDT–roundtrip RMSD」与「perplexity–roundtrip RMSD」
两张密度散点图，并观察到「越自信的设计越容易 roundtrip 成功」。

与论文的差异（重要，别当成 bug）
--------------------------------
* 论文的 1000 个结构是从 180 条单链对称轨迹里**均匀采样的中间结构**；
  而 design_programs.py 只落盘**终态**结构。所以本脚本对终态结构做 roundtrip，
  统计口径不同（趋势可对比，绝对值不宜逐点对齐）。
* 图 S2D 用的是 **ProteinMPNN** 而不是 ESM-IF1，需要外部仓库，本脚本不覆盖。

用法
----
  # 对 symmetric_monomer 的 1000 个终态结构做 roundtrip
  python roundtrip.py --pdb-root ../outputs/p1 --tasks symmetric_monomer \\
      --n-structures 1000 --num-samples 10 --temperature 0.1 \\
      --device cuda:0 --out ../outputs/p1/roundtrip

  # 冒烟
  python roundtrip.py --pdb-root ../outputs/p1 --tasks symmetric_monomer \\
      --n-structures 4 --num-samples 2 --no-esmfold

  # 不重新折叠（只算 ESM-IF1 perplexity，省一半算力）
  python roundtrip.py ... --no-esmfold

产出
----
  <out>/roundtrip.jsonl   每个结构一行：起始 pLDDT、10 条样本的 perplexity 与 cRMSD、最小 cRMSD
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_DIR.parent


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# cRMSD：两份 PDB 的骨架原子叠合后求 RMSD
# --------------------------------------------------------------------------
def backbone_crmsd(pdb_a: str, pdb_b: str) -> Tuple[Optional[float], str]:
    """返回 (cRMSD, 使用模式)。模式为 'N,CA,C,O' 或降级的 'CA'。"""
    from biotite.structure import rmsd, superimpose
    from biotite.structure.io.pdb import PDBFile

    def _load(pdb_text: str):
        return PDBFile.read(io.StringIO(pdb_text)).get_structure(model=1)

    a, b = _load(pdb_a), _load(pdb_b)

    for names, mode in ((("N", "CA", "C", "O"), "N,CA,C,O"), (("CA",), "CA")):
        aa = a[[n in names for n in a.atom_name]]
        bb = b[[n in names for n in b.atom_name]]
        if aa.array_length() == 0 or bb.array_length() == 0:
            continue
        if aa.array_length() != bb.array_length():
            continue
        try:
            fitted, _ = superimpose(aa, bb)
            return float(rmsd(aa, fitted).mean()), mode
        except Exception:
            continue
    return None, "unavailable"


# --------------------------------------------------------------------------
# ESM-IF1
# --------------------------------------------------------------------------
class IF1:
    def __init__(self, device: str = "cpu"):
        import torch

        import esm.inverse_folding
        import esm as esm_mod

        self.torch = torch
        self.if_mod = esm.inverse_folding
        self.esm = esm_mod
        self.device = torch.device(device)
        self.model, self.alphabet = esm_mod.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model = self.model.eval().to(self.device)

    def coords(self, pdb_path: Path, chain: Optional[str]) -> Tuple[Any, str]:
        return self.if_mod.util.load_coords(str(pdb_path), chain)

    def sample(self, coords, num_samples: int, temperature: float) -> List[str]:
        out = []
        for _ in range(num_samples):
            seq = self.model.sample(coords, temperature=temperature, device=self.device)
            out.append(seq)
        return out

    def perplexity(self, coords, seq: str) -> float:
        ll, _ = self.if_mod.util.score_sequence(self.model, self.alphabet, coords, seq)
        return float(math.exp(-float(ll) / max(1, len(seq))))


# --------------------------------------------------------------------------
# ESMFold
# --------------------------------------------------------------------------
class Folder:
    def __init__(self, device: str = "cpu"):
        import esm as esm_mod

        self.esm = esm_mod
        self.model = esm_mod.pretrained.esmfold_v1().eval().to(device)
        self.device = device

    def fold(self, seq: str) -> str:
        return self.model.infer_pdb(seq)


# --------------------------------------------------------------------------
def collect_structures(pdb_root: Path, tasks: List[str], n: int, seed: int) -> List[Dict[str, Any]]:
    """收集 (task, spec, seed, design.pdb, result.json) 组合，随机抽样 n 个。"""
    items: List[Dict[str, Any]] = []
    for task in tasks:
        base = pdb_root / task
        if not base.is_dir():
            log(f"[跳过] 没有 {base}")
            continue
        for pdb in sorted(base.rglob("design.pdb")):
            run_dir = pdb.parent
            res_path = run_dir / "result.json"
            res = {}
            if res_path.exists():
                try:
                    res = json.loads(res_path.read_text(encoding="utf-8"))
                except Exception:
                    res = {}
            items.append(
                {
                    "task": task,
                    "spec": res.get("spec", run_dir.parent.name),
                    "seed": res.get("seed", -1),
                    "pdb": str(pdb),
                    "starting_plddt": res.get("plddt"),
                    "starting_ptm": res.get("ptm"),
                    "sequence": res.get("final_sequence"),
                }
            )
    rng = random.Random(seed)
    rng.shuffle(items)
    if n and n < len(items):
        items = items[:n]
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="论文一：ESM-IF1 roundtrip 实验")
    ap.add_argument("--pdb-root", type=Path,
                    default=REPRO_DIR / "outputs" / "p1",
                    help="design_programs.py 的 --out-dir")
    ap.add_argument("--tasks", nargs="+",
                    default=["symmetric_monomer", "two_level_multimer"],
                    help="参与 roundtrip 的任务（论文图 3C-D 对应 symmetric_monomer，"
                         "图 4C-D 对应 two_level_multimer）")
    ap.add_argument("--n-structures", type=int, default=1000, help="抽多少个结构（论文 1000）")
    ap.add_argument("--num-samples", type=int, default=10, help="每个结构采样几条序列（论文 10）")
    ap.add_argument("--temperature", type=float, default=0.1, help="ESM-IF1 采样温度（论文 0.1）")
    ap.add_argument("--chain", default=None, help="单链模式下的链 ID（默认 None=第一条链）")
    ap.add_argument("--out", type=Path, default=None, help="输出目录（默认 <pdb-root>/roundtrip）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-esmfold", action="store_true",
                    help="不重新折叠（只算 ESM-IF1 perplexity，省算力）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out or (args.pdb_root / "roundtrip")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "roundtrip.jsonl"

    items = collect_structures(args.pdb_root, args.tasks, args.n_structures, args.seed)
    log(f"待处理结构：{len(items)} 个（任务：{', '.join(args.tasks)}）")
    if not items:
        log(f"[错误] 在 {args.pdb_root} 下没找到 design.pdb。"
            "先跑 design_programs.py，或用 --pdb-root 指定正确目录。")
        return 2

    est = len(items) * args.num_samples
    log(f"需要：{est} 次 ESM-IF1 采样" + ("" if args.no_esmfold else f" + {est} 次 ESMFold 折叠"))
    if args.dry_run:
        for it in items[:5]:
            log(f"  [dry-run] {it['task']}/{it['spec']}/seed{it['seed']}  {it['pdb']}")
        log("  ...")
        return 0

    # 已完成的跳过，便于续跑
    done_keys = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done_keys.add((row.get("task"), row.get("spec"), row.get("seed")))
            except Exception:
                pass
    todo = [it for it in items if (it["task"], it["spec"], it["seed"]) not in done_keys]
    log(f"已完成 {len(done_keys)} 个，本次处理 {len(todo)} 个（断点续跑）")

    if not todo:
        log("无需处理。")
        return 0

    if1 = IF1(args.device)
    folder = None if args.no_esmfold else Folder(args.device)

    n_ok = 0
    with jsonl.open("a", encoding="utf-8") as f:
        for i, it in enumerate(todo, start=1):
            try:
                coords, native = if1.coords(Path(it["pdb"]), args.chain)
                samples = if1.sample(coords, args.num_samples, args.temperature)
                per_sample: List[Dict[str, Any]] = []
                for s in samples:
                    entry: Dict[str, Any] = {"sequence": s}
                    try:
                        entry["if1_perplexity"] = if1.perplexity(coords, s)
                    except Exception as exc:
                        entry["if1_perplexity"] = None
                        entry["perplexity_error"] = repr(exc)
                    if folder is not None:
                        try:
                            pred = folder.fold(s)
                            cr, mode = backbone_crmsd(Path(it["pdb"]).read_text(), pred)
                            entry["roundtrip_crmsd"] = cr
                            entry["crmsd_mode"] = mode
                        except Exception as exc:
                            entry["roundtrip_crmsd"] = None
                            entry["fold_error"] = repr(exc)
                    per_sample.append(entry)

                crmsds = [e.get("roundtrip_crmsd") for e in per_sample
                          if e.get("roundtrip_crmsd") is not None]
                ppls = [e.get("if1_perplexity") for e in per_sample
                        if e.get("if1_perplexity") is not None]

                row = {
                    "task": it["task"],
                    "spec": it["spec"],
                    "seed": it["seed"],
                    "pdb": it["pdb"],
                    "starting_plddt": it["starting_plddt"],
                    "starting_ptm": it["starting_ptm"],
                    "num_samples": len(per_sample),
                    "temperature": args.temperature,
                    "if1_perplexity": (sum(ppls) / len(ppls)) if ppls else None,
                    "if1_perplexity_min": min(ppls) if ppls else None,
                    "roundtrip_crmsd_min": min(crmsds) if crmsds else None,
                    "roundtrip_crmsd_mean": (sum(crmsds) / len(crmsds)) if crmsds else None,
                    "per_sample": per_sample,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                log(f"[{i}/{len(todo)}] {it['task']}/{it['spec']}/seed{it['seed']} "
                    f"min_cRMSD={row['roundtrip_crmsd_min']} ppl={row['if1_perplexity']}")
            except KeyboardInterrupt:
                log("\n[中断] 已完成的记录都已落盘，重跑本命令即可续跑。")
                return 130
            except Exception as exc:
                log(f"[{i}/{len(todo)}] 失败：{exc!r}")

    log(f"完成 {n_ok}/{len(todo)}，结果：{jsonl}")
    log("下一步：python aggregate_p1.py --root " + str(args.pdb_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
