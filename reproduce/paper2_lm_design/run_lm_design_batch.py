#!/usr/bin/env python3
"""
论文二《Language models generalize beyond natural proteins》in-silico 设计任务批量驱动。

背景
----
仓库 examples/lm-design/ 提供了 `lm_design.py`（hydra 入口）与两个 notebook，
但一次只跑 **一个 seed 的一个任务**，且结果只打到日志里，不落盘。
论文正文的规模是：
  * 固定骨架设计：39 个 de novo target × 200 designs，每个 design 170,000 步 MCMC（App A.3.1）；
  * 自由生成：25,000 条 generation，170,000 步 blocked Gibbs（App A.3.3）。
本脚本把「多 seed × 多 target」批量化，并把每条跑动的序列 / 困惑度 /
能量轨迹落盘，已完成的自动跳过。

与论文的差异（重要）
--------------------
* 论文的结构 oracle 是 **AlphaFold**（5 个 pTM 模型选 pLDDT 最高 + Amber 松弛，App A.4.1）。
  本脚本内置的 oracle 只有 **ESMFold**（`--oracle esmfold`），两者数值不可直接混比；
  要严格复现请用 ColabFold/AF2 自己做 oracle，把结果 CSV 用 `--oracle-csv` 喂给汇总脚本。
* 论文的 no-LM 基线由 **ColabDesign** 的 `3stage() AfDesign` 产生（App A.3.2），
  那是外部仓库，本工具包不覆盖。

用法
----
  # 看依赖是否就绪
  python run_lm_design_batch.py --check

  # 冒烟：2N2U 上跑 1 个 seed、300 步
  python run_lm_design_batch.py --task fixedbb --pdb 2N2U --seeds 0 \\
      --num-iter 300 --out-dir ../outputs/p2

  # 论文规模（固定骨架，单个 target，200 designs）
  python run_lm_design_batch.py --task fixedbb --pdb 2N2U --seeds 0-199 \\
      --num-iter 170000 --oracle esmfold --out-dir ../outputs/p2

  # 下载论文 App A.1.1 的 39 个 de novo target
  python run_lm_design_batch.py --fetch-pdbs --pdb-dir ../outputs/p2/de_novo_targets

  # 自由生成
  python run_lm_design_batch.py --task free_generation --length 100 --seeds 0-4 \\
      --num-iter 170000 --out-dir ../outputs/p2

产出
----
  <out-dir>/<task>/<tag>/seed<k>/sequence.fasta
  <out-dir>/<task>/<tag>/seed<k>/metrics.json    最终能量、LM 困惑度、长度、耗时、oracle 结果
  <out-dir>/<task>/<tag>/seed<k>/trajectory.csv  每步的 total loss 与温度（用于画图 2D）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_DIR.parent
LM_DIR = REPO_ROOT / "examples" / "lm-design"

# 论文 App A.1.1 的 39 个 de novo target（6DKMA/B、6DLMA/B 只取一份结构文件）
DE_NOVO_TARGETS = [
    "1QYS", "2KL8", "2KPO", "2LN3", "2LTA", "2LVB", "2N2T", "2N2U", "2N3Z", "2N76",
    "4KY3", "4KYZ", "5CW9", "5KPE", "5KPH", "5L33", "5TPJ", "5TRV", "6CZG", "6CZH",
    "6CZI", "6CZJ", "6D0T", "6DG6", "6DKM", "6DLM", "6E5C", "6LLQ", "6MRR", "6MRS",
    "6MSP", "6NUK", "6W3F", "6W3W", "6WI5", "6WVS", "7MCD",
]

RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def log(msg: str) -> None:
    print(msg, flush=True)


# ==========================================================================
def check_deps() -> int:
    import importlib.util

    ok = True
    log("=" * 72)
    log("论文二复现：依赖体检")
    log("=" * 72)
    for mod, why in (
        ("torch", "运行时"),
        ("esm", "fair-esm 本体（本仓库）"),
        ("hydra", "lm_design.py 的配置系统"),
        ("omegaconf", "同上"),
        ("nltk", "n-gram 先验（Engram 项）"),
        ("scipy", "pdb_loader 用"),
        ("pandas", "论文数据复现（可选）"),
    ):
        hit = importlib.util.find_spec(mod) is not None
        log(f"  {'✓' if hit else '✗'} {mod:<10} {why}")
        ok = ok and hit

    log("-" * 72)
    proj = LM_DIR / "linear_projection_model.pt"
    log(f"  Projection 权重 {proj.name}: {'已存在' if proj.exists() else '首次运行会自动下载（约 50MB）'}")
    import shutil

    for cmd, why in (("wget", "下载权重/PDB"), ("jackhmmer", "序列新颖性检索（Fig 2G/4F-G）")):
        hit = shutil.which(cmd) is not None
        log(f"  {'✓' if hit else '✗'} {cmd:<10} {why}")

    try:
        import torch

        cuda = torch.cuda.is_available()
        log(f"  {'✓' if cuda else '✗'} CUDA 可用 = {cuda}   torch={torch.__version__}")
        if not cuda:
            log("     → 170,000 步 MCMC 在 CPU 上约需数天/条，请务必用 GPU。")
    except Exception as exc:
        log(f"  ✗ torch 导入失败：{exc}")
    log("=" * 72)
    return 0 if ok else 1


def fetch_pdbs(pdb_dir: Path, targets: List[str]) -> int:
    import urllib.request

    pdb_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for tid in targets:
        dest = pdb_dir / f"{tid}.pdb"
        if dest.exists() and dest.stat().st_size > 0:
            log(f"  [已存在] {dest.name}")
            n_ok += 1
            continue
        url = RCSB_URL.format(pdb=tid.upper())
        try:
            log(f"  下载 {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            dest.write_bytes(data)
            n_ok += 1
        except Exception as exc:
            log(f"  [失败] {tid}: {exc!r}")
    log(f"共 {n_ok}/{len(targets)} 个 target 就绪 → {pdb_dir}")
    log("注意：这批 de novo 结构里部分是多模型 NMR / 含多条链，"
        "论文的 pipeline 有自己的取链策略；如某 target 报错，请手工挑出单链 PDB。")
    return 0 if n_ok else 1


# ==========================================================================
def parse_seeds(spec: str) -> List[int]:
    """支持 '0-4' / '0,2,7' / '3'"""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def lm_perplexity(des) -> Optional[float]:
    """论文图 2E 的 LM perplexity：对最终序列做 mask-1-out 的伪困惑度。"""
    try:
        import torch

        with torch.no_grad():
            ce, _, _ = des.calc_sequence_loss(des.x_seqs)
        return float(torch.exp(ce.mean()).item())
    except Exception as exc:
        log(f"    [警告] 困惑度计算失败：{exc!r}")
        return None


_ESMFOLD_CACHE: Dict[str, Any] = {}


def oracle_esmfold(sequence: str, target_pdb: Optional[Path], device: str,
                   max_seq_len: int = 1022) -> Dict[str, Any]:
    """用 ESMFold 代替论文的 AlphaFold oracle（数值不可直接对比，仅作相对参考）。"""
    out: Dict[str, Any] = {}
    if len(sequence) > max_seq_len:
        out["oracle_skipped"] = f"length {len(sequence)} > --max-seq-len {max_seq_len}"
        return out
    try:
        import torch

        import esm

        if "model" not in _ESMFOLD_CACHE:
            _ESMFOLD_CACHE["model"] = esm.pretrained.esmfold_v1().eval().to(device)
            chunk = os.environ.get("ESMFOLD_CHUNK")
            if chunk:
                try:
                    _ESMFOLD_CACHE["model"].set_chunk_size(int(chunk))
                except Exception:
                    pass
        model = _ESMFOLD_CACHE["model"]

        try:
            with torch.no_grad():
                out["esmfold_pdb"] = model.infer_pdb(sequence)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.set_chunk_size(128)
            out["oracle_note"] = "OOM：已把 ESMFold 分块降到 128 后重试"
            with torch.no_grad():
                out["esmfold_pdb"] = model.infer_pdb(sequence)
        out["oracle"] = "esmfold"
        if target_pdb and target_pdb.exists():
            import io

            from biotite.structure import rmsd, superimpose
            from biotite.structure.io.pdb import PDBFile

            a = PDBFile.read(io.StringIO(out["esmfold_pdb"])).get_structure(model=1)
            b = PDBFile.read(str(target_pdb)).get_structure(model=1)
            aa = a[a.atom_name == "CA"]
            bb = b[b.atom_name == "CA"]
            n = min(aa.array_length(), bb.array_length())
            if n > 0:
                aa, bb = aa[:n], bb[:n]
                fitted, _ = superimpose(aa, bb)
                out["oracle_ca_rmsd"] = float(rmsd(aa, fitted).mean())
    except Exception as exc:
        out["oracle_error"] = repr(exc)
    return out


# ==========================================================================
def run_seed(task: str, tag: str, seed: int, args, out_dir: Path) -> Dict[str, Any]:
    """跑一个 seed。注意：必须在 lm-design 目录下、且 lm_design 只能被导入一次。"""
    import hydra
    from lm_design import Designer

    overrides = [
        f"task={task}",
        f"seed={seed}",
        "num_seqs=1",
        f"disable_cuda={'True' if args.device == 'cpu' else 'False'}",
    ]
    if args.device.startswith("cuda:"):
        overrides.append(f"cuda_device_idx={args.device.split(':')[1] or '0'}")
    if task == "fixedbb":
        overrides.append(f"pdb_fn={args.pdb_path}")
        overrides.append(f"tasks.fixedbb.num_iter={args.num_iter}")
    else:
        overrides.append(f"free_generation_length={args.length}")
        overrides.append(f"tasks.free_generation.num_iter={args.num_iter}")

    with hydra.initialize_config_module(config_module="conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    des = Designer(cfg, args.pdb_path if task == "fixedbb" else None)

    # ---- 记录能量轨迹：包一层 calc_total_loss
    traj: List[Dict[str, Any]] = []
    orig = des.calc_total_loss

    def wrapped(*a, **kw):
        total, logs = orig(*a, **kw)
        try:
            temp = None
            for key in ("accept_reject.temperature",
                        "stage_fixedbb_args.accept_reject.temperature"):
                sch = des.schedulers.get(key)
                if sch is not None:
                    temp = float(sch())
                    break
            traj.append({"step": len(traj) + 1,
                         "total_loss": float(total.detach().cpu().mean()),
                         "temperature": temp})
        except Exception:
            pass
        return total, logs

    des.calc_total_loss = wrapped

    t0 = time.time()
    des.run_from_cfg()
    seconds = time.time() - t0

    seq = des.output_seq
    metrics: Dict[str, Any] = {
        "task": task,
        "tag": tag,
        "seed": seed,
        "num_iter": args.num_iter,
        "length": len(seq),
        "sequence": seq,
        "seconds": seconds,
        "lm_perplexity": lm_perplexity(des),
        "final_total_loss": traj[-1]["total_loss"] if traj else None,
        "final_temperature": traj[-1]["temperature"] if traj else None,
        "pdb_fn": str(args.pdb_path) if task == "fixedbb" else None,
        "overrides": overrides,
    }
    if args.oracle == "esmfold":
        metrics.update(oracle_esmfold(seq, args.pdb_path if task == "fixedbb" else None,
                                     args.device, args.max_seq_len))
    return metrics, traj


def save(metrics: Dict[str, Any], traj: List[Dict[str, Any]], out_dir: Path) -> None:
    d = out_dir / metrics["task"] / metrics["tag"] / f"seed{metrics['seed']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sequence.fasta").write_text(
        f">p2_{metrics['task']}_{metrics['tag']}_seed{metrics['seed']}\n{metrics['sequence']}\n",
        encoding="utf-8",
    )
    if "esmfold_pdb" in metrics:
        (d / "esmfold_pred.pdb").write_text(metrics.pop("esmfold_pdb"), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (d / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "total_loss", "temperature"])
        w.writeheader()
        for row in traj:
            w.writerow(row)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="论文二：lm-design 多 seed 批量驱动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", choices=["fixedbb", "free_generation"], default="fixedbb")
    ap.add_argument("--pdb", default="2N2U", help="fixedbb 的 target PDB 名（仓库自带的 2N2U）")
    ap.add_argument("--pdb-dir", type=Path, default=LM_DIR,
                    help="在哪里找 <PDB>.pdb；默认 examples/lm-design（仓库只自带 2N2U.pdb）")
    ap.add_argument("--length", type=int, default=100, help="free_generation 的序列长度")
    ap.add_argument("--seeds", default="0", help="seed 列表，如 '0-4' / '0,2,7' / '3'")
    ap.add_argument("--num-iter", type=int, default=None,
                    help="MCMC 步数（论文 170000；冒烟建议 300）")
    ap.add_argument("--oracle", choices=["none", "esmfold"], default="none",
                    help="结构 oracle；论文用 AlphaFold，本工具包只能用 ESMFold 做相对参考")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    ap.add_argument("--out-dir", type=Path, default=REPRO_DIR / "outputs" / "p2")
    ap.add_argument("--fetch-pdbs", action="store_true",
                    help="下载论文 App A.1.1 的 39 个 de novo target 后退出")
    ap.add_argument("--check", action="store_true", help="依赖体检后退出")
    ap.add_argument("--dry-run", action="store_true")
    # ---- 多卡分片（配合 lib/multigpu.sh 的 run_on_gpus）
    ap.add_argument("--shard", type=int, default=0,
                    help="本进程负责第几个分片（从 0 开始）。多卡时由 multigpu.sh 注入")
    ap.add_argument("--shard-total", type=int, default=1,
                    help="分片总数（= 并行 worker 数）。按 seed 取模切分，各分片不重不漏")
    ap.add_argument("--max-seq-len", type=int, default=1022,
                    help="超过该长度的设计在 oracle 折叠时跳过（避免显存溢出）")
    args = ap.parse_args()

    if args.check:
        return check_deps()
    if args.fetch_pdbs:
        return fetch_pdbs(args.pdb_dir, DE_NOVO_TARGETS)

    # 设备归一化：multigpu.sh 的 worker 里传 "cuda"，配合 CUDA_VISIBLE_DEVICES 使用
    if args.device == "cuda":
        args.device = "cuda:0"

    if args.num_iter is None:
        args.num_iter = 170000

    seeds = parse_seeds(args.seeds)
    total_seeds = len(seeds)
    if args.shard_total > 1:
        if not (0 <= args.shard < args.shard_total):
            log(f"[错误] --shard 必须在 [0, {args.shard_total}) 内，当前 {args.shard}")
            return 2
        seeds = [s for i, s in enumerate(seeds) if i % args.shard_total == args.shard]

    tag = args.pdb.lower() if args.task == "fixedbb" else f"L{args.length}"
    args.pdb_path = (args.pdb_dir / f"{args.pdb}.pdb").resolve() if args.task == "fixedbb" else None

    log("=" * 78)
    log(f"论文二 {args.task}  tag={tag}  seeds={total_seeds}  num_iter={args.num_iter}")
    log(f"oracle   : {args.oracle}")
    log(f"设备     : {args.device}")
    log(f"输出目录 : {args.out_dir}")
    if args.shard_total > 1:
        log(f"分片     : 第 {args.shard}/{args.shard_total} 片，本进程负责 {len(seeds)} 条")
    if args.task == "fixedbb":
        log(f"target   : {args.pdb_path}  （存在={bool(args.pdb_path and args.pdb_path.exists())}）")
        if not (args.pdb_path and args.pdb_path.exists()):
            log("[错误] 找不到 target PDB。用 --fetch-pdbs 下载 39 个 de novo target，")
            log("       或 --pdb 2N2U（仓库自带），或 --pdb-dir 指向你自己的 PDB 目录。")
            return 2
    else:
        log(f"生成长度 : {args.length}")
    log(f"预计单条耗时：论文规模下 ~10 小时/条（L≈100，32GB V100）；"
        f"A6000 48GB 通常更快。本进程 {len(seeds)} 条")
    log("=" * 78)

    if args.dry_run:
        for s in seeds[:20]:
            log(f"  [dry-run] seed={s} → {args.out_dir / args.task / tag / f'seed{s}'}")
        if len(seeds) > 20:
            log(f"  ... 其余 {len(seeds) - 20} 个 seed 略")
        return 0

    # lm_design.py 强制要求 cwd 是 examples/lm-design，且要从那里 import utils
    os.chdir(LM_DIR)
    sys.path.insert(0, str(LM_DIR))

    n_ok = n_skip = n_fail = 0
    for seed in seeds:
        d = args.out_dir / args.task / tag / f"seed{seed}"
        if (d / "metrics.json").exists():
            n_skip += 1
            continue
        log(f"[seed {seed}] 开始 ...")
        try:
            metrics, traj = run_seed(args.task, tag, seed, args, args.out_dir)
            save(metrics, traj, args.out_dir)
            n_ok += 1
            log(f"[seed {seed}] 完成：len={metrics['length']} "
                f"ppl={metrics['lm_perplexity']} rmsd={metrics.get('oracle_ca_rmsd')} "
                f"({metrics['seconds']:.0f}s)")
        except KeyboardInterrupt:
            log("\n[中断] 已完成的 seed 都已落盘，重跑本命令即可续跑。")
            return 130
        except Exception as exc:
            n_fail += 1
            d.mkdir(parents=True, exist_ok=True)
            (d / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"[seed {seed}] 失败：{exc!r}（见 {d}/traceback.txt）")

    log("=" * 78)
    log(f"完成 {n_ok}，跳过 {n_skip}（已存在），失败 {n_fail}")
    log(f"下一步：python aggregate_p2.py --root {args.out_dir}")
    log("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
