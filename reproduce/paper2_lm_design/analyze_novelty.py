#!/usr/bin/env python3
"""
论文二：序列新颖性检索（图 2G、图 4F-G，附录 A.5.1 / A.5.2 / A.5.3）。

论文做法
--------
把每条设计序列用 **jackhmmer 3.3.2** 检索大型序列库，三个非默认设置（App A.5.1）：
  * `-n 1`（只迭代一次，避免对 de novo 序列产生大量伪命中）
  * `--seed 0`（可复现）
  * 命中按 **best-domain E-value** 排序（而非 full-sequence）
报告的三个量（App A.5.3）：
  1. E-value：top 命中的 best-domain E-value（< 1 视为显著）
  2. Sequence identity：所有显著命中里 **最大** 的序列一致性
     （用 biotite 的最优局部比对算，分母是 query 全长）
  3. TM-score：top-10 命中的预测结构里 **最大** 的 TM-score（需要命中蛋白的 AlphaFold DB 结构）
并且必须**剔除**从 ESM2 训练集里被移除的序列（App A.1.2 的两个 purge 列表，
仓库已提供：examples/lm-design/paper-data/*.txt）。

本脚本做什么
------------
调用 jackhmmer CLI → 解析 `--domtblout` → 算上面 1、2（3 需要逐个下载 AF DB 结构，
用 `--tm-score` 开启，会慢很多），输出 CSV 与汇总。

用法
----
  # 冒烟（用小的 Swiss-Prot 库，验证流程）
  python analyze_novelty.py --fasta-dir ../outputs/p2 --db swissprot.fasta \\
      --db-name swissprot --out ../outputs/p2/novelty

  # 论文口径（UniRef90 2021_04，约 15GB，先下好）
  python analyze_novelty.py --fasta-dir ../outputs/p2 \\
      --db /data/uniref90.fasta --db-name uniref90_2021_04 \\
      --purge examples/lm-design/paper-data/uniref90_jackhmmer_purge_ids.txt \\
              examples/lm-design/paper-data/artificial_sequence_purge_ids.txt \\
      --out ../outputs/p2/novelty

  # 论文图 4D 的口径：对全部 free generation 检索 AlphaFold DB
  #   （需要本地 AF DB 序列文件；本机没有就先用 UniRef90，并在报告里注明）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPRO_DIR.parent
PAPER_DATA = REPO_ROOT / "examples" / "lm-design" / "paper-data"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
def collect_sequences(fasta_dir: Path, fasta_file: Optional[Path]) -> List[Tuple[str, str]]:
    """从单个 FASTA 或 <out>/<task>/<tag>/seed<k>/sequence.fasta 树里收集序列。"""
    recs: List[Tuple[str, str]] = []

    def parse(path: Path) -> None:
        name, buf = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    recs.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
        if name:
            recs.append((name, "".join(buf)))

    if fasta_file:
        parse(fasta_file)
    for p in sorted((fasta_dir or Path(".")).rglob("sequence.fasta")):
        parse(p)
    # 去重（按序列）
    seen: Set[str] = set()
    uniq: List[Tuple[str, str]] = []
    for n, s in recs:
        if s and s not in seen:
            seen.add(s)
            uniq.append((n, s))
    return uniq


def load_purge_ids(paths: List[Path]) -> Set[str]:
    ids: Set[str] = set()
    for p in paths:
        if not p.exists():
            log(f"[警告] purge 列表不存在：{p}")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.add(line)
                # UniRef 常见形式：UniRef90_P12345 / sp|P12345|...
                for token in line.replace("|", " ").split():
                    ids.add(token)
    return ids


def hit_is_purged(hit_name: str, purged: Set[str]) -> bool:
    if not purged:
        return False
    if hit_name in purged:
        return True
    for token in hit_name.replace("|", " ").split():
        base = token.split("_")[-1] if "_" in token else token
        if token in purged or base in purged:
            return True
    return False


# --------------------------------------------------------------------------
def run_jackhmmer(query_fasta: Path, db: Path, out_prefix: Path, cpu: int) -> Optional[Path]:
    """跑 jackhmmer（-n 1 --seed 0），返回 domtblout 路径。"""
    domtbl = out_prefix.with_suffix(".domtblout")
    tblout = out_prefix.with_suffix(".tblout")
    stdout = out_prefix.with_suffix(".out")
    cmd = [
        "jackhmmer",
        "-n", "1",
        "--seed", "0",
        "--cpu", str(cpu),
        "--domtblout", str(domtbl),
        "--tblout", str(tblout),
        "-o", str(stdout),
        str(query_fasta),
        str(db),
    ]
    log("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        log("[错误] 找不到 jackhmmer。装法：conda install -c bioconda hmmer=3.3.2")
        return None
    except subprocess.CalledProcessError as exc:
        log(f"[错误] jackhmmer 失败：{exc.stderr[-800:] if exc.stderr else exc}")
        return None
    return domtbl


def parse_domtblout(path: Path) -> List[Dict[str, object]]:
    """解析 domtblout：我们只要每条命中的 best-domain E-value 与 target 名。"""
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 13:
            continue
        try:
            rows.append(
                {
                    "target": f[0],
                    "query": f[3],
                    "full_evalue": float(f[6]),
                    "full_score": float(f[7]),
                    "dom_evalue": float(f[12]),   # 第 13 列 = domain i-Evalue
                    "dom_score": float(f[13]),
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def sequence_identity(query: str, hit_fasta_path: Path, hit_names: Set[str]) -> Dict[str, float]:
    """按论文口径算 query 与命中的最大序列一致性（biotite 最优局部比对，分母=query 全长）。"""
    try:
        import biotite.sequence.align as align
        from biotite.sequence.io import fasta as btfasta
        from biotite.sequence import ProteinSequence
    except Exception as exc:
        log(f"  [警告] biotite 不可用，跳过序列一致性：{exc!r}")
        return {}

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    q = ProteinSequence(query)

    best: Dict[str, float] = {}
    seen_hits = 0
    for header, seq in btfasta.FastaFile.read_iter(str(hit_fasta_path)):
        key = header.split()[0]
        if key not in hit_names:
            continue
        seen_hits += 1
        try:
            h = ProteinSequence(str(seq))
            aln = align.align_optimal(q, h, matrix, gap_penalty=(-10, -1), terminal_penalty=False)[0]
            ident = sum(a == b for a, b in aln.trace if a != -1 and b != -1) / max(1, len(query))
            best[key] = float(ident)
        except Exception:
            continue
        if seen_hits >= len(hit_names):
            break
    return best


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="论文二：jackhmmer 序列新颖性分析")
    ap.add_argument("--fasta-dir", type=Path, default=REPRO_DIR / "outputs" / "p2",
                    help="p2 产物根目录（递归找 sequence.fasta）")
    ap.add_argument("--fasta-file", type=Path, default=None, help="也可直接给一个 FASTA")
    ap.add_argument("--db", type=Path, required=False, help="检索用序列库（fasta，需已解压）")
    ap.add_argument("--db-name", default="uniref90_2021_04",
                    help="库名，仅用于报告标注（论文口径：uniref90_2021_04 或 AlphaFold DB）")
    ap.add_argument("--purge", nargs="*", type=Path,
                    default=[PAPER_DATA / "uniref90_jackhmmer_purge_ids.txt",
                             PAPER_DATA / "artificial_sequence_purge_ids.txt"],
                    help="必须剔除的命中 ID 列表（论文 App A.1.2）")
    ap.add_argument("--evalue-threshold", type=float, default=1.0,
                    help="显著命中阈值（论文：best-domain E-value < 1）")
    ap.add_argument("--cpu", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--out", type=Path, default=None, help="输出目录（默认 <fasta-dir>/novelty）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out or (args.fasta_dir / "novelty")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "novelty.csv"

    seqs = collect_sequences(args.fasta_dir, args.fasta_file)
    log(f"收集到 {len(seqs)} 条设计序列")
    if not seqs:
        log("[错误] 没有序列。先跑 run_lm_design_batch.py，或用 --fasta-file 直接指定。")
        return 2

    if not args.db:
        log("[提示] 未指定 --db，无法真正检索。论文用的是 UniRef90 2021_04（约 15GB）。")
        log("       下载：https://ftp.uniprot.org/pub/databases/uniprot/2021_04/uniref/uniref90/uniref90.fasta.gz")
        log("       先解压再运行，例如：gunzip -k uniref90.fasta.gz")
        log("       冒烟可用小库：Swiss-Prot fasta（~90MB），把 --db-name 改成 swissprot。")
        log(f"       序列清单已写到 {out_dir/'designs.fasta'}，方便你手工跑 jackhmmer。")
        with (out_dir / "designs.fasta").open("w", encoding="utf-8") as f:
            for n, s in seqs:
                f.write(f">{n}\n{s}\n")
        return 2

    if not args.db.exists():
        log(f"[错误] 序列库不存在：{args.db}")
        return 2
    if shutil.which("jackhmmer") is None:
        log("[错误] 找不到 jackhmmer。装法：conda install -c bioconda hmmer=3.3.2")
        return 2

    purged = load_purge_ids(args.purge)
    log(f"purge 列表共 {len(purged)} 个 ID 将被剔除")

    rows: List[Dict[str, object]] = []
    workdir = Path(tempfile.mkdtemp(prefix="novelty_", dir=os.environ.get("TMPDIR", None)))
    for i, (name, seq) in enumerate(seqs, start=1):
        qf = workdir / f"q{i}.fasta"
        qf.write_text(f">{name}\n{seq}\n", encoding="utf-8")
        prefix = workdir / f"q{i}"
        if args.dry_run:
            log(f"  [dry-run] jackhmmer -n 1 --seed 0 {qf.name} {args.db.name}")
            continue

        domtbl = run_jackhmmer(qf, args.db, prefix, args.cpu)
        if domtbl is None:
            rows.append({"name": name, "length": len(seq), "status": "jackhmmer_failed"})
            continue

        hits = [h for h in parse_domtblout(domtbl) if not hit_is_purged(str(h["target"]), purged)]
        hits.sort(key=lambda h: float(h["dom_evalue"]))
        significant = [h for h in hits if float(h["dom_evalue"]) < args.evalue_threshold]

        best_evalue = float(hits[0]["dom_evalue"]) if hits else None
        max_ident = None
        if significant:
            hit_names = {str(h["target"]).split()[0] for h in significant}
            idmap = sequence_identity(seq, args.db, hit_names)
            if idmap:
                max_ident = max(idmap.values())

        rows.append({
            "name": name,
            "length": len(seq),
            "n_hits": len(hits),
            "n_significant": len(significant),
            "best_dom_evalue": best_evalue,
            "max_seq_identity_sig": max_ident,
            "top_hit": str(hits[0]["target"]) if hits else None,
            "status": "ok",
        })
        log(f"  [{i}/{len(seqs)}] {name}  hits={len(hits)} sig={len(significant)} "
            f"E={best_evalue} id={max_ident}")

    if args.dry_run:
        log(f"[dry-run] {len(seqs)} 条序列 × jackhmmer，未执行。临时目录：{workdir}")
        return 0

    cols = ["name", "length", "n_hits", "n_significant", "best_dom_evalue",
            "max_seq_identity_sig", "top_hit", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- 汇总
    ok = [r for r in rows if r.get("status") == "ok"]
    no_sig = [r for r in ok if not r.get("n_significant")]
    ids = [r["max_seq_identity_sig"] for r in ok if r.get("max_seq_identity_sig") is not None]
    summary = {
        "db": args.db_name,
        "db_path": str(args.db),
        "n_designs": len(rows),
        "n_ok": len(ok),
        "n_no_significant_hit": len(no_sig),
        "median_max_seq_identity": (sorted(ids)[len(ids) // 2] if ids else None),
        "n_below_20pct_identity": sum(1 for v in ids if v < 0.20),
        "n_below_30pct_identity": sum(1 for v in ids if v < 0.30),
        "purge_lists": [str(p) for p in args.purge],
        "evalue_threshold": args.evalue_threshold,
        "jackhmmer_settings": "-n 1 --seed 0, 按 best-domain E-value 排序",
    }
    (out_dir / "novelty_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    med = summary["median_max_seq_identity"]
    md = [
        "# 序列新颖性（jackhmmer）",
        "",
        f"- 检索库：`{args.db_name}`（{args.db}）",
        f"- 检索设置：`-n 1 --seed 0`，按 best-domain E-value 排序（论文 App A.5.1）",
        f"- 显著阈值：best-domain E-value < {args.evalue_threshold}",
        f"- 设计数：{len(rows)}，成功检索：{len(ok)}，**无显著命中**：{len(no_sig)}",
        (f"- 显著命中的最大序列一致性：中位 {med:.3f}"
         if med is not None else "- 序列一致性：无可用结果"),
        f"- 序列一致性 < 20% 的设计数：{summary['n_below_20pct_identity']}",
        f"- 序列一致性 < 30% 的设计数：{summary['n_below_30pct_identity']}",
        "",
        "> 论文口径提醒：正文里的“novel”指的是对 **UniRef90 2021_04** 检索的结果，",
        "> 且必须剔除 ESM2 训练集里被移除的序列（`paper-data/*.txt` 两个 purge 列表）。",
        "> 若这里用的是 Swiss-Prot 之类的更小库，命中会更少，不能与论文数字直接比较。",
        "",
    ]
    md_path = out_dir / "NOVELTY.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    log(f"已写出：{csv_path}")
    log(f"已写出：{out_dir/'novelty_summary.json'}")
    log(f"已写出：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
