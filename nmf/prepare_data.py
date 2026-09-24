#!/usr/bin/env python
"""数据集准备：下载权威序列库、质量控制、去污染、按固定种子切分。

支持的数据源
------------
- ``swissprot``：UniProtKB/Swiss-Prot（reviewed，人工审阅）通过 UniProt REST 流式下载；
- ``file:<路径>``：使用本地 FASTA（例如仓库自带的 ``examples/data/*.fasta``）。

质量控制
--------
1. 只保留 20 种标准氨基酸（丢弃 B/Z/X/U/O/*/- 等非常规字符）；
2. 长度限制在 ``[min_len, max_len]``（ESM-2 的位置编码上限为 1022，默认上限取 1022）；
3. 全库精确去重（相同序列只保留第一条）；
4. **去污染**：用 6-mer 包含度作为同源性代理，剔除与 valid/test 高度相似的 train 序列
   （对应 ESM-2 论文用 MMseqs 以 50% 序列一致性清洗训练集的思路，这里用无需外部二进制的
   近似实现）。判定标准：某条 train 序列覆盖某条 valid/test 序列 ≥ ``--decontam-threshold``
   比例的 6-mer 即视为污染并剔除。

产物
----
``<out-dir>/train.fasta``、``valid.fasta``、``test.fasta`` 与 ``stats.json``
（条数、长度分位数、每个文件的 sha256、全部参数），便于复现与论文引用。

示例
----
.. code-block:: bash

    python -m nmf.prepare_data --source swissprot \
        --out-dir data/processed/swissprot \
        --train 20000 --valid 1000 --test 2000 --seed 0 --decontaminate
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Set, Tuple

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_SWISSPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
DEFAULT_SWISSPROT_QUERY = (
    "(reviewed:true) AND (fragment:false) AND (length:[50 TO 1022])"
)


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """读取 FASTA，label 取 ``>`` 后的第一个空白分隔字段。"""
    records: List[Tuple[str, str]] = []
    label = None
    chunks: List[str] = []
    with open_maybe_gzip(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    records.append((label, "".join(chunks)))
                label = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line)
    if label is not None:
        records.append((label, "".join(chunks)))
    return records


def write_fasta(path: str, records: Sequence[Tuple[str, str]], width: int = 60) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        for label, seq in records:
            handle.write(f">{label}\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i : i + width] + "\n")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download_swissprot(
    path: str,
    query: str = DEFAULT_SWISSPROT_QUERY,
    url: str = DEFAULT_SWISSPROT_URL,
    timeout: int = 3600,
    chunk_mb: int = 16,
) -> str:
    """通过 UniProt REST 流式接口下载（避免整库压缩包，便于断点式重试）。"""
    import urllib.parse
    import urllib.request

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"已存在，跳过下载：{path}")
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    full_url = f"{url}?{urllib.parse.urlencode({'query': query, 'format': 'fasta'})}"
    print(f"下载 {full_url}")
    print(f"  -> {path}")
    start = time.time()
    downloaded = 0
    with urllib.request.urlopen(full_url, timeout=timeout) as response, open(path, "wb") as out:
        while True:
            block = response.read(chunk_mb << 20)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if downloaded % (64 << 20) < (chunk_mb << 20):
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"  {downloaded / 1048576:8.1f} MB  "
                    f"({downloaded / 1048576 / elapsed:.2f} MB/s)",
                    flush=True,
                )
    print(f"下载完成：{downloaded / 1048576:.1f} MB，耗时 {time.time() - start:.0f}s")
    return path


# --------------------------------------------------------------------------- #
# 质量控制
# --------------------------------------------------------------------------- #
def quality_control(
    records: Sequence[Tuple[str, str]],
    min_len: int,
    max_len: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """标准残基过滤 + 长度过滤 + 精确去重。"""
    stats = {"input": len(records), "nonstandard": 0, "too_short": 0, "too_long": 0, "duplicate": 0}
    seen: Set[str] = set()
    kept: List[Tuple[str, str]] = []
    for label, seq in records:
        if not set(seq) <= STANDARD_AA:
            stats["nonstandard"] += 1
            continue
        if len(seq) < min_len:
            stats["too_short"] += 1
            continue
        if len(seq) > max_len:
            stats["too_long"] += 1
            continue
        if seq in seen:
            stats["duplicate"] += 1
            continue
        seen.add(seq)
        kept.append((label, seq))
    stats["kept"] = len(kept)
    return kept, stats


def kmers(seq: str, k: int) -> Set[int]:
    """序列的 k-mer 集合（用内置 hash 之外的稳定哈希，保证跨进程一致）。"""
    return {hash_kmer(seq[i : i + k]) for i in range(len(seq) - k + 1)}


def hash_kmer(kmer: str) -> int:
    """稳定的 64 位 k-mer 哈希（不用 Python 内置 hash，避免 PYTHONHASHSEED 影响）。"""
    h = 1469598103934665603  # FNV-1a
    for ch in kmer.encode():
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def decontaminate(
    train: List[Tuple[str, str]],
    guards: Sequence[Tuple[str, str]],
    threshold: float = 0.6,
    k: int = 6,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """剔除与 ``guards``（valid/test）高度相似的 train 序列。

    判定：train 序列覆盖某条 guard 序列 ≥ ``threshold`` 比例的 k-mer 即视为污染。
    实现上用倒排索引累加，复杂度约为 O(Σ|guard| × 平均倒排表长度)。
    """
    train_sets = [kmers(seq, k) for _, seq in train]
    postings: Dict[int, List[int]] = {}
    for idx, tokens in enumerate(train_sets):
        for token in tokens:
            postings.setdefault(token, []).append(idx)

    flagged: Set[int] = set()
    guard_tokens: List[Set[int]] = [kmers(seq, k) for _, seq in guards]
    guard_len = [max(len(t), 1) for t in guard_tokens]
    max_guard = max(guard_len) if guard_len else 1

    for tokens, glen in zip(guard_tokens, guard_len):
        need = threshold * glen
        counts: Dict[int, int] = {}
        for token in tokens:
            for idx in postings.get(token, ()):
                counts[idx] = counts.get(idx, 0) + 1
        for idx, shared in counts.items():
            # train ∪ guard <= train + guard，用较保守的分母（guard 长度）判断包含度
            if shared >= need:
                flagged.add(idx)
        del counts

    kept = [rec for idx, rec in enumerate(train) if idx not in flagged]
    stats = {"train_before": len(train), "removed": len(flagged), "train_after": len(kept), "k": k,
             "threshold": threshold, "max_guard_kmers": max_guard}
    if verbose:
        print(
            f"去污染：剔除 {len(flagged)}/{len(train)} 条训练序列"
            f"（{k}-mer 覆盖度 ≥ {threshold:.0%}）"
        )
    return kept, stats


def length_summary(records: Sequence[Tuple[str, str]]) -> Dict:
    lengths = sorted(len(s) for _, s in records)
    if not lengths:
        return {"n": 0}

    def pct(p: float) -> int:
        idx = min(int(round(p * (len(lengths) - 1))), len(lengths) - 1)
        return lengths[idx]

    return {
        "n": len(lengths),
        "min": lengths[0],
        "p5": pct(0.05),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": lengths[-1],
        "mean": round(sum(lengths) / len(lengths), 1),
        "total_residues": sum(lengths),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="准备用于 ESM 非负分解实验的权威数据集（下载 → 质控 → 去污染 → 切分）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="swissprot",
                        help="swissprot（联网下载）或 file:<本地 FASTA 路径>")
    parser.add_argument("--raw-file", default="data/raw/swissprot_reviewed.fasta",
                        help="swissprot 的下载缓存路径")
    parser.add_argument("--query", default=DEFAULT_SWISSPROT_QUERY, help="UniProt 查询式")
    parser.add_argument("--out-dir", default="data/processed/swissprot", help="切分结果目录")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1022,
                        help="ESM-2 位置编码上限为 1022，超长序列会被截断，故默认过滤")
    parser.add_argument("--train", type=int, default=20000, help="训练集条数")
    parser.add_argument("--valid", type=int, default=1000, help="验证集条数")
    parser.add_argument("--test", type=int, default=2000, help="测试集条数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decontaminate", action="store_true", default=True)
    parser.add_argument("--no-decontaminate", dest="decontaminate", action="store_false")
    parser.add_argument("--decontam-threshold", type=float, default=0.6)
    parser.add_argument("--kmer", type=int, default=6)
    parser.add_argument("--stats", default=None, help="统计 JSON 路径，默认 <out-dir>/stats.json")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)

    if args.source.startswith("file:"):
        raw_path = args.source.split(":", 1)[1]
        print(f"使用本地 FASTA：{raw_path}")
    else:
        raw_path = download_swissprot(args.raw_file, query=args.query)

    print(f"读取 {raw_path} …")
    records = read_fasta(raw_path)
    print(f"  原始条数：{len(records)}")

    records, qc = quality_control(records, args.min_len, args.max_len)
    print(
        f"质控后：{len(records)} 条（丢弃 非常规残基 {qc['nonstandard']}、"
        f"过短 {qc['too_short']}、过长 {qc['too_long']}、重复 {qc['duplicate']}）"
    )

    need = args.train + args.valid + args.test
    if len(records) < need:
        raise SystemExit(f"可用序列不足：需要 {need} 条，只有 {len(records)} 条。")

    random.shuffle(records)
    test = records[: args.test]
    valid = records[args.test : args.test + args.valid]
    train = records[args.test + args.valid : need]
    print(f"切分（seed={args.seed}）：train {len(train)} / valid {len(valid)} / test {len(test)}")

    decon_stats: Dict = {}
    if args.decontaminate:
        train, decon_stats = decontaminate(
            train, list(valid) + list(test), threshold=args.decontam_threshold, k=args.kmer
        )

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(args.out_dir, "train.fasta"),
        "valid": os.path.join(args.out_dir, "valid.fasta"),
        "test": os.path.join(args.out_dir, "test.fasta"),
    }
    write_fasta(paths["train"], train)
    write_fasta(paths["valid"], valid)
    write_fasta(paths["test"], test)

    splits = {"train": train, "valid": valid, "test": test}
    report = {
        "source": args.source,
        "raw_file": os.path.abspath(raw_path),
        "raw_file_sha256": sha256_file(raw_path),
        "query": args.query if not args.source.startswith("file:") else None,
        "quality_control": qc,
        "decontamination": decon_stats,
        "args": vars(args),
        "splits": {
            name: {
                "path": os.path.abspath(paths[name]),
                "sha256": sha256_file(paths[name]),
                "length": length_summary(splits[name]),
            }
            for name in ("train", "valid", "test")
        },
    }
    stats_path = args.stats or os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\n各划分的序列长度分布：")
    for name in ("train", "valid", "test"):
        s = report["splits"][name]["length"]
        print(
            f"  {name:5s} n={s['n']:6d}  长度 min={s['min']:4d} p50={s['p50']:4d} "
            f"p95={s['p95']:4d} max={s['max']:4d}  平均={s['mean']:.0f}  "
            f"总残基={s['total_residues']:,}"
        )
    print(f"\n统计与校验信息已写入：{stats_path}")


if __name__ == "__main__":
    main()
