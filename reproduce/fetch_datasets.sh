#!/usr/bin/env bash
# ============================================================================
#  数据集获取（两篇论文）
#
#  用法：
#     bash fetch_datasets.sh plan            # 只打印：要下什么、多大、从哪下
#     bash fetch_datasets.sh core            # 【必下】PDB 目标 + 模型权重 + 论文数据包（< 9 GB）
#     bash fetch_datasets.sh pdb             # 只下 PDB 结构（目标 + 位点复合物）
#     bash fetch_datasets.sh weights         # 只下模型权重
#     bash fetch_datasets.sh paperdata       # 只下论文二公开数据包
#     bash fetch_datasets.sh seqdb           # 【可选】UniRef90（论文二序列新颖性，~32 GB）
#     bash fetch_datasets.sh seqdb --release 2021_04   # 论文原始版本（~158 GB，含三级 UniRef）
#     bash fetch_datasets.sh afdb            # 【可选】AlphaFold DB swissprot 结构（图 4D）
#     bash fetch_datasets.sh pdb-snapshot    # 【可选】PDB 全库快照（论文一 TM-score 穷举，>200 GB）
#     bash fetch_datasets.sh processing      # 生成 processed/（单链 target 清洗 + 位点 JSON）
#     bash fetch_datasets.sh verify          # 校验已下载文件（存在性 + 体积 + sha256）
#
#  环境变量：
#     DATA_ROOT   数据根目录，默认 <repo>/data
#     TORCH_HOME  权重缓存根，默认 $DATA_ROOT/weights/torch
#     JOBS        并发下载数（aria2c/wget），默认 4
#     DRY_RUN=1   只打印命令
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/weights/torch}"
JOBS="${JOBS:-4}"
FBAI=https://dl.fbaipublicfiles.com/fair-esm
RCSB=https://files.rcsb.org/download
export TORCH_HOME

# ---------------------------------------------------------------- 数据清单
# 论文一：固定骨架设计的 6 个 de novo target（Methods A.3.2）
P1_BACKBONE_TARGETS="1QYS 5L33 6D0T 6MRS 6W3W 6WVS"
# 论文一：功能位点脚手架用的 5 个复合物（Methods A.3.4）
P1_SITE_COMPLEXES="1Y6K 6M0J 1GHQ 5JW3 7MMO"
# 论文二：App A.1.1 的 39 个 de novo target（去重后的结构文件）
P2_NOVO_TARGETS="1QYS 2KL8 2KPO 2LN3 2LTA 2LVB 2N2T 2N2U 2N3Z 2N76 \
4KY3 4KYZ 5CW9 5KPE 5KPH 5L33 5TPJ 5TRV 6CZG 6CZH 6CZI 6CZJ 6D0T 6DG6 \
6DKM 6DLM 6E5C 6LLQ 6MRR 6MRS 6MSP 6NUK 6W3F 6W3W 6WI5 6WVS 7MCD"

DIR_TARGETS="$DATA_ROOT/raw/pdb_targets"
DIR_COMPLEXES="$DATA_ROOT/raw/pdb_complexes"
DIR_SNAPSHOT="$DATA_ROOT/raw/pdb_snapshot_2022-08"
DIR_SEQDB="$DATA_ROOT/raw/uniref90"
DIR_AFDB="$DATA_ROOT/raw/alphafold_db"
DIR_PAPERDATA="$DATA_ROOT/raw/paper_data"
DIR_PROJ="$DATA_ROOT/raw/projection_train"
DIR_NATURAL="$DATA_ROOT/raw/natural_baseline"
MANIFEST="$DATA_ROOT/DATASETS.manifest.tsv"

# ---------------------------------------------------------------- 下载器
fetch() {
  local url="$1" out="$2" desc="${3:-}"
  if [[ -s "$out" ]]; then
    dim "  [跳过] $(basename "$out") 已存在（$(du -h "$out" | cut -f1)）"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  log "  ↓ ${desc:-$(basename "$out")}"
  dim "    $url"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  local tmp="$out.part"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x "$JOBS" -s "$JOBS" -k 1M -q --allow-overwrite=true -o "$(basename "$tmp")" -d "$(dirname "$tmp")" "$url" \
      || { warn "aria2c 失败，回退 wget"; wget -q --tries=3 -O "$tmp" "$url"; }
  else
    wget -q --tries=3 --show-progress -O "$tmp" "$url" || curl -sSL -o "$tmp" "$url"
  fi
  mv "$tmp" "$out"
  record "$out" "$url"
}

record() {
  local out="$1" url="$2"
  [[ "$DRY_RUN" == "1" ]] && return 0
  local sz sha
  sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
  sha=$(sha256sum "$out" 2>/dev/null | cut -d' ' -f1 || echo "-")
  printf '%s\t%s\t%s\t%s\n' "$(realpath --relative-to="$DATA_ROOT" "$out")" "$sz" "$sha" "$url" >> "$MANIFEST"
}

# ---------------------------------------------------------------- plan
do_plan() {
  banner "数据集清单（DATA_ROOT=$DATA_ROOT）"
  cat <<'EOT'
【必下 · core】合计约 8.8 GB
  1. 论文一 PDB 结构  6 个固定骨架 target + 5 个位点复合物（11 个文件，每个 <1 MB）
  2. 论文二 PDB 结构  39 个 de novo target（含上面的 1QYS/5L33/6D0T/6MRS/6W3W/6WVS）
  3. 模型权重（论文一）
       esmfold_3B_v1.pt                 2.6 GB   ESMFold，模拟退火的每一步都要它
       esm_if1_gvp4_t16_142M_UR50.pt    1.6 GB   ESM-IF1，逆折叠 roundtrip
  4. 模型权重（论文二）
       esm2_t33_650M_UR50D.pt           2.5 GB   ESM2 650M，设计用的语言模型
       esm2_t33_650M_UR50D-contact-regression.pt  配套 contact 回归头
       linear_projection_model.pt       0.5 MB   结构投影层（已训练，直接可用）
  5. 论文二公开数据包
       free_generations_full.db          4.9 MB   25k 自由生成的统计 + PDB
       design_lm_data_2022_v1.hdf5      51.5 MB   湿实验长表（SEC 曲线、产率）
  6. 仓库自带（无需下载）
       examples/lm-design/paper-data/{data.csv, *purge_ids.txt}
       examples/lm-design/utils/ngram_stats/*.p         n-gram 先验（unigram/bigram/trigram/quadgram）
       examples/lm-design/2N2U.pdb                      样例 target
       examples/data/*.a3m, *.fasta                     小样例

【可选 · 论文规模才需要】
  - UniRef90（论文二序列新颖性，图 2G/4F-G）
      当前版 uniref90.fasta.gz                32 GB（解压 ~80 GB）
      2021_04 原始版 uniref2021_04.tar.gz    158 GB（含 UniRef50/90/100）
  - AlphaFold DB（论文二图 4D 的结构对照）
      swissprot_pdb_v6.tar                   ~2 GB + 逐 accession 拉取（量大）
  - PDB 全库快照（论文一图 S2C/S3C 的 TM-score 穷举）  >200 GB（不解压）或 ~100 GB（.ent.gz）
  - 天然蛋白对照集（论文二 214 条；15k 天然蛋白）
  - 结构投影层训练集（Yang et al. 2020，15,051 条）—— 只有要「重训投影层」才需要
EOT
}

# ---------------------------------------------------------------- pdb
do_pdb() {
  banner "1/3 PDB 结构"
  local all="$(echo $P1_BACKBONE_TARGETS $P2_NOVO_TARGETS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
  for id in $all; do
    fetch "$RCSB/${id}.pdb" "$DIR_TARGETS/${id}.pdb" "target ${id}"
  done
  for id in $P1_SITE_COMPLEXES; do
    fetch "$RCSB/${id}.pdb" "$DIR_COMPLEXES/${id}.pdb" "site complex ${id}"
  done
  ok "PDB 结构就绪：$DIR_TARGETS（$(ls "$DIR_TARGETS" 2>/dev/null | wc -l) 个）、$DIR_COMPLEXES"
  dim "  注意：6DKM / 6DLM 在论文里出现 A/B 两个链版本（6DKM A、6DLM B），"
  dim "  工具包会下载整条 entry，取链由各自的 loader 决定。"
}

# ---------------------------------------------------------------- weights
do_weights() {
  banner "2/3 模型权重（缓存到 TORCH_HOME=$TORCH_HOME）"
  local ck="$TORCH_HOME/hub/checkpoints"
  mkdir -p "$ck"
  fetch "$FBAI/models/esmfold_3B_v1.pt"                     "$ck/esmfold_3B_v1.pt"                     "ESMFold v1（论文一）"
  fetch "$FBAI/models/esm_if1_gvp4_t16_142M_UR50.pt"        "$ck/esm_if1_gvp4_t16_142M_UR50.pt"        "ESM-IF1（论文一 roundtrip）"
  fetch "$FBAI/models/esm2_t33_650M_UR50D.pt"               "$ck/esm2_t33_650M_UR50D.pt"               "ESM2 650M（论文二）"
  fetch "$FBAI/regression/esm2_t33_650M_UR50D-contact-regression.pt" \
        "$ck/esm2_t33_650M_UR50D-contact-regression.pt"     "ESM2 650M contact 回归头"
  fetch "$FBAI/examples/lm_design/linear_projection_model.pt" \
        "$REPO_ROOT/examples/lm-design/linear_projection_model.pt" "结构投影层（论文二）"
  ok "权重就绪：$ck"
  dim "  用的时候让 python 认这个目录即可：export TORCH_HOME=$TORCH_HOME"
  dim "  或在代码里先 torch.hub.set_dir('$TORCH_HOME/hub')"
  dim "  可选的 ProteinMPNN 权重（论文一图 S2D 对照）需另从 ProteinMPNN 仓库取。"
}

# ---------------------------------------------------------------- paperdata
do_paperdata() {
  banner "3/3 论文二公开数据包"
  fetch "$FBAI/examples/lm_design/free_generations_full.db"     "$DIR_PAPERDATA/free_generations_full.db"     "25k 自由生成统计"
  fetch "$FBAI/examples/lm_design/design_lm_data_2022_v1.hdf5" "$DIR_PAPERDATA/design_lm_data_2022_v1.hdf5" "湿实验长表"
  ok "数据包就绪：$DIR_PAPERDATA"
  cat <<EOT
  用法：
    python - <<'PY'
    import pandas as pd
    df = pd.read_sql('free_generations_full', 'sqlite:///$DIR_PAPERDATA/free_generations_full.db')
    lf = pd.read_hdf('$DIR_PAPERDATA/design_lm_data_2022_v1.hdf5')
    PY
  注意：这两份是**论文的产出数据**（用于复算结论），不是训练输入。
EOT
}

# ---------------------------------------------------------------- seqdb
do_seqdb() {
  local release="${1:-current}"
  banner "UniRef90 序列库（release=$release）"
  if [[ "$release" == "2021_04" ]]; then
    local url="https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/uniref/uniref2021_04.tar.gz"
    warn "论文用的是 2021_04，但官方只提供 158 GB 的三级 UniRef 打包（含 UniRef50/100）。"
    warn "磁盘紧张时建议用当前版 UniRef90（~32 GB），只是新颖性绝对数值会略有偏移。"
    fetch "$url" "$DIR_SEQDB/uniref2021_04.tar.gz" "UniRef 2021_04 打包"
    dim "  解压后取 uniref90.fasta：tar -xzf $DIR_SEQDB/uniref2021_04.tar.gz -C $DIR_SEQDB"
  else
    fetch "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/uniref90.fasta.gz" \
          "$DIR_SEQDB/uniref90.fasta.gz" "UniRef90（当前版）"
    dim "  解压：gunzip -k $DIR_SEQDB/uniref90.fasta.gz"
    dim "  ⚠️ 论文的 purge 列表是按 2021_04 编的，用当前版时 purge 与实际库略有错位，"
    dim "     报告里要注明这一点（analyze_novelty.py 的 NOVELTY.md 会自动标注库名）。"
  fi
  ok "序列库就绪：$DIR_SEQDB"
  dim "  jackhmmer 会自己在同目录建索引；首次检索较慢，之后复用。"
}

# ---------------------------------------------------------------- afdb
do_afdb() {
  banner "AlphaFold DB（swissprot 结构 + 逐 accession 拉取说明）"
  fetch "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar" \
        "$DIR_AFDB/swissprot_pdb_v6.tar" "AFDB swissprot 结构打包"
  cat <<EOT
  论文图 4D 的做法（App A.5.2）：对每条设计在 AlphaFold DB 里找 best-domain E-value 最小的
  top-1 命中，然后比对「序列一致性」与「预测结构 TM-score」。需要两样东西：

  1) 命中序列：AF DB 覆盖的就是 UniProt 2021_04（论文当时的版本）。
     用 UniProt 2021_04 序列集：https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/knowledgebase/
     或直接用当前版 UniRef100/UniProt 检索（数值会略有偏移）。
  2) 命中结构：按 accession 逐个拉（论文当时用 v3，现在是 v6）：
       https://alphafold.ebi.ac.uk/files/AF-<UniProtID>-F1-model_v3.pdb
     批量建议先用 Foldseek 预筛，再只下 top-N 的结构，否则要拉几万个文件。

  本工具包的 analyze_novelty.py 负责 1)；第 2) 步（TM-score）目前要自己接。
EOT
  ok "AFDB 目录：$DIR_AFDB"
}

# ---------------------------------------------------------------- pdb snapshot
do_pdb_snapshot() {
  banner "PDB 全库快照（论文一图 S2C/S3C 的 TM-score 穷举）"
  warn "这是最大的可选数据：不解压 >200 GB；.ent.gz 直接喂 TM-align 也需要大量 IO。"
  cat <<EOT
  论文用的是 PDB 2022-08 快照 + TM-align 20210107，穷举找与设计结构 TM-score 最高的条目。

  推荐做法（不要真的下全库）：
    1. 先拿条目清单和日期：
       curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
       # 列：IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST, RESOLUTION, ...
    2. 用 Foldseek 在 AFDB/PDB 结构库上做一次快速预筛，取 top-100 候选：
       foldseek easy-search design.pdb pdb_db out.m8 tmp --alignment-type 1 -e 1e-3
    3. 只对候选下载 PDB 并用 TM-align 精确打分：
       $RCSB/<ID>.pdb
       tm-align -byresi design.pdb hit.pdb | grep "^TM-score"
    4. 取最大值 = 图 S2C/S3C 的「nearest-PDB TM-score」。

  如果确实要整库快照（需要 ~200 GB 以上空余）：
       rsync -av --include='*/' --include='*.ent.gz' --exclude='*' \\
         rsync.wwpdb.org::ftp_data/structures/divided/pdb/ $DIR_SNAPSHOT/
EOT
  mkdir -p "$DIR_SNAPSHOT"
  fetch "https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx" \
        "$DIR_SNAPSHOT/entries.idx" "PDB 条目索引（含发布日期）"
  ok "已下 entries.idx；整库快照请按上面的说明自行决定"
}

# ---------------------------------------------------------------- processing
do_processing() {
  banner "生成 processed/（可直接被实验脚本吃的数据）"
  local out="$DATA_ROOT/processed"
  mkdir -p "$out/scaffold_sites" "$out/lm_design_targets"

  # 论文一位点的残基区间（Methods A.3.4，闭区间；语言运行时用 [start, end) 半开区间）
  cat > "$out/scaffold_sites/paper_sites.json" <<'JSON'
{
  "_comment": "论文《A high-level programming language for generative protein design》Methods A.3.4 的五个功能位点。start/end 为论文给出的闭区间残基号；design_programs.py --start/--end 用的是 [start, end) 半开区间，故 end 需 +1。",
  "sites": [
    {"name": "il10", "pdb_id": "1y6k", "chain": "L", "residues_inclusive": ["31-40"],
     "note": "IL-10R1 结合位点"},
    {"name": "ace2", "pdb_id": "6m0j", "chain": "A", "residues_inclusive": ["5-23"],
     "note": "SARS-CoV-2 spike RBD 与 ACE2 的结合面；仓库自带 programs/functional_site_scaffolding.py 用的是 23-42 另一切法"},
    {"name": "c3d",  "pdb_id": "1ghq", "chain": "A", "residues_inclusive": ["104-126", "170-184"],
     "note": "补体受体 2 的 C3d 结合位点，两段不连续"},
    {"name": "ha2",  "pdb_id": "5jw3", "chain": "B", "residues_inclusive": ["14-21", "33-42", "45-49"],
     "note": "流感 HA2 表位（抗体 MEDI8825）"},
    {"name": "rbd",  "pdb_id": "7mmo", "chain": "C", "residues_inclusive": ["439-450", "498-506"],
     "note": "SARS-CoV-2 RBD 表位（抗体 bebtelovimab）"}
  ],
  "p1_backbone_targets": ["1QYS", "5L33", "6D0T", "6MRS", "6W3W", "6WVS"],
  "_p2_note": "论文 App A.1.1 列了 39 个 de novo target，但其中 6DKM A / 6DKM B / 6DLM A / 6DLM B 是 4 个「PDB ID + 链」条目，对应 2 个 PDB 文件。所以去重后只需下载 37 个 PDB，取链由各自的 loader 决定。",
  "p2_de_novo_targets": ["1QYS","2KL8","2KPO","2LN3","2LTA","2LVB","2N2T","2N2U","2N3Z","2N76",
    "4KY3","4KYZ","5CW9","5KPE","5KPH","5L33","5TPJ","5TRV","6CZG","6CZH",
    "6CZI","6CZJ","6D0T","6DG6","6DKM","6DLM","6E5C","6LLQ","6MRR","6MRS",
    "6MSP","6NUK","6W3F","6W3W","6WI5","6WVS","7MCD"],
  "p2_de_novo_targets_count_in_paper": 39,
  "p2_de_novo_targets_unique_pdb": 37
}
JSON
  ok "位点定义：$out/scaffold_sites/paper_sites.json"

  # 用 Python 把整条 entry 拆成单链、去水、去杂原子 —— 论文二 pipeline 需要干净的单链 PDB
  if [[ "$DRY_RUN" != "1" ]]; then
    "$PY" - "$DATA_ROOT" <<'PY'
import sys, json
from pathlib import Path
data = Path(sys.argv[1])
src, dst = data / "raw" / "pdb_targets", data / "processed" / "lm_design_targets"
dst.mkdir(parents=True, exist_ok=True)
try:
    from biotite.structure import filter_amino_acids
    from biotite.structure.io.pdb import PDBFile
except Exception as e:
    print(f"  [跳过清洗] 需要 biotite：{e}")
    raise SystemExit(0)
meta = []
for p in sorted(src.glob("*.pdb")):
    try:
        f = PDBFile.read(str(p))
        arr = f.get_structure(model=1)
        # 只留第一个蛋白链的氨基酸原子
        chains = [c for c in dict.fromkeys(arr.chain_id) if arr[arr.chain_id == c].array_length()]
        best = None
        for c in chains:
            sub = arr[arr.chain_id == c]
            sub = sub[filter_amino_acids(sub)]
            if sub.array_length() and (best is None or sub.array_length() > best.array_length()):
                best = sub
        if best is None:
            continue
        out = dst / f"{p.stem}.pdb"
        w = PDBFile()
        w.set_structure(best)
        w.write(str(out))
        nres = len(set(best.res_id))
        meta.append({"id": p.stem.upper(), "chain": str(best.chain_id[0]),
                     "n_residues": nres})
    except Exception as e:
        print(f"  [失败] {p.name}: {e!r}")
(dst / "targets.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  清洗出 {len(meta)} 个单链 target → {dst}")
lens = sorted(m["n_residues"] for m in meta)
if lens:
    print(f"  长度范围 {lens[0]}–{lens[-1]}（论文 App A.1.1 声明 67 ≤ L ≤ 184）")
PY
  fi

  # 天然对照集（论文二 214 条）只能按论文给的规则重建，这里给出可执行的筛选清单
  cat > "$out/natural_baseline_recipe.md" <<'MD'
# 论文二「天然蛋白对照集」重建规则（App A.1.1）

论文原文：*A small (N = 214) set of natural proteins with structures in the PDB was selected to
serve as a baseline comparison ... The set is composed of PDBs available on July 2020 that have
sequence identity < 0.3 to the dataset used to train the structure projection, according to
mmseqs2. A length filter of 50 ≤ L < 250 was applied.*

重建步骤：

1. 取 PDB 发布日期 ≤ 2020-07 的条目：
   `curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx`
   （第 3 列 `ACCESSION DATE` 即初次发布日期）
2. 长度过滤 50 ≤ L < 250（可由清理后的单链 PDB 统计残基数）
3. 用 mmseqs2 与「结构投影训练集」比对，保留 sequence identity < 0.3 的条目：
   `mmseqs easy-search candidate.fasta projection_train.fasta out.m8 tmp --min-seq-id 0.3`
4. 随机取 214 条（论文未给随机种子，条目集合不必逐条一致）

论文用这 214 条做的是 Fig S1 / S2 / Table S1（结构理解基线），
**不参与任何设计实验**，所以不做也不影响主结论复现。
MD
  ok "天然对照重建规则：$out/natural_baseline_recipe.md"
  ok "processed/ 就绪：$out"
}

# ---------------------------------------------------------------- verify
do_verify() {
  banner "校验已下载数据"
  [[ -f "$MANIFEST" ]] || { warn "没有 manifest（$MANIFEST），先跑 core。"; return 1; }
  local bad=0 n=0
  printf '%-58s %12s %10s\n' "文件" "大小" "sha256"
  while IFS=$'\t' read -r rel sz sha url; do
    n=$((n+1))
    local f="$DATA_ROOT/$rel"
    if [[ ! -f "$f" ]]; then
      printf '%-58s %12s %10s  ✗ 缺失\n' "$rel" "-" "-"; bad=$((bad+1)); continue
    fi
    local cur; cur=$(stat -c%s "$f")
    if [[ "$cur" != "$sz" ]]; then
      printf '%-58s %12s %10s  ✗ 体积不符（期望 %s）\n' "$rel" "$cur" "$sha" "$sz"; bad=$((bad+1)); continue
    fi
    printf '%-58s %12s %10s  ✓\n' "$rel" "$cur" "${sha:0:10}…"
  done < "$MANIFEST"
  echo
  if (( bad == 0 )); then ok "全部 $n 个文件校验通过"; else err "$bad/$n 个文件有问题"; fi
  return $(( bad > 0 ))
}

# ---------------------------------------------------------------- main
case "${1:-plan}" in
  plan)          do_plan ;;
  core)          do_pdb; do_weights; do_paperdata; do_processing; do_plan ;;
  pdb)           do_pdb ;;
  weights)       do_weights ;;
  paperdata)     do_paperdata ;;
  seqdb)         shift || true; do_seqdb "${1:-current}" ;;
  afdb)          do_afdb ;;
  pdb-snapshot)  do_pdb_snapshot ;;
  processing)    do_processing ;;
  verify)        do_verify ;;
  *)             err "未知子命令：$1（plan|core|pdb|weights|paperdata|seqdb|afdb|pdb-snapshot|processing|verify）"; exit 2 ;;
esac

echo
dim "数据根目录：$DATA_ROOT"
dim "清单文件  ：$MANIFEST"
dim "目录规范  ：data/raw/<来源> → data/processed/<用途> → data/{embeddings,structures,scores}"
