#!/usr/bin/env bash
# ============================================================================
#  论文二《Language models generalize beyond natural proteins》
#  一键跑 in-silico 设计实验 + 汇总数据
#
#  用法：
#     MODE=smoke bash run_p2.sh                 # 冒烟：2N2U × 1 seed × 300 步 + 100 步自由生成
#     MODE=full  bash run_p2.sh                 # 论文规模（需要 GPU，且要自己准备 target PDB）
#     DRY_RUN=1  bash run_p2.sh                 # 只打印计划
#     MODE=smoke ALL_TARGETS=1 bash run_p2.sh   # 先 --fetch-pdbs 拿 39 个 de novo target 再跑
#     NOVELTY_DB=/data/uniref90.fasta bash run_p2.sh   # 顺带跑序列新颖性检索
#
#  环境变量：
#     MODE            smoke（默认）| full
#     TASKS           fixedbb / free_generation / "fixedbb free_generation"
#     PDBS            fixedbb 的 target 列表，默认 "2N2U"
#     ALL_TARGETS     1 = 用论文 App A.1.1 的 39 个 de novo target（会自动下载）
#     SEEDS           seed 列表，如 "0-4"；full 默认 0-199（固定骨架）/ 0-24（自由生成）
#     NUM_ITER        MCMC 步数；full 默认 170000，smoke 默认 300
#     FG_LENGTH       自由生成长度，默认 100
#     ORACLE          none | esmfold（默认 none；论文用的是 AlphaFold）
#     DEVICE          cuda:0 | cpu
#     OUT_ROOT        默认 <repo>/reproduce/outputs
#     NOVELTY_DB      给了就跑 jackhmmer 新颖性检索（需本地序列库 fasta）
#     DRY_RUN / SKIP_AGG
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

P2_OUT="$OUT_ROOT/p2"
MODE="${MODE:-smoke}"
TASKS="${TASKS:-fixedbb free_generation}"
ORACLE="${ORACLE:-none}"
FG_LENGTH="${FG_LENGTH:-100}"
PDB_DIR="$P2_OUT/de_novo_targets"

if [[ "$MODE" == "full" ]]; then
  NUM_ITER="${NUM_ITER:-170000}"
  SEEDS_FIXEDBB="${SEEDS:-0-199}"
  SEEDS_FG="${SEEDS:-0-24}"
else
  NUM_ITER="${NUM_ITER:-300}"
  SEEDS_FIXEDBB="${SEEDS:-0-1}"
  SEEDS_FG="${SEEDS:-0-1}"
fi

if [[ "${ALL_TARGETS:-0}" == "1" ]]; then
  PDBS="${PDBS:-}"
  PDB_SRC="$PDB_DIR"
else
  PDBS="${PDBS:-2N2U}"
  # 仓库只在 examples/lm-design/ 下自带 2N2U.pdb；其余 target 要用 --fetch-pdbs 下载
  PDB_SRC="$REPO_ROOT/examples/lm-design"
fi

banner "论文二 一键复现  MODE=$MODE  NUM_ITER=$NUM_ITER  ORACLE=$ORACLE"
log "解释器    : $PY"
log "设备      : $DEVICE"
log "产物根目录: $P2_OUT"
log "任务      : $TASKS"
log "target    : ${PDBS:-<全部 39 个 de novo target>}"
log "seed      : fixedbb=$SEEDS_FIXEDBB  free_generation=$SEEDS_FG"
[[ "$DRY_RUN" == "1" ]] && warn "DRY_RUN=1：只打印，不会执行任何采样"

gpu_hint
check_python_deps torch esm hydra omegaconf nltk scipy || {
  warn "缺依赖时先跑：bash $REPRO_ROOT/00_setup_env.sh base"
  [[ "$DRY_RUN" == "1" ]] || exit 3
}

mkdir -p "$P2_OUT" "$PDB_DIR"

# ---------------------------------------------------------------- 0) 论文公开数据复算（零成本，先做）
banner "阶段 0/4：复算论文公开实验数据（不加载模型）"
run "$PY" "$HERE/paper_data_report.py" --out "$P2_OUT/paper_data_report"

# ---------------------------------------------------------------- 1) 下载 target PDB
if [[ "${ALL_TARGETS:-0}" == "1" ]]; then
  banner "阶段 1/4：下载 39 个 de novo target（App A.1.1）"
  run "$PY" "$HERE/run_lm_design_batch.py" --fetch-pdbs --pdb-dir "$PDB_DIR"
fi

# ---------------------------------------------------------------- 2) 设计实验
banner "阶段 2/4：设计实验（lm-design）"
for task in $TASKS; do
  if [[ "$task" == "fixedbb" ]]; then
    seeds="$SEEDS_FIXEDBB"
    if [[ -n "$PDBS" ]]; then
      for pdb in $PDBS; do
        log "▶ fixedbb target=$pdb"
        run_timed "fixedbb/$pdb" \
          "$PY" "$HERE/run_lm_design_batch.py" \
            --task fixedbb --pdb "$pdb" --pdb-dir "$PDB_SRC" \
            --seeds "$seeds" --num-iter "$NUM_ITER" \
            --oracle "$ORACLE" --device "$DEVICE" --out-dir "$P2_OUT" \
          || warn "target $pdb 有失败 seed（已记录 traceback，可重跑续跑）"
      done
    else
      log "▶ fixedbb 全部 39 个 de novo target（顺序执行）"
      for pdb in $(ls "$PDB_DIR"/*.pdb 2>/dev/null | xargs -n1 basename | sed 's/\.pdb$//'); do
        run_timed "fixedbb/$pdb" \
          "$PY" "$HERE/run_lm_design_batch.py" \
            --task fixedbb --pdb "$pdb" --pdb-dir "$PDB_DIR" \
            --seeds "$seeds" --num-iter "$NUM_ITER" \
            --oracle "$ORACLE" --device "$DEVICE" --out-dir "$P2_OUT" \
          || warn "target $pdb 有失败 seed"
      done
    fi
  else
    log "▶ free_generation length=$FG_LENGTH"
    run_timed "free_generation" \
      "$PY" "$HERE/run_lm_design_batch.py" \
        --task free_generation --length "$FG_LENGTH" \
        --seeds "$SEEDS_FG" --num-iter "$NUM_ITER" \
        --device "$DEVICE" --out-dir "$P2_OUT" \
      || warn "free_generation 有失败 seed"
  fi
done

# ---------------------------------------------------------------- 3) 序列新颖性
if [[ -n "${NOVELTY_DB:-}" ]]; then
  banner "阶段 3/4：序列新颖性（jackhmmer，图 2G / 4F-G）"
  run_timed "novelty" \
    "$PY" "$HERE/analyze_novelty.py" \
      --fasta-dir "$P2_OUT" --db "$NOVELTY_DB" \
      --db-name "${NOVELTY_DB_NAME:-uniref90_2021_04}" \
      --out "$P2_OUT/novelty" \
    || warn "新颖性检索未完成"
else
  log "跳过序列新颖性（想跑就设 NOVELTY_DB=<本地序列库 fasta>）"
  dim "  论文用的是 UniRef90 2021_04（约 15GB）+ 两个 purge 列表，见 analyze_novelty.py 头部说明。"
fi

# ---------------------------------------------------------------- 4) 汇总
if [[ "${SKIP_AGG:-0}" != "1" ]]; then
  banner "阶段 4/4：汇总"
  run "$PY" "$HERE/aggregate_p2.py" --root "$P2_OUT"
  ok "汇总产物：$P2_OUT/summary/P2_SUMMARY.md"
fi

banner "论文二 全部完成"
cat <<EOT
产物：
  $P2_OUT/<task>/<tag>/seed<k>/metrics.json     每条跑动的指标
  $P2_OUT/<task>/<tag>/seed<k>/trajectory.csv   能量轨迹（图 2D）
  $P2_OUT/summary/P2_SUMMARY.md                 汇总报告
  $P2_OUT/paper_data_report/PAPER_DATA_REPORT.md 论文公开数据的逐项复算（7/8 项精确命中）
  $P2_OUT/novelty/                              新颖性检索结果（若跑了）

论文中仍需外部资源才能补齐的部分：
  * AlphaFold oracle（图 2A/3C）      → ColabFold / 本地 AF2 + Amber 松弛
  * no-LM 基线（图 2C/2F）            → ColabDesign 3stage() AfDesign（commit e7bb3def）
  * Rosetta 过滤（App A.4.2-4.4）     → Rosetta PackStat / SSShapeComplementarity / SAP
  * 图 4D 的 AlphaFold DB 全库对照     → 下载 AF DB fasta + 结构文件
  * 湿实验 SEC（App A.7）             → 不可计算复现
EOT
