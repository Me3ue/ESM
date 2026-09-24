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
source "$HERE/../lib/common.sh"   # 已内含 lib/multigpu.sh

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

mkdir -p "$P2_OUT" "$PDB_DIR" "$P2_OUT/logs"

# ---- 多卡：挑空闲卡，每张卡一个 worker，各自跑一个不重叠的 seed 分片
hw_probe; hw_auto_tune
GPU_LIST="${GPUS:-$(hw_free_gpus || true)}"
WORKER_DEVICE="cuda:0"                              # worker 内 CUDA_VISIBLE_DEVICES 已重映射
if [[ "$DEVICE" == "cpu" || -z "$GPU_LIST" ]]; then WORKER_DEVICE="cpu"; GPU_LIST=""; fi
N_SHARDS=$([[ -n "$GPU_LIST" ]] && awk -F, '{print NF}' <<< "$GPU_LIST" || echo 1)
log "参与并行的卡: ${GPU_LIST:-无（单进程）}   分片数 $N_SHARDS"
multigpu_plan

# ---------------------------------------------------------------- 0) 论文公开数据复算（零成本，先做）
banner "阶段 0/4：复算论文公开实验数据（不加载模型）"
run "$PY" "$HERE/paper_data_report.py" --out "$P2_OUT/paper_data_report"

# ---------------------------------------------------------------- 1) 下载 target PDB
if [[ "${ALL_TARGETS:-0}" == "1" ]]; then
  banner "阶段 1/4：下载 39 个 de novo target（App A.1.1）"
  run "$PY" "$HERE/run_lm_design_batch.py" --fetch-pdbs --pdb-dir "$PDB_DIR"
fi

# ---------------------------------------------------------------- 2) 设计实验
banner "阶段 2/4：设计实验（lm-design，多卡分片）"

# 一次 target 的固定骨架设计（多卡按 seed 分片）
run_fixedbb_target() {
  local pdb="$1" src="$2"
  log "▶ fixedbb target=$pdb  （$N_SHARDS 个分片并行）"
  run_on_gpus --gpus "$GPU_LIST" --log-dir "$P2_OUT/logs" -- \
    "$PY" "$HERE/run_lm_design_batch.py" \
      --task fixedbb --pdb "$pdb" --pdb-dir "$src" \
      --seeds "$SEEDS_FIXEDBB" --num-iter "$NUM_ITER" \
      --oracle "$ORACLE" --device "$WORKER_DEVICE" \
      --shard "{SHARD}" --shard-total "{NSHARD}" \
      --out-dir "$P2_OUT" \
    || warn "target $pdb 有失败 seed（日志在 $P2_OUT/logs，重跑同命令可续跑）"
}

for task in $TASKS; do
  if [[ "$task" == "fixedbb" ]]; then
    if [[ -n "$PDBS" ]]; then
      for pdb in $PDBS; do run_fixedbb_target "$pdb" "$PDB_SRC"; done
    else
      log "▶ fixedbb 全部 39 个 de novo target（顺序执行）"
      for pdb in $(ls "$PDB_DIR"/*.pdb 2>/dev/null | xargs -n1 basename | sed 's/\.pdb$//'); do
        run_fixedbb_target "$pdb" "$PDB_DIR"
      done
    fi
  else
    log "▶ free_generation length=$FG_LENGTH  （$N_SHARDS 个分片并行）"
    run_on_gpus --gpus "$GPU_LIST" --log-dir "$P2_OUT/logs" -- \
      "$PY" "$HERE/run_lm_design_batch.py" \
        --task free_generation --length "$FG_LENGTH" \
        --seeds "$SEEDS_FG" --num-iter "$NUM_ITER" \
        --device "$WORKER_DEVICE" \
        --shard "{SHARD}" --shard-total "{NSHARD}" \
        --out-dir "$P2_OUT" \
      || warn "free_generation 有失败 seed（日志在 $P2_OUT/logs）"
  fi
done

# ---------------------------------------------------------------- 3) 序列新颖性
if [[ -n "${NOVELTY_DB:-}" ]]; then
  banner "阶段 3/4：序列新颖性（jackhmmer，图 2G / 4F-G）"
  log "jackhmmer 线程数 = $JACKHMMER_CPU（按 $HW_CPU_CORES 核自动定，可用 JACKHMMER_CPU 覆盖）"
  run_timed "novelty" \
    "$PY" "$HERE/analyze_novelty.py" \
      --fasta-dir "$P2_OUT" --db "$NOVELTY_DB" \
      --db-name "${NOVELTY_DB_NAME:-uniref90_2021_04}" \
      --cpu "$JACKHMMER_CPU" \
      --out "$P2_OUT/novelty" \
    || warn "新颖性检索未完成"
else
  log "跳过序列新颖性（想跑就设 NOVELTY_DB=<本地序列库 fasta>）"
  dim "  论文用的是 UniRef90 2021_04（158GB 打包）或当前版（32GB）+ 两个 purge 列表，"
  dim "  见 DATASETS.md §3.4。本机内存 $((HW_MEM_TOTAL_MB/1024))GB，大库建议放数据盘而不是 /dev/shm。"
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
