#!/usr/bin/env bash
# ============================================================================
#  论文一《A high-level programming language for generative protein design》
#  一键跑全部设计实验 + 汇总数据
#
#  用法：
#     MODE=smoke bash run_p1.sh                      # 冒烟：每 spec 2 seed / 200 步
#     MODE=full  bash run_p1.sh                      # 论文正文规模（需要 GPU，很贵）
#     DRY_RUN=1  bash run_p1.sh                      # 只打印计划
#     TASKS="free_hallucination symmetric_monomer" MODE=full bash run_p1.sh
#     WITH_ROUNDTRIP=1 MODE=full bash run_p1.sh      # 跑完设计后再跑逆折叠 roundtrip
#
#  环境变量：
#     MODE            smoke（默认）| full
#     TASKS           空格分隔的任务名，默认全部；见 `--list-tasks`
#     OUT_ROOT        产物根目录，默认 <repo>/reproduce/outputs
#     SEEDS           覆盖每个 spec 的 seed 数
#     STEPS           覆盖模拟退火步数（full 默认 30000，smoke 默认 200）
#     WEIGHTS         paper（默认）| repo
#     DEVICE          cuda:0 | cpu（默认自动探测）
#     DRY_RUN         1 = 只打印
#     WITH_ROUNDTRIP  1 = 设计跑完后接着跑 roundtrip
#     SKIP_AGG        1 = 不跑汇总
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

P1_OUT="$OUT_ROOT/p1"
MODE="${MODE:-smoke}"
WEIGHTS="${WEIGHTS:-paper}"

if [[ "$MODE" == "full" ]]; then
  GRID="paper"
  STEPS="${STEPS:-30000}"
else
  GRID="smoke"
  STEPS="${STEPS:-200}"
fi

ALL_TASKS=(
  free_hallucination
  fixed_backbone
  secondary_structure
  functional_site_scaffolding
  symmetric_monomer
  homo_oligomer
  two_level_multimer
  symmetric_binding
  hierarchical_asymmetric
)

if [[ -n "${TASKS:-}" ]]; then
  read -r -a RUN_TASKS <<< "$TASKS"
else
  RUN_TASKS=("${ALL_TASKS[@]}")
fi

banner "论文一 一键复现  MODE=$MODE  GRID=$GRID  STEPS=$STEPS"
log "解释器    : $PY"
log "设备      : $DEVICE"
log "产物根目录: $P1_OUT"
log "任务      : ${RUN_TASKS[*]}"
log "权重口径  : $WEIGHTS"
[[ "$DRY_RUN" == "1" ]] && warn "DRY_RUN=1：只打印，不会执行任何训练"

gpu_hint
check_python_deps torch esm biotite rich || {
  warn "缺依赖时先跑：bash $REPRO_ROOT/00_setup_env.sh base"
  [[ "$DRY_RUN" == "1" ]] || exit 3
}
if [[ "$MODE" == "full" ]] && ! python_has openfold; then
  warn "缺 openfold → ESMFold 无法使用，full 模式一定失败。先跑 00_setup_env.sh esmfold"
  [[ "$DRY_RUN" == "1" ]] || exit 3
fi

mkdir -p "$P1_OUT"

# ---------------------------------------------------------------- 1) 设计实验
banner "阶段 1/3：设计实验（模拟退火 + ESMFold）"
for task in "${RUN_TASKS[@]}"; do
  log "▶ $task"
  run_timed "$task" \
    "$PY" "$HERE/design_programs.py" \
      --task "$task" \
      --grid "$GRID" \
      --steps "$STEPS" \
      --weights "$WEIGHTS" \
      --device "$DEVICE" \
      --out-dir "$P1_OUT" \
      ${SEEDS:+--seeds "$SEEDS"} \
      || warn "$task 有失败条目（已记录 traceback，可重跑续跑）"
done

# ---------------------------------------------------------------- 2) roundtrip
if [[ "${WITH_ROUNDTRIP:-0}" == "1" ]]; then
  banner "阶段 2/3：逆折叠 roundtrip（图 3C-D / 4C-D）"
  RT_N=$([[ "$MODE" == "full" ]] && echo 1000 || echo 4)
  run_timed "roundtrip" \
    "$PY" "$HERE/roundtrip.py" \
      --pdb-root "$P1_OUT" \
      --tasks symmetric_monomer two_level_multimer \
      --n-structures "$RT_N" \
      --num-samples 10 \
      --temperature 0.1 \
      --device "$DEVICE" \
      --out "$P1_OUT/roundtrip" \
    || warn "roundtrip 未完成（可重跑，会自动续跑）"
else
  log "跳过逆折叠 roundtrip（想跑就加 WITH_ROUNDTRIP=1）"
  dim "  它需要 ESM-IF1 权重（约 1.4GB）+ 每结构 10 次 ESMFold，非常耗时。"
fi

# ---------------------------------------------------------------- 3) 汇总
if [[ "${SKIP_AGG:-0}" != "1" ]]; then
  banner "阶段 3/3：汇总"
  run "$PY" "$HERE/aggregate_p1.py" --root "$P1_OUT"
  ok "汇总产物：$P1_OUT/summary/P1_SUMMARY.md"
fi

banner "论文一 全部完成"
cat <<EOT
产物：
  $P1_OUT/<task>/<spec>/seed<k>/result.json   每条跑动的完整记录
  $P1_OUT/<task>/<spec>/seed<k>/design.pdb    设计结构
  $P1_OUT/summary/P1_SUMMARY.md               对照论文图示的汇总表
  $P1_OUT/summary/p1_runs.csv                 逐跑动明细
  $P1_OUT/summary/p1_by_spec.csv              按 spec 聚合

论文中仍需外部工具才能补齐的部分：
  * 图 2C  ssAF2 pLDDT        → 单序列 AlphaFold2（ColabFold --single-sequence）
  * 图 S2C/S3C 结构新颖性      → TM-align + PDB 全库
  * 图 S2D  ProteinMPNN 对照   → 需外部 ProteinMPNN 仓库
  * Discussion 湿实验验证      → 不可计算复现
EOT
