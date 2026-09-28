#!/usr/bin/env bash
# 完整实验矩阵：数据 → {满阶,半秩} 分解 → {满阶,半秩} 训练 → 论文级评测 → 指标汇总
#
#   bash nmf/run_matrix.sh
#
# 已经存在的产物会被**自动跳过**，所以可以随时中断、随时重跑（长任务建议配合 tee 存日志）。
#
# 常用环境变量（全部可选）：
#   DRY_RUN=1                只打印将要执行的命令，不真正跑
#   DEVICE=cuda              计算设备（默认 auto：有 CUDA 就用 GPU）；分解阶段可用
#                            FACTORIZE_DEVICE 单独指定（HALS 未必适合 GPU）
#   TAGS="fullrank"          只跑某个 tag（默认 "fullrank halfrank"）
#   OUTDIR=...               产物目录（默认 nmf/outputs/fullmodel）
#   FORCE_DATA=1 FORCE_FACTORIZE=1 FORCE_TRAIN=1 FORCE_BENCH=1   强制重跑对应阶段
#   N_EVAL / MASK_ROUNDS / PPL_RECORDS / PPL_MAX_LEN             评测规模（默认 64 / 5 / 8 / 256）
#   EPOCHS / MAX_RECORDS / LR / MAX_RECON_DEG                    训练规模（默认 3 / 4000 / 1e-3 / 1.0）
#
# 预计耗时（本机 16 线程 CPU）：从零到有约 3.5~4 小时；
#   若 factors_* 与 trained_fullrank 已存在（当前仓库状态），只补半秩训练 + 评测，约 2 小时。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=nmf/_env.sh
source "$REPO_ROOT/nmf/_env.sh"

MODEL="${MODEL:-esm2_t6_8M_UR50D}"
DATA_DIR="${DATA_DIR:-data/processed/swissprot}"
RAW_FILE="${RAW_FILE:-data/raw/swissprot_reviewed.fasta}"
OUTDIR="${OUTDIR:-nmf/outputs/fullmodel}"
BENCH_DIR="${BENCH_DIR:-$OUTDIR/bench_matrix}"
LOG_DIR="${LOG_DIR:-$OUTDIR/logs}"
TAGS="${TAGS:-fullrank halfrank}"

# 评测规模
N_EVAL="${N_EVAL:-64}"
MASK_ROUNDS="${MASK_ROUNDS:-5}"
PPL_RECORDS="${PPL_RECORDS:-8}"
PPL_MAX_LEN="${PPL_MAX_LEN:-256}"

# 训练规模（沿用已验证的配置：见 nmf/README.md §8）
EPOCHS="${EPOCHS:-3}"
MAX_RECORDS="${MAX_RECORDS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LR="${LR:-1e-3}"
KL_SCOPE="${KL_SCOPE:-all}"
DISTILL_W="${DISTILL_W:-1.0}"
RECON_W="${RECON_W:-0.1}"
MAX_RECON_DEG="${MAX_RECON_DEG:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-50}"

DRY_RUN="${DRY_RUN:-0}"
FORCE_DATA="${FORCE_DATA:-0}"
FORCE_FACTORIZE="${FORCE_FACTORIZE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_BENCH="${FORCE_BENCH:-0}"

mkdir -p "$OUTDIR" "$LOG_DIR"
nmf_check_device

# 每个 tag 的分解超参：满阶用 RANK_RATIO=1.0，半秩用 0.5
rank_ratio_of() { case "$1" in halfrank) echo "${HALFRANK_RANK_RATIO:-0.5}";; *) echo "${FULLRANK_RANK_RATIO:-1.0}";; esac; }
iters_of()      { case "$1" in halfrank) echo "${HALFRANK_ITERS:-400}";;      *) echo "${FULLRANK_ITERS:-600}";;      esac; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [DRY_RUN] %s\n' "$*"
  else
    "$@"
  fi
}

banner() {
  echo
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

banner "配置"
echo " model       : $MODEL"
echo " data        : $DATA_DIR    （已有 stats.json 则跳过数据准备）"
echo " 产物目录    : $OUTDIR"
echo " 评测目录    : $BENCH_DIR"
echo " tags        : $TAGS"
echo " 评测规模    : n-eval=$N_EVAL  mask-rounds=$MASK_ROUNDS  ppl=$PPL_RECORDS×$PPL_MAX_LEN"
echo " 训练规模    : epochs=$EPOCHS records=$MAX_RECORDS batch=$BATCH_SIZE lr=$LR trust-region=$MAX_RECON_DEG"
echo " device      : $DEVICE（分解用 $FACTORIZE_DEVICE）  TMPDIR=$TMPDIR"
[ "$DRY_RUN" = "1" ] && echo " >>> DRY_RUN 模式：只打印命令 <<<"

# --------------------------------------------------------------------------- #
banner "0/4 数据集准备（已有 $DATA_DIR/stats.json 就跳过）"
if [ -f "$DATA_DIR/stats.json" ] && [ "$FORCE_DATA" != "1" ]; then
  echo "已存在，跳过。"
else
  run "$PY" -u -m nmf.prepare_data --source swissprot --raw-file "$RAW_FILE" \
    --out-dir "$DATA_DIR" --min-len 50 --max-len 1022 \
    --train 30000 --valid 2000 --test 4000 --seed 0 \
    --decontaminate --decontam-threshold 0.6 --kmer 6
fi

# --------------------------------------------------------------------------- #
banner "1/4 分解（每个 tag 独立跳过）"
for TAG in $TAGS; do
  RANK_RATIO="$(rank_ratio_of "$TAG")"
  ITERS="$(iters_of "$TAG")"
  if [ -f "$OUTDIR/factors_${TAG}.pt" ] && [ "$FORCE_FACTORIZE" != "1" ]; then
    echo "[$TAG] factors_${TAG}.pt 已存在，跳过（rank-ratio=$RANK_RATIO, iters=$ITERS）。"
    continue
  fi
  echo "[$TAG] 开始分解：rank-ratio=$RANK_RATIO iters=$ITERS（CPU 约 20~30 分钟）"
  run "$PY" -u -m nmf.factorize --model "$MODEL" \
    --layers all --include-all-linear \
    --rank-ratio "$RANK_RATIO" \
    --solver hals --iters "$ITERS" --n-inner 20 \
    --shift min --init nndsvd \
    --device "$FACTORIZE_DEVICE" \
    --out    "$OUTDIR/factors_${TAG}.pt" \
    --report "$OUTDIR/factor_report_${TAG}.json" \
    2>&1 | tee "$LOG_DIR/factorize_${TAG}.log"
done

# --------------------------------------------------------------------------- #
banner "2/4 训练（只更新 A/S/B/offset；每个 tag 独立跳过）"
for TAG in $TAGS; do
  if [ -f "$OUTDIR/trained_${TAG}.pt" ] && [ "$FORCE_TRAIN" != "1" ]; then
    echo "[$TAG] trained_${TAG}.pt 已存在，跳过。"
    continue
  fi
  if [ ! -f "$OUTDIR/factors_${TAG}.pt" ]; then
    echo "[$TAG] 缺少 factors_${TAG}.pt，跳过训练。" >&2
    continue
  fi
  echo "[$TAG] 开始训练：$EPOCHS 轮 × $MAX_RECORDS 条（CPU 约 60~70 分钟）"
  run "$PY" -u -m nmf.train_nmf --model "$MODEL" \
    --layers all --include-all-linear \
    --init-checkpoint "$OUTDIR/factors_${TAG}.pt" \
    --fasta "$DATA_DIR/train.fasta" --max-records "$MAX_RECORDS" --seq-max-len 1022 \
    --batch-size "$BATCH_SIZE" --max-tokens-per-batch "$MAX_TOKENS" \
    --epochs "$EPOCHS" --max-batches 100000 \
    --lr "$LR" --lr-scale auto --kl-scope "$KL_SCOPE" \
    --distill-weight "$DISTILL_W" --recon-weight "$RECON_W" --ce-weight 0.0 \
    --max-recon-degradation "$MAX_RECON_DEG" --rollback-tries 3 --grad-clip 1.0 \
    --eval-every "$EVAL_EVERY" --eval-fasta "$DATA_DIR/valid.fasta" --eval-records 32 \
    --save-every "$SAVE_EVERY" --seed 0 --device "$DEVICE" \
    --out     "$OUTDIR/trained_${TAG}.pt" \
    --history "$OUTDIR/history_${TAG}.json" \
    --csv     "$OUTDIR/history_${TAG}.csv" \
    --plot \
    2>&1 | tee "$LOG_DIR/train_${TAG}.log"
done

# --------------------------------------------------------------------------- #
banner "3/4 论文级评测（所有变体写进同一目录；逐变体落盘，中断可续跑）"
CKPTS=()
for TAG in $TAGS; do
  # 检查点已含 stage 字段，这里按"未训练/训练后"给显示名
  case "$TAG" in
    fullrank) UNTRAINED_NAME="满阶未训练"; TRAINED_NAME="满阶训练后";;
    halfrank) UNTRAINED_NAME="半秩未训练"; TRAINED_NAME="半秩训练后";;
    *)        UNTRAINED_NAME="${TAG}未训练"; TRAINED_NAME="${TAG}训练后";;
  esac
  [ -f "$OUTDIR/factors_${TAG}.pt" ] && CKPTS+=("${UNTRAINED_NAME}=$OUTDIR/factors_${TAG}.pt")
  [ -f "$OUTDIR/trained_${TAG}.pt" ] && CKPTS+=("${TRAINED_NAME}=$OUTDIR/trained_${TAG}.pt")
done

if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "没有任何可评测的检查点，跳过评测。" >&2
else
  echo "待评测变体：${#CKPTS[@]} 个"
  printf '  - %s\n' "${CKPTS[@]}"
  if [ -f "$BENCH_DIR/benchmark.json" ] && [ "$FORCE_BENCH" != "1" ]; then
    echo "评测报告已存在；如需重算请加 FORCE_BENCH=1，或只重画表用 --from-dir。"
  fi
  # 注意：即使中途被中断，也不要丢掉已完成的部分 —— 用 --reuse-base --reuse-variants 续跑即可
  run "$PY" -u -m nmf.benchmark --model "$MODEL" \
    --checkpoint "${CKPTS[@]}" \
    --fasta "$DATA_DIR/test.fasta" --n-eval "$N_EVAL" --seq-max-len 1022 \
    --mask-frac 0.15 --mask-rounds "$MASK_ROUNDS" \
    --ppl-records "$PPL_RECORDS" --ppl-max-len "$PPL_MAX_LEN" --ppl-batch 32 \
    --seed 0 --device "$DEVICE" \
    --out-dir "$BENCH_DIR" --plot \
    2>&1 | tee "$LOG_DIR/benchmark.log"
  BENCH_RC=$?
  if [ "$BENCH_RC" != "0" ]; then
    echo
    echo "评测未完整跑完（退出码 $BENCH_RC）。两种续跑方式："
    echo "  1) 补跑缺的变体：把上面那条 nmf.benchmark 命令原样重跑，并加上 --reuse-base --reuse-variants"
    echo "  2) 先用已有结果出表：$PY -m nmf.benchmark --from-dir $BENCH_DIR --plot"
  fi
fi

# --------------------------------------------------------------------------- #
banner "4/4 指标汇总"
run "$PY" -u -m nmf.summarize \
  --root "$OUTDIR" --bench "$BENCH_DIR" --data "$DATA_DIR" \
  --out "$OUTDIR/SUMMARY.md"

banner "完成"
echo "主要产物："
echo "  $OUTDIR/factors_{fullrank,halfrank}.pt      # 分解结果（含 HALS 收敛历史）"
echo "  $OUTDIR/factor_report_{...}.json            # 逐层分解指标"
echo "  $OUTDIR/trained_{...}.pt                    # 训练后的分解结果"
echo "  $OUTDIR/history_{...}.json / .csv           # 训练历史（含分组学习率、回滚记录）"
echo "  $BENCH_DIR/benchmark.md                     # 论文级四张表"
echo "  $BENCH_DIR/benchmark.json / layer_stats.csv # 机器可读与逐层明细"
echo "  $BENCH_DIR/base.json + variants/            # 可复用的评测中间结果"
echo "  $OUTDIR/SUMMARY.md                          # ★ 全部指标汇总（人读）"
echo "  $OUTDIR/SUMMARY.json                        # ★ 全部指标汇总（机器可读）"
echo "  $OUTDIR/layer_metrics.csv                   # ★ 逐层分解明细"
echo "  $LOG_DIR/                                   # 各阶段日志"
