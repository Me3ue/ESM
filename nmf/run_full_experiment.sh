#!/usr/bin/env bash
# 论文级完整实验：数据准备 -> 全模型非负分解 -> 批次训练 -> 严格评测
#
#   bash nmf/run_full_experiment.sh
#   DEVICE=cuda bash nmf/run_full_experiment.sh        # GPU（需要 CUDA 版 torch）
#   TAG=halfrank RANK_RATIO=0.5 bash nmf/run_full_experiment.sh
#
# 可用环境变量覆盖（详见下方默认值）。产物写入 nmf/outputs/fullmodel/。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=nmf/_env.sh
source "$REPO_ROOT/nmf/_env.sh"

MODEL="${MODEL:-esm2_t6_8M_UR50D}"
DATA_DIR="${DATA_DIR:-data/processed/swissprot}"
RAW_FILE="${RAW_FILE:-data/raw/swissprot_reviewed.fasta}"
OUTDIR="${OUTDIR:-nmf/outputs/fullmodel}"

# 数据规模（Swiss-Prot reviewed 随机切分 + k-mer 去污染）
TRAIN_N="${TRAIN_N:-30000}"
VALID_N="${VALID_N:-2000}"
TEST_N="${TEST_N:-4000}"

# 分解：hals = 逐列/逐行精确 NNLS，满阶时相对重构误差可达 1e-3 量级
RANK_RATIO="${RANK_RATIO:-1.0}"
ITERS="${ITERS:-600}"
N_INNER="${N_INNER:-20}"
SHIFT="${SHIFT:-min}"
TAG="${TAG:-fullrank}"

# 训练：全模型 38 个线性层、只更新 A/S/B/offset
EPOCHS="${EPOCHS:-2}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
KL_SCOPE="${KL_SCOPE:-all}"
DISTILL_W="${DISTILL_W:-1.0}"
RECON_W="${RECON_W:-0.5}"
MAX_RECON_DEG="${MAX_RECON_DEG:-0.3}"

mkdir -p "$OUTDIR"

nmf_check_device
echo "=============================================================="
echo " model       : $MODEL"
echo " data        : $DATA_DIR (train=$TRAIN_N valid=$VALID_N test=$TEST_N)"
echo " 分解参数    : rank-ratio=$RANK_RATIO solver=hals iters=$ITERS shift=$SHIFT"
echo " 训练参数    : epochs=$EPOCHS max-tokens/batch=$MAX_TOKENS lr=$LR kl-scope=$KL_SCOPE"
echo "               distill=$DISTILL_W recon=$RECON_W trust-region=$MAX_RECON_DEG"
echo " 输出        : $OUTDIR  (tag=$TAG)"
echo " device      : $DEVICE（分解用 $FACTORIZE_DEVICE）"
nmf_env_report
echo "=============================================================="

echo
echo "### 0/4 数据集准备（下载 + 质控 + 去污染 + 切分）"
"$PY" -m nmf.prepare_data --source swissprot --raw-file "$RAW_FILE" \
  --out-dir "$DATA_DIR" --min-len 50 --max-len 1022 \
  --train "$TRAIN_N" --valid "$VALID_N" --test "$TEST_N" --seed 0 \
  --decontaminate --decontam-threshold 0.6 --kmer 6

echo
echo "### 1/4 全模型非负分解（所有 38 个线性层）"
"$PY" -u -m nmf.factorize --model "$MODEL" \
  --layers all --include-all-linear \
  --rank-ratio "$RANK_RATIO" \
  --solver hals --iters "$ITERS" --n-inner "$N_INNER" \
  --shift "$SHIFT" --init nndsvd \
  --device "$FACTORIZE_DEVICE" \
  --out    "$OUTDIR/factors_${TAG}.pt" \
  --report "$OUTDIR/factor_report_${TAG}.json"

echo
echo "### 2/4 批次训练（冻结其余参数，只训练 A/S/B/offset，信任域保护）"
"$PY" -u -m nmf.train_nmf --model "$MODEL" \
  --layers all --include-all-linear \
  --init-checkpoint "$OUTDIR/factors_${TAG}.pt" \
  --fasta "$DATA_DIR/train.fasta" --max-records "$TRAIN_N" --seq-max-len 1022 \
  --batch-size "$BATCH_SIZE" --max-tokens-per-batch "$MAX_TOKENS" \
  --epochs "$EPOCHS" --max-batches 100000 \
  --lr "$LR" --kl-scope "$KL_SCOPE" \
  --distill-weight "$DISTILL_W" --recon-weight "$RECON_W" --ce-weight 0.0 \
  --max-recon-degradation "$MAX_RECON_DEG" --rollback-tries 3 \
  --device "$DEVICE" \
  --eval-every 25 --eval-fasta "$DATA_DIR/valid.fasta" --eval-records 32 \
  --out     "$OUTDIR/trained_${TAG}.pt" \
  --history "$OUTDIR/history_${TAG}.json" \
  --csv     "$OUTDIR/history_${TAG}.csv" \
  --plot

echo
echo "### 3/4 严格评测（留出测试集；逐变体落盘，中断可续跑）"
"$PY" -u -m nmf.benchmark --model "$MODEL" \
  --checkpoint "未训练=$OUTDIR/factors_${TAG}.pt" "训练后=$OUTDIR/trained_${TAG}.pt" \
  --fasta "$DATA_DIR/test.fasta" --n-eval 128 \
  --mask-frac 0.15 --mask-rounds 5 \
  --ppl-records 12 --ppl-max-len 256 --ppl-batch 32 \
  --device "$DEVICE" \
  --out-dir "$OUTDIR/bench_${TAG}" --plot

echo
echo "### 4/4 后续用法提示"
echo "  - 补跑被中断的变体：同样的 --out-dir 上加 --reuse-base --reuse-variants"
echo "  - 报告丢了只想重画表：python -m nmf.benchmark --from-dir $OUTDIR/bench_${TAG} --plot"
echo "  - 半秩（压缩）对照：RANK_RATIO=0.5 TAG=halfrank bash nmf/run_full_experiment.sh"
echo "    再把两个 TAG 的检查点一起传给同一条 --checkpoint（沿用同一 --out-dir 即合成一张表）"

echo
echo "完成。主要产物："
echo "  $OUTDIR/factors_${TAG}.pt               # 全模型分解（未训练）"
echo "  $OUTDIR/trained_${TAG}.pt               # 训练后"
echo "  $OUTDIR/history_${TAG}.csv              # 训练历史"
echo "  $OUTDIR/bench_${TAG}/benchmark.md       # 论文级对比报告"
echo "  $OUTDIR/bench_${TAG}/benchmark.json     # 机器可读结果"
echo "  $OUTDIR/bench_${TAG}/layer_stats.csv    # 逐层重构指标"
echo "  $OUTDIR/bench_${TAG}/base.json          # 原模型侧指标 + 评测口径指纹"
echo "  $OUTDIR/bench_${TAG}/variants/          # 逐变体中间结果（可复用）"
