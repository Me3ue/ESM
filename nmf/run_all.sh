#!/usr/bin/env bash
# 一键跑通：三因子非负分解 -> 批次训练 -> 与原模型对比
#
#   bash nmf/run_all.sh                          # 用当前激活环境里的 python
#   PY=/path/to/python bash nmf/run_all.sh       # 指定解释器
#   DEVICE=cuda bash nmf/run_all.sh              # 用 GPU（auto 会自动判断）
#   RANK_RATIO=1.0 bash nmf/run_all.sh           # 满阶方阵（几乎无压缩）
#   LAYERS="layers.5.fc1,layers.5.self_attn.out_proj" bash nmf/run_all.sh
#
# 全部产物写入 nmf/outputs/。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=nmf/_env.sh
source "$REPO_ROOT/nmf/_env.sh"

MODEL="${MODEL:-esm2_t6_8M_UR50D}"
LAYERS="${LAYERS:-layers.5.fc1}"
RANK_RATIO="${RANK_RATIO:-0.5}"          # 1.0 = 满阶方阵；0.5 = 中间方阵降一半秩
SOLVER="${SOLVER:-hals}"                  # hals = 精确 NNLS（默认，精度最高）；pgd / mu 亦可
ITERS="${ITERS:-500}"
SHIFT="${SHIFT:-min}"                    # min = 整体平移（默认，实测更优）；row = 逐行平移
TRAIN_FASTA="${TRAIN_FASTA:-examples/data/P62593.fasta}"
EVAL_FASTA="${EVAL_FASTA:-examples/data/some_proteins.fasta}"
MAX_RECORDS="${MAX_RECORDS:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCHES="${MAX_BATCHES:-60}"
LR="${LR:-3e-3}"
OUTDIR="${OUTDIR:-nmf/outputs}"

mkdir -p "$OUTDIR"
nmf_check_device

echo "=============================================================="
echo " model        : $MODEL"
echo " layers       : $LAYERS  (rank-ratio=$RANK_RATIO, solver=$SOLVER, shift=$SHIFT)"
echo " train fasta  : $TRAIN_FASTA  (max-records=$MAX_RECORDS, batch=$BATCH_SIZE, batches=$MAX_BATCHES, lr=$LR)"
echo " eval  fasta  : $EVAL_FASTA"
echo " output dir   : $OUTDIR"
echo " device       : $DEVICE（分解用 $FACTORIZE_DEVICE）"
nmf_env_report
echo "=============================================================="

echo
echo "### 1/3 三因子非负分解（W ≈ A S B + offset，A/S/B >= 0，S 为方阵）"
"$PY" -m nmf.factorize \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --rank-ratio "$RANK_RATIO" \
  --solver "$SOLVER" --iters "$ITERS" --shift "$SHIFT" \
  --device "$FACTORIZE_DEVICE" \
  --out    "$OUTDIR/nmf_factors.pt" \
  --report "$OUTDIR/factorize_report.json" \
  --plot

echo
echo "### 2/3 批次训练（每步投影回非负象限，只训练 A/S/B）"
"$PY" -m nmf.train_nmf \
  --model "$MODEL" \
  --layers "$LAYERS" \
  --init-checkpoint "$OUTDIR/nmf_factors.pt" \
  --fasta "$TRAIN_FASTA" --max-records "$MAX_RECORDS" \
  --batch-size "$BATCH_SIZE" --max-batches "$MAX_BATCHES" \
  --lr "$LR" --temperature 2.0 --mask-frac 0.15 \
  --distill-weight 1.0 --recon-weight 0.5 \
  --max-recon-degradation 0.3 \
  --device "$DEVICE" \
  --eval-every 5 --eval-fasta "$EVAL_FASTA" --eval-records 8 \
  --out     "$OUTDIR/nmf_trained.pt" \
  --history "$OUTDIR/nmf_history.json" \
  --csv     "$OUTDIR/nmf_history.csv" \
  --plot

echo
echo "### 3/3 与原模型能力对比"
echo "--- (a) 同分布留出集：P62593 的另一批序列（不同随机种子）"
"$PY" -m nmf.evaluate \
  --model "$MODEL" \
  --checkpoint "$OUTDIR/nmf_factors.pt" "$OUTDIR/nmf_trained.pt" \
  --names 未训练 训练后 \
  --fasta "$TRAIN_FASTA" --max-records "${EVAL_N:-8}" --seed 12345 \
  --mask-frac 0.15 --mask-rounds 3 --device "$DEVICE" \
  --out-dir "$OUTDIR/eval_indist"

echo
echo "--- (b) 跨分布集：some_proteins（与训练集不同来源的多样蛋白）"
"$PY" -m nmf.evaluate \
  --model "$MODEL" \
  --checkpoint "$OUTDIR/nmf_factors.pt" "$OUTDIR/nmf_trained.pt" \
  --names 未训练 训练后 \
  --fasta "$EVAL_FASTA" --max-records 8 \
  --mask-frac 0.15 --mask-rounds 3 --device "$DEVICE" \
  --out-dir "$OUTDIR/eval_ood"

echo
echo "完成。主要产物："
echo "  $OUTDIR/nmf_factors.pt                  # 分解结果（未训练）"
echo "  $OUTDIR/nmf_factors_convergence.png     # 非负分解收敛曲线"
echo "  $OUTDIR/nmf_trained.pt                  # 训练后的分解结果"
echo "  $OUTDIR/nmf_history.json / .csv         # 训练历史"
echo "  $OUTDIR/nmf_trained_training.png        # 训练曲线"
echo "  $OUTDIR/eval_indist/report.md           # 同分布留出集对比报告"
echo "  $OUTDIR/eval_ood/report.md              # 跨分布集对比报告"
echo "  $OUTDIR/*/report.json / layer_metrics.csv"
