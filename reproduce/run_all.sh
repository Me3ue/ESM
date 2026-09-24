#!/usr/bin/env bash
# ============================================================================
#  两篇论文复现的顶层一键入口
#
#  用法：
#     MODE=smoke bash run_all.sh          # 【推荐先跑】极小规模打通全流程
#     MODE=full  bash run_all.sh          # 论文正文规模（需要 GPU + 数十小时~数天）
#     DRY_RUN=1  bash run_all.sh          # 只打印计划
#     ONLY=p1 bash run_all.sh             # 只跑论文一（p1 / p2 / paperdata）
#
#  环境变量（透传给子脚本）：
#     MODE, DRY_RUN, DEVICE, OUT_ROOT, PY, WEIGHTS, SEEDS, STEPS, NUM_ITER,
#     TASKS, PDBS, ALL_TARGETS, ORACLE, NOVELTY_DB, WITH_ROUNDTRIP, SKIP_AGG
#
#  前置：
#     bash 00_setup_env.sh check          # 看清缺什么
#     bash 00_setup_env.sh base           # 装 CPU 可跑的依赖
#     bash 00_setup_env.sh esmfold        # 打 ESMFold（论文一要真的跑起来必须有）
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

MODE="${MODE:-smoke}"
ONLY="${ONLY:-all}"
T0=$SECONDS

banner "两篇论文复现  MODE=$MODE  ONLY=$ONLY"
log "解释器：$PY"
log "设备  ：$DEVICE"
log "产物  ：$OUT_ROOT"
dim  "论文一：A high-level programming language for generative protein design"
dim  "论文二：Language models generalize beyond natural proteins"

# ---------------------------------------------------------------- 0) 体检
banner "阶段 0：环境体检"
run "$PY" "$HERE/aggregate_all.py" --help >/dev/null 2>&1 || true
python_has torch || warn "torch 缺失"
python_has esm   || warn "esm 缺失（本仓库需要 pip install -e .）"
python_has openfold || warn "openfold 缺失 → 论文一的 ESMFold 设计实验无法运行"
python_has hydra    || warn "hydra 缺失 → 论文二的 lm-design 无法运行"
dim "  （论文二公开数据复算不需要上面任何模型，永远可跑。）"

# ---------------------------------------------------------------- 0.5) 数据集
if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  banner "阶段 0.5：数据集（PDB + 权重 + 论文数据包，≈ 8.8 GB）"
  dim "  清单与说明见 DATASETS.md；大库（UniRef90 / AF DB / PDB 快照）不在此阶段下载。"
  bash "$HERE/fetch_datasets.sh" core || warn "数据集准备未完成（可重跑 fetch_datasets.sh core）"
  dim "  想跳过就设 SKIP_DATA=1。要下 UniRef90 等大库：bash fetch_datasets.sh seqdb"
else
  log "跳过数据集准备（SKIP_DATA=1）"
fi

# ---------------------------------------------------------------- 1) 论文一
if [[ "$ONLY" == "all" || "$ONLY" == "p1" ]]; then
  banner "阶段 1：论文一（设计实验 + roundtrip + 汇总）"
  bash "$HERE/paper1_programming/run_p1.sh" || warn "论文一有未完成项（可重跑续跑）"
fi

# ---------------------------------------------------------------- 2) 论文二
if [[ "$ONLY" == "all" || "$ONLY" == "p2" ]]; then
  banner "阶段 2：论文二（设计实验 + 新颖性 + 汇总）"
  bash "$HERE/paper2_lm_design/run_p2.sh" || warn "论文二有未完成项（可重跑续跑）"
fi

# ---------------------------------------------------------------- 3) 公开数据复算
if [[ "$ONLY" == "all" || "$ONLY" == "paperdata" ]]; then
  banner "阶段 3：论文二公开实验数据复算（零成本，不加载模型）"
  run "$PY" "$HERE/paper2_lm_design/paper_data_report.py" \
    --out "$OUT_ROOT/p2/paper_data_report"
fi

# ---------------------------------------------------------------- 4) 总汇总
banner "阶段 4：总汇总"
run "$PY" "$HERE/aggregate_all.py" --root "$OUT_ROOT"

ELAPSED=$((SECONDS - T0))
banner "全部完成（用时 ${ELAPSED}s）"
cat <<EOT
总报告：$OUT_ROOT/REPRO_SUMMARY.md

论文一：$OUT_ROOT/p1/summary/P1_SUMMARY.md
论文二：$OUT_ROOT/p2/summary/P2_SUMMARY.md
公开数据复算：$OUT_ROOT/p2/paper_data_report/PAPER_DATA_REPORT.md

提醒：
  * MODE=smoke 只是「流程跑通」，数值没有统计意义；
    论文正文的结论需要 MODE=full + GPU + 长时间运行。
  * 论文里「湿实验 / AlphaFold oracle / ColabDesign 基线 / Rosetta 过滤 /
    TM-align 结构新颖性」这几块需要额外外部资源，见各报告最后一节。
EOT
