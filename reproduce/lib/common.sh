#!/usr/bin/env bash
# 复现工具包公共库：环境、日志、dry-run、断点续跑。
# 用法：在其它脚本里 `source "$(dirname "$0")/lib/common.sh"`
#
# 可用环境变量：
#   PY        解释器路径（默认本机 conda ESM 环境）
#   MODE      smoke | full      （默认 smoke，避免误触发超算级任务）
#   DRY_RUN   1 只打印不执行
#   OUT_ROOT  产物根目录（默认 <repo>/reproduce/outputs）
#   DEVICE    cuda:0 | cpu      （默认自动探测）

set -euo pipefail

# ---------------------------------------------------------------- 路径
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$COMMON_DIR/.." && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"

# ---------------------------------------------------------------- 运行环境
# 本机 /tmp 是 10MB tmpfs，任何下载/编译前必须改 TMPDIR
export TMPDIR="${TMPDIR:-/home/zzj/tmp}"
mkdir -p "$TMPDIR"

PY="${PY:-/home/zzj/anaconda3/envs/ESM/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
OUT_ROOT="${OUT_ROOT:-$REPRO_ROOT/outputs}"

# 运行仓库内脚本时清空 PYTHONPATH，避免宿主 shim 目录干扰
export PYTHONPATH=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/mplconfig}"

# 探测设备
if [[ -z "${DEVICE:-}" ]]; then
  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi
export DEVICE

# ---------------------------------------------------------------- 日志
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BLUE=$'\033[34m'; C_YEL=$'\033[33m'
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_DIM=$'\033[2m'
else
  C_RESET=""; C_BLUE=""; C_YEL=""; C_RED=""; C_GRN=""; C_DIM=""
fi

log()   { printf '%s[%s]%s %s\n' "$C_BLUE" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
warn()  { printf '%s[%s] 警告%s %s\n' "$C_YEL" "$(date +%H:%M:%S)" "$C_RESET" "$*" >&2; }
err()   { printf '%s[%s] 错误%s %s\n' "$C_RED" "$(date +%H:%M:%S)" "$C_RESET" "$*" >&2; }
ok()    { printf '%s[%s] 完成%s %s\n' "$C_GRN" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
dim()   { printf '%s%s%s\n' "$C_DIM" "$*" "$C_RESET"; }

banner() {
  printf '\n%s%s%s\n' "$C_BLUE" "════════════════════════════════════════════════════════════════" "$C_RESET"
  printf '%s  %s%s\n' "$C_BLUE" "$*" "$C_RESET"
  printf '%s%s%s\n\n' "$C_BLUE" "════════════════════════════════════════════════════════════════" "$C_RESET"
}

# 在真正执行前统一确认；DRY_RUN=1 时只回显
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    dim "  [dry-run] $*"
    return 0
  fi
  dim "  \$ $*"
  "$@"
}

# 记时执行，返回真实退出码（set -e 下失败即中断，便于定位）
run_timed() {
  local label="$1"; shift
  local t0=$SECONDS
  if [[ "$DRY_RUN" == "1" ]]; then dim "  [dry-run] $*"; return 0; fi
  dim "  \$ $*"
  if "$@"; then
    log "$label 用时 $((SECONDS - t0))s"
  else
    err "$label 失败（用时 $((SECONDS - t0))s）"; return 1
  fi
}

# ---------------------------------------------------------------- 前置检查
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "缺少命令：$1（$2）"; return 1; }
}

python_has() {
  "$PY" - "$1" <<'EOF' >/dev/null 2>&1
import importlib, sys
importlib.import_module(sys.argv[1])
EOF
}

check_python_deps() {
  local missing=()
  for m in "$@"; do
    python_has "$m" || missing+=("$m")
  done
  if ((${#missing[@]})); then
    warn "缺少 Python 模块：${missing[*]}"
    warn "请先运行：bash $REPRO_ROOT/00_setup_env.sh"
    return 1
  fi
  return 0
}

gpu_hint() {
  if [[ "$DEVICE" == "cpu" ]]; then
    warn "当前 DEVICE=cpu。ESMFold/AlphaFold 类实验在 CPU 上比 GPU 慢 1~2 个数量级，"
    warn "论文正文规模的优化循环（3 万步 × 数百 seed）在 CPU 上实际不可完成。"
    warn "建议：先跑 MODE=smoke 打通流程，再上 GPU（见 README 第 1.3 节）。"
  fi
}

# 小工具：消解路径
abs_path() { "$PY" -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
