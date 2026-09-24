#!/usr/bin/env bash
# 复现工具包公共库：环境、日志、dry-run、硬件自适应、断点续跑。
# 用法：在其它脚本里 `source "$(dirname "$0")/lib/common.sh"`
#
# 可用环境变量（命令行传的永远优先于 machine.env）：
#   PY        解释器路径（默认自动找：machine.env → conda 环境 → PATH 上的 python3）
#   MODE      smoke | full      （默认 smoke，避免误触发超算级任务）
#   DRY_RUN   1 只打印不执行
#   OUT_ROOT  产物根目录（默认 <repo>/reproduce/outputs）
#   DEVICE    cuda:N | cuda | cpu  （默认自动挑最空闲的卡）
#   GPUS      "0,2,4,5" 指定参与多卡并行的卡
#   TMPDIR    临时目录（默认自动挑 /dev/shm 或仓库内目录）
#
# 机器档案：若存在 reproduce/machine.env 会自动 source（由 00_setup_env.sh hardware --write 生成）

set -euo pipefail

# 防止重复 source
if [[ -n "${_REPRO_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_REPRO_COMMON_LOADED=1

# ---------------------------------------------------------------- 路径
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$COMMON_DIR/.." && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"
export REPRO_ROOT REPO_ROOT

# ---------------------------------------------------------------- 日志（先定义，后面都要用）
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

# ---------------------------------------------------------------- 机器档案
# 先生成一份「最小可用」的环境，再让 machine.env 覆盖（machine.env 里全是 ${VAR:-默认} 形式，
# 所以外部显式传进来的环境变量优先级最高）
export PYTHONPATH=""                       # 运行仓库内脚本时清空，避免宿主 shim 干扰
if [[ -f "$REPRO_ROOT/machine.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPRO_ROOT/machine.env"
fi

# ---------------------------------------------------------------- 硬件
# shellcheck disable=SC1091
source "$COMMON_DIR/hardware.sh"

# TMPDIR：优先显式指定 → 上一轮探测结果 → 自动挑
if [[ -z "${TMPDIR:-}" || ! -w "${TMPDIR:-/nonexistent}" ]]; then
  TMPDIR="$(hw_pick_tmpdir)"
fi
export TMPDIR
mkdir -p "$TMPDIR" 2>/dev/null || true
HW_TMPDIR="$TMPDIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$TMPDIR/mplconfig}"
# A6000 等大卡上能明显减少显存碎片导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------- 解释器
if [[ -z "${PY:-}" || ! -x "${PY:-}" ]]; then
  for cand in \
      "$REPO_ROOT/.venv/bin/python" \
      "/home/zzj/anaconda3/envs/ESM/bin/python" \
      "${CONDA_PREFIX:-}/bin/python" \
      "$(command -v python3 2>/dev/null || true)" \
      "/usr/bin/python3"; do
    if [[ -n "$cand" && -x "$cand" ]]; then PY="$cand"; break; fi
  done
fi
if [[ -z "${PY:-}" ]]; then
  err "找不到可用的 python 解释器，请在 machine.env 里设置 PY，或 export PY=/path/to/python"
  exit 3
fi
export PY

# ---------------------------------------------------------------- 设备
# 判定顺序：DEVICE 显式指定 > torch 能看到 CUDA（权威）且在 nvidia-smi 里有空闲卡 → 挑最空的卡 > cpu
# 「torch 看不到 CUDA」是常见坑：驱动在、nvidia-smi 正常，但装的是 CPU 版 torch。
if [[ -z "${DEVICE:-}" ]]; then
  hw_probe
  if hw_torch_cuda && [[ "${HW_SMI_OK:-0}" == "1" ]] && [[ "${HW_GPU_COUNT:-0}" -gt 0 ]]; then
    DEVICE="$(hw_pick_device)"
  else
    DEVICE="cpu"
  fi
fi
export DEVICE

MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
OUT_ROOT="${OUT_ROOT:-$REPRO_ROOT/outputs}"
export MODE DRY_RUN OUT_ROOT

# 数据/权重目录（machine.env 可能已设）
: "${DATA_ROOT:=$REPO_ROOT/data}"
: "${TORCH_HOME:=$DATA_ROOT/weights/torch}"
export DATA_ROOT TORCH_HOME

# /dev/shm 是内存盘：跑得快但**重启就清空**。临时文件放它没问题，产物/数据不能放。
if [[ "${OUT_ROOT:-}" == /dev/shm/* || "${DATA_ROOT:-}" == /dev/shm/* ]]; then
  warn "OUT_ROOT/DATA_ROOT 位于 /dev/shm —— 内存盘重启即清空，长时间实验的产物会丢！"
  warn "  请改成磁盘路径（export OUT_ROOT=/your/disk/repro，或编辑 reproduce/machine.env）"
fi
if [[ "${TMPDIR:-}" == /dev/shm/* ]]; then
  dim " 提示：TMPDIR=$TMPDIR 在内存盘（快，但占内存；下载/解压大库请改用磁盘）"
fi

# ---------------------------------------------------------------- 执行助手
run() {
  if [[ "$DRY_RUN" == "1" ]]; then dim "  [dry-run] $*"; return 0; fi
  dim "  \$ $*"
  "$@"
}

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
  for m in "$@"; do python_has "$m" || missing+=("$m"); done
  if ((${#missing[@]})); then
    warn "缺少 Python 模块：${missing[*]}"
    warn "请先运行：bash $REPRO_ROOT/00_setup_env.sh base（当前解释器 $PY）"
    return 1
  fi
  return 0
}

# 依赖体检：GPU 环境相关
gpu_hint() {
  hw_probe
  if [[ "$DEVICE" == "cpu" ]]; then
    warn "当前 DEVICE=cpu。论文正文规模的优化循环（3 万步 × 数百 seed）在 CPU 上不可完成。"
    warn "请确认：① 装了 CUDA 版 torch（见 00_setup_env.sh）；② machine.env 里的 PY 指向那个环境。"
  elif [[ "$HW_GPU_COUNT" -gt 0 ]]; then
    hw_auto_tune
    dim " 设备 $DEVICE；GPU $HW_GPU_COUNT 张；空闲卡 $(hw_free_gpus || echo 无)；worker 数 $N_WORKERS"
  fi
}

# 小工具
abs_path() { "$PY" -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }

# 断言解释器里有 CUDA 版 torch，否则早失败并给出可执行建议
require_cuda_torch() {
  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    return 0
  fi
  err "当前解释器（$PY）的 torch 看不到 CUDA。"
  err "  查看：$PY -c \"import torch;print(torch.__version__, torch.version.cuda)\""
  err "  修复：$PY -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
  err "  若集群有 module 系统：module load cuda/12.8 后再装，注意 nvcc 版本要匹配。"
  return 1
}

# ---------------------------------------------------------------- 多卡调度
# 放在最后 source：run_on_gpus 依赖上面的 log/run_timed 等函数
# shellcheck disable=SC1091
source "$COMMON_DIR/multigpu.sh"
