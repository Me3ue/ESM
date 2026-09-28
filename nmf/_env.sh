#!/usr/bin/env bash
# 公共环境准备：被 run_all.sh / run_full_experiment.sh / run_matrix.sh 共同 source。
#
# 负责三件事，让脚本能在任意机器（含 3090 服务器）上直接跑：
#   1. 解析解释器 PY（$PY > 当前 PATH 里的 python，找不到就报错并给提示）；
#   2. 确定计算设备 DEVICE（cpu / cuda / auto），默认 auto；
#   3. 设置可用的 TMPDIR（/tmp 可能是小容量 tmpfs，默认放仓库内的 .tmp）。
#
# 调用方需先定义 REPO_ROOT 并 cd 到仓库根目录。

# ---------------------------------------------------------------- 解释器 ----
if [ -z "${PY:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  fi
fi
if [ -z "${PY:-}" ] || [ ! -x "$PY" ]; then
  cat >&2 <<'MSG'
[错误] 找不到 python 解释器。
       先激活环境：  conda activate <你的环境名>
       或显式指定：  PY=/path/to/python bash nmf/run_xxx.sh
MSG
  exit 1
fi

# ------------------------------------------------------------------ 设备 ----
DEVICE="${DEVICE:-auto}"          # cpu / cuda / auto
# 分解阶段可单独指定（HALS 含 Python 循环，GPU 上未必更快；默认跟随 DEVICE）
FACTORIZE_DEVICE="${FACTORIZE_DEVICE:-$DEVICE}"

# ---------------------------------------------------------------- TMPDIR ----
if [ -z "${TMPDIR:-}" ]; then
  TMPDIR="$REPO_ROOT/.tmp"
fi
mkdir -p "$TMPDIR" 2>/dev/null || TMPDIR="${TMPDIR:-/tmp}"
export TMPDIR

# 不要继承宿主的 PYTHONPATH（某些环境会塞入不兼容的模块）
unset PYTHONPATH || true
export PYTHONUNBUFFERED=1

# ------------------------------------------------------- 环境自检（打印）----
nmf_env_report() {
  "$PY" - <<'PYEOF'
import sys
try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"  torch        : 不可用（{exc}）")
    raise SystemExit(0)
line = f"  python       : {sys.executable} (Python {sys.version.split()[0]})"
print(line)
print(f"  torch        : {torch.__version__}")
print(f"  cuda 可用    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}"
          f"（{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}，"
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB）")
else:
    print("  GPU          : 未检测到（DEVICE=cuda 会被自动回退到 cpu）")
PYEOF
}

# 若 DEVICE 是 cuda 但实际不可用，直接提示（运行时会自动回退到 cpu，但要让用户知道）
nmf_check_device() {
  case "$DEVICE" in
    cuda*)
      if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "[警告] DEVICE=$DEVICE 但 torch.cuda.is_available() 为 False —— 将回退到 CPU。" >&2
        echo "       常见原因：装的是 CPU 版 torch（用 pip install torch --index-url .../cu121 重装），" >&2
        echo "       或驱动/CUDA 运行时不匹配。" >&2
        DEVICE="cpu"
        FACTORIZE_DEVICE="cpu"
      fi
      ;;
  esac
}
