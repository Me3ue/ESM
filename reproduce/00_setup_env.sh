#!/usr/bin/env bash
# ============================================================================
#  复现两篇论文的环境准备（不自动跑实验）
#
#  用法：
#     bash 00_setup_env.sh check      # 只做体检，打印还缺什么
#     bash 00_setup_env.sh base       # 装分析/设计所需的纯 Python 依赖（CPU 可用）
#     bash 00_setup_env.sh esmfold    # 装 ESMFold + OpenFold（需要 GPU + CUDA 工具链）
#     bash 00_setup_env.sh extern     # 打印外部命令行工具（hmmer/tm-align/foldseek）装法
#     bash 00_setup_env.sh all        # base + esmfold + extern
#
#  环境变量：
#     PY=<python 路径>     默认本机 conda ESM 环境
#     DRY_RUN=1            只打印命令
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

PY="${PY:-/home/zzj/anaconda3/envs/ESM/bin/python}"

banner "环境准备：$(basename "$REPO_ROOT")"

# ---------------------------------------------------------------- check
do_check() {
  log "解释器：$PY"
  "$PY" -V || true
  log "设备：DEVICE=$DEVICE"
  if [[ "$DEVICE" == "cpu" ]]; then
    warn "torch 是 CPU 版（或 CUDA 不可用）。论文正文规模必须用 GPU。"
    dim "  改 GPU：$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu124"
  fi
  echo
  log "Python 模块体检："
  for m in torch esm biotite rich hydra omegaconf nltk pandas scipy matplotlib openfold; do
    if python_has "$m"; then
      printf '  %s✓%s %-12s\n' "$C_GRN" "$C_RESET" "$m"
    else
      printf '  %s✗%s %-12s\n' "$C_RED" "$C_RESET" "$m"
    fi
  done
  echo
  log "外部命令行工具体检："
  for c in jackhmmer tm-align foldseek aria2c wget nvcc; do
    if command -v "$c" >/dev/null 2>&1; then
      printf '  %s✓%s %-12s\n' "$C_GRN" "$C_RESET" "$c"
    else
      printf '  %s✗%s %-12s\n' "$C_RED" "$C_RESET" "$c"
    fi
  done
  echo
  log "缓存目录（权重下载位置）："
  dim "  torch hub: ${TORCH_HOME:-${HOME:-~}/.cache/torch/hub/checkpoints}"
  dim "  TMPDIR   : $TMPDIR   （本机 /tmp 只有 10MB，务必保持这个设置）"
}

# ---------------------------------------------------------------- base
do_base() {
  log "安装纯 Python 依赖（biotite/rich/hydra/omegaconf/nltk/pandas/scipy）"
  run "$PY" -m pip install -r "$HERE/requirements-repro.txt"
  ok "base 依赖装好"
}

# ---------------------------------------------------------------- esmfold
do_esmfold() {
  banner "安装 ESMFold / OpenFold（重、需要 GPU + nvcc）"
  cat <<'EOT'
官方推荐做法（OpenFold 对 Python>=3.10 不友好，建议单独建 Python 3.9 环境）：

  # 1) 建一个专用环境（cudatoolkit 版本要和驱动匹配）
  conda create -y -n esmfold python=3.9
  conda activate esmfold
  export TMPDIR=/home/zzj/tmp

  # 2) 装 CUDA 版 torch（按你的驱动选 cu121/cu124）
  pip install torch --index-url https://download.pytorch.org/whl/cu124

  # 3) 装本仓库 + ESMFold 附加依赖
  cd <repo> && pip install -e .
  pip install "fair-esm[esmfold]"
  pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
  pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

  # 4) 验证
  python -c "import esm; m=esm.pretrained.esmfold_v1().eval().cuda(); print(m.infer_pdb('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ')[:200])"

注意：
  * 本机已有 NVIDIA GeForce RTX 5070 Ti Laptop（sm_120），需要较新的 CUDA/torch；
    OpenFold 的 CUDA kernel 对 sm_120 不一定有预编译支持，若失败可退回
    `ESMFOLD_USE_CPU=1`（极慢）或用 ESM Atlas 官方 API（见 README 1.4 节）。
  * 不要装在现有 conda 环境 ESM 里——那是 CPU 版 torch，重装会破坏已跑通的环境。
EOT
  ok "打印完毕（脚本不代为执行，请按上面命令操作）"
}

# ---------------------------------------------------------------- extern
do_extern() {
  banner "外部命令行工具"
  cat <<'EOT'
  # jackhmmer 3.3.2（论文二序列新颖性检索）
  conda install -y -c bioconda hmmer=3.3.2

  # tm-align >= 20210107（论文一结构新颖性 TM-score）
  #   官方：https://zhanggroup.org/TM-align/  （下载后 chmod +x，放进 PATH）
  #   国内镜像可搜 tm-align static binary

  # foldseek v7d0c07f89a（论文二 App A.5.4 结构检索对照）
  conda install -y -c bioconda foldseek

  # 下载器
  conda install -y -c conda-forge aria2 wget

  # nvcc（仅当要自己编 OpenFold CUDA kernel）
  #   随 cudatoolkit 一起装：conda install -c nvidia cuda-toolkit
EOT
  ok "打印完毕"
}

# ---------------------------------------------------------------- main
case "${1:-check}" in
  check)   do_check ;;
  base)    do_base ;;
  esmfold) do_esmfold ;;
  extern)  do_extern ;;
  all)     do_base; do_esmfold; do_extern; do_check ;;
  *)       err "未知子命令：$1（可用：check|base|esmfold|extern|all）"; exit 2 ;;
esac
