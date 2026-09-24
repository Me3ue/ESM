#!/usr/bin/env bash
# ============================================================================
#  复现两篇论文的环境准备（不自动跑实验）
#
#  用法：
#     bash 00_setup_env.sh hardware         # 【先跑这个】探测本机硬件并打印报告
#     bash 00_setup_env.sh hardware --write # 探测并写 reproduce/machine.env（后续脚本自动加载）
#     bash 00_setup_env.sh check            # 依赖体检
#     bash 00_setup_env.sh base             # 装分析/设计所需的纯 Python 依赖
#     bash 00_setup_env.sh cuda             # 装 CUDA 版 torch（按驱动自动选 cu12x）
#     bash 00_setup_env.sh esmfold          # 装 ESMFold + OpenFold（需要 GPU + CUDA 工具链）
#     bash 00_setup_env.sh extern           # 打印外部命令行工具（hmmer/tm-align/foldseek）装法
#     bash 00_setup_env.sh all              # base + cuda + esmfold + extern
#
#  环境变量：
#     PY=<python 路径>     默认由 lib/common.sh 自动探测（可用 machine.env 固定）
#     DRY_RUN=1            只打印命令
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

banner "环境准备：$(basename "$REPO_ROOT")"

# ---------------------------------------------------------------- hardware
do_hardware() {
  hw_report
  cat <<EOT

 按本机硬件自动给出的建议：
   * 解释器      : $PY
   * 并行策略    : worker 数 = 空闲卡数（当前 N_WORKERS=$N_WORKERS）
   * ESMFold 分块: $ESMFOLD_CHUNK
   * jackhmmer   : $JACKHMMER_CPU 线程
   * TMPDIR      : $TMPDIR
   * 产物目录    : $OUT_ROOT
   * 数据目录    : $DATA_ROOT
   * 权重缓存    : $TORCH_HOME
EOT
  if [[ "${1:-}" == "--write" ]]; then
    local f
    f="$(hw_write_profile "$REPRO_ROOT/machine.env")"
    ok "机器档案已写入：$f"
    dim "  后续所有脚本会自动 source 它；里面全是 \${VAR:-默认} 形式，命令行显式传的值仍然优先。"
    dim "  要换设备/调并行数，直接编辑该文件即可。"
  else
    dim "  想固化成档案（后续脚本自动加载）：bash $0 hardware --write"
  fi
}

# ---------------------------------------------------------------- check
do_check() {
  log "解释器：$PY"
  "$PY" -V || true
  log "设备：DEVICE=$DEVICE"
  if [[ "$DEVICE" == "cpu" ]]; then
    warn "当前解释器的 torch 看不到 CUDA。若本机确实有 GPU："
    dim "  查看：$PY -c \"import torch;print(torch.__version__, torch.version.cuda)\""
    dim "  修复：bash $0 cuda"
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
  echo
  hw_report
}

# ---------------------------------------------------------------- cuda
do_cuda() {
  local drv cu idx
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  cu="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1 || true)"
  log "驱动版本：${drv:-未知}   驱动支持的 CUDA：${cu:-未知}"

  idx=cu124
  case "${cu:-12.0}" in
    12.8*) idx=cu128 ;; 12.6*) idx=cu126 ;; 12.4*) idx=cu124 ;;
    12.1*) idx=cu121 ;; 11.8*) idx=cu118 ;;
  esac
  log "给「论文二」环境装 CUDA 版 torch（index-url 后缀 $idx）"
  run "$PY" -m pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${idx}"
  echo
  log "验证："
  run "$PY" -c "import torch;print('torch',torch.__version__,'| cuda',torch.version.cuda,'| 可用',torch.cuda.is_available(),'| 卡数',torch.cuda.device_count())"
  if hw_torch_cuda; then
    ok "torch 已能看到 CUDA。建议接着跑：bash $0 hardware --write"
  else
    err "装完仍不可用：检查 nvcc 与驱动是否匹配，或用 module load 切 CUDA 版本后重装。"
    warn "另外：论文一（ESMFold）建议单独建 python3.9 环境，别和这个混装。"
    return 1
  fi
}

# ---------------------------------------------------------------- base
do_base() {
  log "安装纯 Python 依赖（biotite/rich/hydra/omegaconf/nltk/pandas/scipy）"
  run "$PY" -m pip install -r "$HERE/requirements-repro.txt"
  ok "base 依赖装好"
}

# ---------------------------------------------------------------- esmfold
do_esmfold() {
  local cu idx
  cu="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1 || true)"
  idx=cu124
  case "${cu:-12.0}" in
    12.8*) idx=cu128 ;; 12.6*) idx=cu126 ;; 12.4*) idx=cu124 ;;
    12.1*) idx=cu121 ;; 11.8*) idx=cu118 ;;
  esac
  banner "安装 ESMFold / OpenFold（重、需要 GPU + nvcc）"
  cat <<EOT
官方推荐做法（OpenFold 对 Python>=3.10 不友好，建议单独建 Python 3.9 环境）：

  # 1) 建专用环境
  conda create -y -n esmfold python=3.9
  conda activate esmfold
  export TMPDIR=$TMPDIR        # 已按本机情况自动选好

  # 2) 装 CUDA 版 torch（本机驱动 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '未知')，
  #    驱动支持的 CUDA ${cu:-未知}，故选 $idx）
  pip install torch --index-url https://download.pytorch.org/whl/${idx}

  # 3) 装本仓库 + ESMFold 附加依赖
  cd $REPO_ROOT && pip install -e .
  pip install "fair-esm[esmfold]"
  pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
  pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
  pip install biotite rich

  # 4) 验证（成功会打印一段 PDB）
  python -c "import esm; m=esm.pretrained.esmfold_v1().eval().cuda(); print(m.infer_pdb('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQ')[:80])"

  # 5) 把这个环境写进机器档案，后续脚本就会自动用它
  PY=\$(which python) bash $HERE/00_setup_env.sh hardware --write

注意：
  * 不要装在装论文二那个环境里——两边对 python/torch 要求不同，混装容易两头坏。
  * A6000 是 sm_86，有成熟预编译支持，一般不会踩到新架构缺 kernel 的坑。
  * 若 OpenFold 编译报 nvcc 版本不符：conda install -c nvidia cuda-toolkit=12.8 后再装。
  * 本机可用 CPU 核数 $HW_CPU_CORES、内存 $((HW_MEM_TOTAL_MB/1024))GB，
    编译 OpenFold 的 CUDA kernel 建议加 -j$HW_CPU_CORES 加速。
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
case "${1:-hardware}" in
  hardware) shift || true; do_hardware "${1:-}" ;;
  check)    do_check ;;
  base)     do_base ;;
  cuda)     do_cuda ;;
  esmfold)  do_esmfold ;;
  extern)   do_extern ;;
  all)      do_hardware; do_base; do_cuda; do_esmfold; do_extern; do_check ;;
  *)        err "未知子命令：$1"; err "可用：hardware [--write] | check | base | cuda | esmfold | extern | all"; exit 2 ;;
esac
