#!/usr/bin/env bash
# ============================================================================
#  硬件探测与机器档案（被 lib/common.sh 自动 source）
#
#  目标：让同一套脚本在「笔记本 1 张卡」和「6×A6000 服务器」上都能自动跑对。
#
#  做的事情：
#    1. 探测 GPU（型号/显存/占用/利用率）、CPU 核数、内存、磁盘、CUDA 版本；
#    2. 挑出**当前空闲**的 GPU（共享机器上很重要：别人的进程可能占着卡）；
#    3. 按显存自动调参（ESMFold chunk、jackhmmer 线程数、TMPDIR 位置）；
#    4. 把结论写成 `reproduce/machine.env`，后续脚本自动加载（可手工编辑覆盖）。
#
#  主要函数：
#    hw_probe            探测并填充 HW_* 变量（轻量，一次 nvidia-smi）
#    hw_report           打印人类可读的硬件报告
#    hw_free_gpus        打印空闲 GPU 索引（逗号分隔）
#    hw_pick_device      选一张最空闲的卡 → 回显 cuda:N / cpu
#    hw_auto_tune        按显存/核数给出推荐参数
#    hw_write_profile    写 machine.env
#
#  可覆盖的环境变量：
#    GPUS="0,2,4"        指定用哪几张卡（优先级最高）
#    GPU_MAX_UTIL=20     认为「空闲」的利用率上限（%）
#    GPU_MIN_FREE_MB=20000  认为「空闲」的最小空闲显存
#    ESMFOLD_CHUNK=512   ESMFold 分块大小（越大越快、越吃显存）
#    JACKHMMER_CPU=32    jackhmmer 线程数
# ============================================================================

# 防止重复 source
if [[ -n "${_REPRO_HARDWARE_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_REPRO_HARDWARE_LOADED=1

HW_PROBED=0

# ---------------------------------------------------------------- 探测
hw_probe() {
  [[ "$HW_PROBED" == "1" ]] && return 0

  HW_HOST="$(hostname 2>/dev/null || echo unknown)"
  HW_USER="${USER:-$(id -un 2>/dev/null || echo unknown)}"
  HW_CPU_CORES="$(nproc 2>/dev/null || echo 1)"
  HW_CPU_SOCKETS="$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2);print $2}' | head -1)"
  HW_CPU_MODEL="$(lscpu 2>/dev/null | awk -F: '/^Model name/{sub(/^ +/,"",$2);print $2}' | head -1)"

  # 内存（MB）
  HW_MEM_TOTAL_MB="$(awk '/^MemTotal:/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)"
  HW_MEM_AVAIL_MB="$(awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)"

  # GPU：用 query-gpu 而不是解析表格；必须先确认 nvidia-smi 真的能工作，
  # 否则失败信息会跑到 stdout 被当成 CSV 解析（踩过这个坑）
  HW_GPU_COUNT=0
  HW_GPU_NAMES=""
  HW_GPU_TOTAL_MB=""
  HW_GPU_USED_MB=""
  HW_GPU_UTIL=""
  HW_DRIVER=""
  HW_SMI_OK=0
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HW_SMI_OK=1
    local csv
    csv="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null || true)"
    # 只在首行确实是 "<数字>, " 时才认为解析有效
    if [[ "$csv" =~ ^[0-9]+,[[:space:]] ]]; then
      HW_GPU_COUNT="$(printf '%s\n' "$csv" | grep -c '^[0-9]' || true)"
      HW_GPU_NAMES="$(printf '%s\n' "$csv" | awk -F', ' '{printf "%s|", $2}')"
      HW_GPU_TOTAL_MB="$(printf '%s\n' "$csv" | awk -F', ' '{printf "%s|", $3}')"
      HW_GPU_USED_MB="$(printf '%s\n' "$csv" | awk -F', ' '{printf "%s|", $4}')"
      HW_GPU_UTIL="$(printf '%s\n' "$csv" | awk -F', ' '{printf "%s|", $5}')"
      HW_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    else
      HW_SMI_OK=0
    fi
  fi

  # 磁盘（MB）：仓库所在分区 + 候选 TMPDIR 分区
  HW_TMPDIR="$(hw_pick_tmpdir)"
  HW_DISK_REPO_FREE_MB="$(df -Pm "${REPO_ROOT:-.}" 2>/dev/null | awk 'NR==2{print $4}')"
  HW_DISK_TMP_FREE_MB="$(df -Pm "$HW_TMPDIR" 2>/dev/null | awk 'NR==2{print $4}')"
  HW_DISK_TMP_POINT="$(df -P "$HW_TMPDIR" 2>/dev/null | awk 'NR==2{print $6}')"

  HW_PROBED=1
}

# TMPDIR 选择：优先显式指定 → /dev/shm（内存盘，最快，但重启会丢）→ 仓库内 → /tmp
hw_pick_tmpdir() {
  if [[ -n "${TMPDIR_OVERRIDE:-}" && -d "${TMPDIR_OVERRIDE:-}" ]]; then
    echo "$TMPDIR_OVERRIDE"; return 0
  fi
  # 已有的 TMPDIR 若可写且空间够（>2GB）就直接用
  if [[ -n "${TMPDIR:-}" && -d "${TMPDIR:-}" && -w "${TMPDIR:-}" ]]; then
    local fre
    fre="$(df -Pm "$TMPDIR" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$fre" && "$fre" -gt 2048 ]]; then echo "$TMPDIR"; return 0; fi
  fi
  local cand
  for cand in "/dev/shm/${USER:-user}/repro_tmp" "${REPO_ROOT:-.}/.repro_tmp" "/tmp"; do
    if mkdir -p "$cand" 2>/dev/null && [[ -w "$cand" ]]; then
      local fre
      fre="$(df -Pm "$cand" 2>/dev/null | awk 'NR==2{print $4}')"
      if [[ -n "$fre" && "$fre" -gt 2048 ]]; then echo "$cand"; return 0; fi
    fi
  done
  echo "/tmp"
}

# ---------------------------------------------------------------- 空闲 GPU
# 输出：空闲 GPU 的索引，逗号分隔（顺序按「空闲显存从多到少」）
hw_free_gpus() {
  hw_probe
  [[ "${HW_SMI_OK:-0}" == "1" ]] || return 0
  local max_util="${GPU_MAX_UTIL:-20}"
  local min_free="${GPU_MIN_FREE_MB:-20000}"

  # 显式指定 GPUS 时直接用它（仍会过滤掉不存在的索引）
  if [[ -n "${GPUS:-}" ]]; then
    local total out=() g
    total="${HW_GPU_COUNT:-0}"
    IFS=',' read -ra _req <<< "$GPUS"
    for g in "${_req[@]}"; do
      g="${g// /}"; [[ -z "$g" ]] && continue
      if [[ "$g" =~ ^[0-9]+$ ]] && (( g < total )); then out+=("$g"); fi
    done
    ((${#out[@]})) && { local IFS=','; echo "${out[*]}"; }
    return 0
  fi

  # 按空闲显存降序挑。注意查询里带 name 列，字段位置是：
  #   $1=index  $2=name  $3=memory.total  $4=memory.used  $5=utilization
  # （这里踩过一次坑：漏算 name 列导致字段整体错位，空闲卡永远为空）
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',[ ]*' -v mu="$max_util" -v mf="$min_free" '
        NF >= 5 {
          free = $3 - $4
          if (($5 + 0) <= mu && (free + 0) >= mf) printf "%09d %s\n", free, $1
        }' \
    | sort -rn \
    | awk '{printf "%s,", $2}' \
    | sed 's/,$//'
}

hw_pick_device() {
  hw_probe
  if [[ "$HW_GPU_COUNT" -eq 0 ]]; then echo "cpu"; return 0; fi
  local g
  g="$(hw_free_gpus | cut -d, -f1)"
  if [[ -n "$g" ]]; then echo "cuda:${g}"; else echo "cuda:0"; fi
}

# ---------------------------------------------------------------- 自动调参
hw_auto_tune() {
  hw_probe
  # ESMFold 分块：显存越大可以一次算完越长的序列（L <= chunk 时不切块，最快）
  if [[ -z "${ESMFOLD_CHUNK:-}" ]]; then
    local vram=0
    if [[ "${HW_SMI_OK:-0}" == "1" ]]; then
      vram="$(printf '%s' "$HW_GPU_TOTAL_MB" | tr '|' '\n' | grep -E '^[0-9]+$' | sort -rn | head -1)"
      vram="${vram:-0}"
    fi
    if   (( vram >= 40000 )); then ESMFOLD_CHUNK=512
    elif (( vram >= 24000 )); then ESMFOLD_CHUNK=256
    elif (( vram >= 16000 )); then ESMFOLD_CHUNK=128
    elif (( vram >  0    )); then ESMFOLD_CHUNK=64
    else                          ESMFOLD_CHUNK=128   # CPU
    fi
  fi
  # jackhmmer / dataloader 线程：留 1/4 给系统
  if [[ -z "${JACKHMMER_CPU:-}" ]]; then
    local c=$(( HW_CPU_CORES * 3 / 4 ))
    (( c < 1 )) && c=1; (( c > 64 )) && c=64
    JACKHMMER_CPU=$c
  fi
  # GPU 工作进程数 = 空闲卡数（上限 8）
  if [[ -z "${N_WORKERS:-}" ]]; then
    local n=0
    n="$(hw_free_gpus | tr ',' '\n' | grep -c '^[0-9]' || true)"
    n="${n:-0}"
    (( n < 1 )) && n=$(( HW_GPU_COUNT > 0 ? HW_GPU_COUNT : 1 ))
    (( n > 8 )) && n=8
    N_WORKERS=$n
  fi
  export ESMFOLD_CHUNK JACKHMMER_CPU N_WORKERS
}

# ---------------------------------------------------------------- 报告
hw_report() {
  hw_probe
  hw_auto_tune
  echo "────────────────────────────────────────────────────────────────"
  echo " 机器档案  ${HW_USER}@${HW_HOST}"
  echo "────────────────────────────────────────────────────────────────"
  printf ' CPU      : %s 核' "$HW_CPU_CORES"
  [[ -n "${HW_CPU_SOCKETS:-}" ]] && printf '（%s socket）' "$HW_CPU_SOCKETS"
  echo
  [[ -n "${HW_CPU_MODEL:-}" ]] && printf ' 型号     : %s\n' "$HW_CPU_MODEL"
  printf ' 内存     : 总 %s GB / 可用 %s GB\n' \
    "$((HW_MEM_TOTAL_MB/1024))" "$((HW_MEM_AVAIL_MB/1024))"
  printf ' TMPDIR   : %s（所在分区 %s，可用 %s GB）\n' "$HW_TMPDIR" \
    "${HW_DISK_TMP_POINT:-?}" "$((HW_DISK_TMP_FREE_MB/1024))"
  printf ' 仓库分区 : 可用 %s GB\n' "$((HW_DISK_REPO_FREE_MB/1024))"

  if [[ "${HW_SMI_OK:-0}" == "1" && "$HW_GPU_COUNT" -gt 0 ]]; then
    printf ' GPU      : %s 张（驱动 %s）\n' "$HW_GPU_COUNT" "${HW_DRIVER:-?}"
    echo "   idx | 型号                          |  总显存 |  已用 | 空闲 | 利用率"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' '{printf "  %3s | %-29s | %5.0fGB | %4.0fGB | %4.0fGB | %4s%%\n",
                        $1, $2, $3/1024, $4/1024, ($3-$4)/1024, $5}'
    printf ' 空闲 GPU : %s   （判定：利用率≤%s%%，空闲显存≥%s GB）\n' \
      "$(hw_free_gpus | sed 's/^$/无/')" "${GPU_MAX_UTIL:-20}" "$(( ${GPU_MIN_FREE_MB:-20000} / 1024 ))"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo " GPU      : 无可用（nvidia-smi 无法与驱动通信，或本机确实没有 GPU）"
  else
    echo " GPU      : 无（未安装 nvidia-smi）"
  fi

  echo "────────────────────────────────────────────────────────────────"
  printf ' 自动调参 : ESMFOLD_CHUNK=%s  JACKHMMER_CPU=%s  N_WORKERS=%s\n' \
    "$ESMFOLD_CHUNK" "$JACKHMMER_CPU" "$N_WORKERS"
  printf ' 选定设备 : DEVICE=%s' "${DEVICE:-$(hw_pick_device)}"
  if [[ -n "${PY:-}" ]]; then
    printf '    解释器 %s' "$PY"
    if ! hw_torch_cuda; then printf '  %s（torch 看不到 CUDA）%s' "${C_YEL:-}" "${C_RESET:-}"; fi
  fi
  echo
  echo "────────────────────────────────────────────────────────────────"
}

# torch 是否真的能用 CUDA（比 nvidia-smi 更权威 —— 驱动在但 torch 是 CPU 版时两者会不一致）
hw_torch_cuda() {
  [[ -n "${PY:-}" && -x "${PY:-}" ]] || return 1
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null
}

# ---------------------------------------------------------------- 机器档案
hw_write_profile() {
  hw_probe
  hw_auto_tune
  local out="${1:-$REPRO_ROOT/machine.env}"
  local py="${PY:-$(command -v python3)}"
  cat > "$out" <<EOT
# 机器档案（由 00_setup_env.sh hardware --write 生成，可手工编辑）
# 语法约定：全部用 \${VAR:-默认值}，所以命令行传的环境变量永远优先。
# 机器：${HW_USER}@${HW_HOST}  生成时间：$(date '+%F %T')
#
# ---------- 硬件摘要 ----------
# CPU ${HW_CPU_CORES} 核；内存 $((HW_MEM_TOTAL_MB/1024)) GB；GPU ${HW_GPU_COUNT} 张
# TMPDIR ${HW_TMPDIR}（分区可用 $((HW_DISK_TMP_FREE_MB/1024)) GB）；仓库分区可用 $((HW_DISK_REPO_FREE_MB/1024)) GB

# ---------- 解释器（改成你自己的环境路径） ----------
: "\${PY:=$py}"
export PY

# ---------- 设备 ----------
# 留空则每次运行自动挑最空闲的卡；共享机器上建议保持自动
: "\${DEVICE:=}"
export DEVICE
# 要多卡并行时在这里固定用哪几张卡，例如 "0,2,4,5"
: "\${GPUS:=}"
export GPUS
# 「空闲」判定阈值
: "\${GPU_MAX_UTIL:=20}"; export GPU_MAX_UTIL
: "\${GPU_MIN_FREE_MB:=20000}"; export GPU_MIN_FREE_MB

# ---------- 显存相关的性能参数 ----------
: "\${ESMFOLD_CHUNK:=${ESMFOLD_CHUNK}}"; export ESMFOLD_CHUNK
: "\${JACKHMMER_CPU:=${JACKHMMER_CPU}}"; export JACKHMMER_CPU
: "\${N_WORKERS:=${N_WORKERS}}"; export N_WORKERS

# ---------- 目录 ----------
: "\${TMPDIR:=${HW_TMPDIR}}"; export TMPDIR
# 产物目录：**不要**放在 /dev/shm（重启会丢）
: "\${OUT_ROOT:=$REPRO_ROOT/outputs}"; export OUT_ROOT
# 数据根目录（见 DATASETS.md）
: "\${DATA_ROOT:=$REPO_ROOT/data}"; export DATA_ROOT
: "\${TORCH_HOME:=\$DATA_ROOT/weights/torch}"; export TORCH_HOME

# ---------- 显存碎片（A6000 上实测能减少 OOM） ----------
: "\${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"; export PYTORCH_CUDA_ALLOC_CONF
EOT
  echo "$out"
}
