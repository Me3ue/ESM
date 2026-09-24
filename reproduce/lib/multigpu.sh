#!/usr/bin/env bash
# ============================================================================
#  多卡并行调度（被 run_p1.sh / run_p2.sh 使用）
#
#  为什么是「进程级分片」而不是「单进程多卡」：
#    - ESMFold 是单卡模型，一次只能放在一张卡上；
#    - 论文二的 lm_design 里硬写了 `assert num_seqs == 1`（不支持进程内批量）。
#    所以唯一干净的做法：每张卡起一个进程，各自跑互不重叠的工作分片，
#    由被调脚本的 `--shard I/N` 保证「所有分片合起来恰好是全部工作，且不重不漏」。
#
#  用法：
#      source "$HERE/../lib/multigpu.sh"
#      run_on_gpus --gpus "$GPU_LIST" --log-dir "$OUT/logs" -- \
#          "$PY" "$HERE/design_programs.py" --task X --shard {SHARD} --shard-total {NSHARD} \
#              --device cuda:0 --out-dir "$OUT"
#
#    占位符：{SHARD} 从 0 开始；{NSHARD} 分片总数；{GPU} 物理卡号。
#    注意：worker 里统一写 `--device cuda:0`，因为 CUDA_VISIBLE_DEVICES 已把物理卡重映射为 0。
# ============================================================================

# 防止重复 source
if [[ -n "${_REPRO_MULTIGPU_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_REPRO_MULTIGPU_LOADED=1

# 用空格拼接数组（不要用 "${arr[*]}"，会被 local IFS=',' 影响成逗号分隔）
_join_spaces() {
  local out="" a
  for a in "$@"; do out+="$a "; done
  printf '%s' "${out% }"
}

# 把命令模板里的三个占位符替换掉
_subst_placeholders() {
  local shard="$1" nshard="$2" gpu="$3"; shift 3
  local -a out=()
  local a
  for a in "$@"; do
    a="${a//\{SHARD\}/$shard}"
    a="${a//\{NSHARD\}/$nshard}"
    a="${a//\{GPU\}/$gpu}"
    out+=("$a")
  done
  printf '%s\0' "${out[@]}"
}

run_on_gpus() {
  local gpus="${GPUS:-}" log_dir="" extra_env="${WORKER_ENV:-}" resume=""
  while (($#)); do
    case "$1" in
      --gpus)    gpus="$2"; shift 2 ;;
      --log-dir) log_dir="$2"; shift 2 ;;
      --env)     extra_env="$2"; shift 2 ;;
      --)        shift; break ;;
      *)         break ;;
    esac
  done
  local -a cmd=("$@")
  ((${#cmd[@]})) || { err "run_on_gpus: 没有给命令"; return 2; }

  # ---- 解析 GPU 列表
  local -a gpu_arr=()
  local g
  if [[ -n "$gpus" ]]; then
    local IFS=','
    for g in $gpus; do g="${g// /}"; [[ -n "$g" ]] && gpu_arr+=("$g"); done
  fi

  # ---- 没有卡：退化成单进程
  if ((${#gpu_arr[@]} == 0)); then
    if [[ "${DEVICE:-cpu}" == "cpu" ]]; then
      warn "没有可用 GPU：改用单进程 CPU 模式（论文规模下实际不可完成，仅供冒烟）"
      mapfile -d '' -t _c < <(_subst_placeholders 0 1 -1 "${cmd[@]}")
      run_timed "单进程(CPU)" "${_c[@]}" && return 0 || return 1
    fi
    gpu_arr=("${DEVICE##*:}")
    warn "未指定 --gpus，退化为单卡 cuda:${gpu_arr[0]}"
  fi

  local n=${#gpu_arr[@]}

  # ---- dry-run：只打印
  if [[ "$DRY_RUN" == "1" ]]; then
    local i=0
    for g in "${gpu_arr[@]}"; do
      mapfile -d '' -t _c < <(_subst_placeholders "$i" "$n" "$g" "${cmd[@]}")
      dim "  [dry-run] CUDA_VISIBLE_DEVICES=$g ${extra_env}$(_join_spaces "${_c[@]}")"
      i=$((i+1))
    done
    return 0
  fi

  (( n > 1 )) && log "多卡并行：$n 个 worker，卡片 = ${gpu_arr[*]}"
  [[ -n "$log_dir" ]] && mkdir -p "$log_dir"

  # ---- 启动
  local -a pids=() shards=() used_gpus=()
  local i=0 lf
  for g in "${gpu_arr[@]}"; do
    mapfile -d '' -t _c < <(_subst_placeholders "$i" "$n" "$g" "${cmd[@]}")
    lf="/dev/null"
    [[ -n "$log_dir" ]] && lf="$log_dir/worker_gpu${g}_shard${i}of${n}.log"

    if [[ -n "$extra_env" ]]; then
      # shellcheck disable=SC2086
      env CUDA_VISIBLE_DEVICES="$g" $extra_env "${_c[@]}" >"$lf" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="$g" "${_c[@]}" >"$lf" 2>&1 &
    fi
    pids+=($!); shards+=("$i"); used_gpus+=("$g")
    log "启动 worker $((i+1))/$n  GPU=$g  shard=$i/$n  →  $lf"
    i=$((i+1))
  done

  # ---- 收集退出码
  local fail=0 k=0
  for p in "${pids[@]}"; do
    if wait "$p"; then
      ok "worker shard=${shards[$k]} GPU=${used_gpus[$k]} 正常结束"
    else
      err "worker shard=${shards[$k]} GPU=${used_gpus[$k]} 非零退出"
      [[ -n "$log_dir" ]] && warn "  看日志：$log_dir/worker_gpu${used_gpus[$k]}_shard${shards[$k]}of${n}.log"
      fail=$((fail+1))
    fi
    k=$((k+1))
  done

  if (( fail > 0 )); then
    warn "$fail/$n 个 worker 失败。已完成的跑动都已落盘，重跑同一条命令即可续跑。"
    return 1
  fi
  return 0
}

# 打印「本机打算怎么并行」的说明（给 run_all.sh / run_p1.sh 用）
multigpu_plan() {
  hw_probe; hw_auto_tune
  if [[ "$HW_GPU_COUNT" -eq 0 ]]; then
    dim " 并行策略：无 GPU → 单进程 CPU"
    return 0
  fi
  dim " 并行策略：worker 数 = 空闲卡数（N_WORKERS=$N_WORKERS，当前空闲卡=$(hw_free_gpus || echo 无)）"
  dim "   空闲判定：利用率 ≤ ${GPU_MAX_UTIL:-20}%、空闲显存 ≥ $(( ${GPU_MIN_FREE_MB:-20000} / 1024 )) GB"
  dim "   ESMFold 分块 ESMFOLD_CHUNK=$ESMFOLD_CHUNK（序列长 ≤ 分块时不切块，最快）"
  if [[ -n "${GPUS:-}" ]]; then dim "   已固定 GPUS=$GPUS"; else dim "   想固定卡片：export GPUS=0,2,4,5"; fi
}
