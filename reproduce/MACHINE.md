# 设备适配说明

本文档讲**这套脚本怎么自动适配你的机器**，以及在你这台 **6 × RTX A6000 服务器**上该怎么用。

---

## 0. 你的机器 → 脚本会自动得出什么

按你贴的 `nvidia-smi` / `free -h`（驱动 **570.133.20**，CUDA **12.8**，6×A6000 每张 48 GB，内存 503 GB），
脚本运行 `bash reproduce/00_setup_env.sh hardware` 会输出：

```
────────────────────────────────────────────────────────────────
 机器档案  zhangzijian@hit
────────────────────────────────────────────────────────────────
 CPU      : <自动探测> 核
 内存     : 总 503 GB / 可用 361 GB
 TMPDIR   : /dev/shm/zhangzijian/repro_tmp（内存盘，快）
 仓库分区 : <自动探测>
 GPU      : 6 张（驱动 570.133.20）
   idx | 型号                          |  总显存 |  已用 | 空闲 | 利用率
    0 | NVIDIA RTX A6000              |    48GB |    6GB |   42GB |    0%
    1 | NVIDIA RTX A6000              |    48GB |   14GB |   34GB |   91%   ← 别人在用
    2 | NVIDIA RTX A6000              |    48GB |    0GB |   48GB |    0%
    3 | NVIDIA RTX A6000              |    48GB |   16GB |   32GB |    0%
    4 | NVIDIA RTX A6000              |    48GB |    0GB |   48GB |    0%
    5 | NVIDIA RTX A6000              |    48GB |   12GB |   36GB |   27%   ← 别人在用
 空闲 GPU : 2,4,0,3    （判定：利用率≤20%，空闲显存≥19 GB）
────────────────────────────────────────────────────────────────
 自动调参 : ESMFOLD_CHUNK=512  JACKHMMER_CPU=<核数×3/4，上限64>  N_WORKERS=4
 选定设备 : DEVICE=cuda:2          ← 自动挑最空的那张
────────────────────────────────────────────────────────────────
```

四个关键推断：

| 推断 | 依据 | 结果 |
|:--|:--|:--|
| **用哪几张卡** | 6 张里 1 号（91%）和 5 号（27%）有别人的进程 → 剔除 | **2,4,0,3 四张可用** |
| **并行度** | worker 数 = 空闲卡数 | **4 个进程并行** |
| **ESMFold 分块** | 48 GB 显存足够一次算完 512 残基 | **chunk=512**（论文一最长设计 720 残基会切 2 块） |
| **TMPDIR** | `/dev/shm` 可写且 >2 GB | **内存盘**（下载/解压快，但重启丢；大库仍建议放磁盘） |

> 这一步**必须最先做**，因为它决定了后面所有实验的并行度与显存参数。

---

## 1. 一次性配置（3 步）

```bash
cd /path/to/esm/reproduce

# ① 探测硬件并写成机器档案（后续所有脚本自动加载）
bash 00_setup_env.sh hardware --write
#    → 生成 reproduce/machine.env，里面全是 ${VAR:-默认} 形式，
#      所以命令行显式传的环境变量永远优先；要改就编辑这个文件

# ② 体检 + 装依赖
bash 00_setup_env.sh check
bash 00_setup_env.sh base          # biotite/rich/hydra/omegaconf/nltk/pandas/scipy

# ③ 装 CUDA 版 torch（脚本会读你的驱动，自动选 cu128）
bash 00_setup_env.sh cuda
#    → 然后跑 00_setup_env.sh hardware --write 让 PY 指向这个环境

# ④ 论文一的 ESMFold 单独建环境（python3.9 + OpenFold），再写一次档案
bash 00_setup_env.sh esmfold       # 打印带本机参数的安装命令
PY=$(which python) bash 00_setup_env.sh hardware --write
```

`machine.env` 生成样例（关键几行）：

```bash
: "${PY:=/path/to/envs/esmfold/bin/python}"
: "${DEVICE:=}"                 # 留空 = 每次运行自动挑最空的卡
: "${GPUS:=}"                   # 固定卡片时写 "0,2,4,5"
: "${GPU_MAX_UTIL:=20}"         # 利用率 ≤ 20% 才算空闲
: "${GPU_MIN_FREE_MB:=20000}"   # 空闲显存 ≥ 19GB 才算空闲
: "${ESMFOLD_CHUNK:=512}"       # A6000 48GB
: "${JACKHMMER_CPU:=<自动>}"
: "${N_WORKERS:=4}"
: "${TMPDIR:=/dev/shm/zhangzijian/repro_tmp}"
: "${OUT_ROOT:=$REPO_ROOT/reproduce/outputs}"   # ⚠️ 不要放 /dev/shm
: "${TORCH_HOME:=$DATA_ROOT/weights/torch}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
```

---

## 2. 多卡怎么工作

### 2.1 为什么是「进程级分片」而不是「单进程多卡」

查过源码，两条硬约束：

1. **ESMFold 是单卡模型**，一个实例只能待在一张卡上；
2. 论文二的 `lm_design.py` 里写死了 `assert num_seqs == 1`（"Only 1 sequence design in parallel supported for now."），
   没法把多个 seed 塞进一个进程批量跑。

所以唯一干净的做法是**每张卡起一个进程，各自跑一个互不重叠的工作分片**。

### 2.2 分片是怎么切的（保证不重不漏）

两个驱动都支持 `--shard I --shard-total N`：

- **论文一**：把 `(spec, seed)` 扁平成一个列表，按下标取模切分。
  例：单链对称论文规模共 180 条，`N=4` 时每个分片 45 条。
- **论文二**：按 seed 列表取模切分。例：`--seeds 0-199`、`N=4` 时每个分片 50 个 seed。

因为切分只依赖「网格定义 + 分片号」，与运行顺序无关，所以：

- **各分片合起来恰好等于全部工作**；
- 中断后重跑同一条命令，已完成的自动跳过（`design.fasta` / `metrics.json` 为判据），**续跑不会重复算**。

### 2.3 一键脚本已经接好

`run_p1.sh` / `run_p2.sh` 会自动：挑空闲卡 → 起 N 个 worker → 每个 worker
`CUDA_VISIBLE_DEVICES=<物理卡>` + 注入 `--shard/--shard-total` → 收齐退出码 → 汇总。

worker 内部统一写 `--device cuda:0`，因为 `CUDA_VISIBLE_DEVICES` 已经把物理卡重映射成 0。

日志分流到 `<OUT>/logs/worker_gpu<卡>_shard<i>of<n>.log`，方便定位是哪个分片出问题。

```bash
# 什么都不用改，直接跑
MODE=full OUT_ROOT=/your/disk/repro bash reproduce/run_all.sh

# 想固定用几张卡（比如避开夜间别人的调度）
export GPUS=0,2,4,5
# 想只用 2 张卡降低对别人的影响
export GPUS=2,4

# 想调整「空闲」的判定门槛（别人的任务占着显存但利用率是 0 的情况）
export GPU_MAX_UTIL=10 GPU_MIN_FREE_MB=30000
```

---

## 3. 针对 A6000 做的具体调参

| 参数 | 默认值 | A6000 上的取值 | 为什么 |
|:--|:--|:--|:--|
| `ESMFOLD_CHUNK` | 128 | **512** | 分块只在 L > chunk 时才生效。论文一最长设计 720 残基（8 聚体），512 只需切 2 块；L≤512 的单链/对称设计**完全不切块**，最快 |
| `--chunk-size` 透传 | — | 已接进 `design_programs.py` | 命令行可覆盖，例如显存被挤时 `--chunk-size 256` |
| OOM 自动降档 | 无 | **有** | 共享机器上别人的进程会突然吃显存。捕获 CUDA OOM → 清缓存 → 分块减半 → 重试（最多 3 次），跑动不会整条失败 |
| `PYTORCH_CUDA_ALLOC_CONF` | — | `expandable_segments:True` | 减少碎片型 OOM（大卡上效果明显） |
| `JACKHMMER_CPU` | 核数/2 | **核数×3/4（上限 64）** | 503 GB 内存 + 多核，jackhmmer 可以开满；论文新颖性检索是 CPU 密集 |
| `TMPDIR` | `/tmp` | **`/dev/shm/$USER/repro_tmp`** | 内存盘，下载/解压/临时文件快一个量级；但**重启即清空** |
| 显存占用（单 worker） | — | ESMFold 约 5–8 GB；lm-design 约 6–10 GB | 48 GB 一张卡放一个 worker 绰绰有余，甚至可给同一张卡塞 2 个（见 §4） |

---

## 4. 可选：一张卡跑两个 worker（榨干 48 GB）

6 张卡里只有 4 张空闲时，如果还想加并行度，可以让一张卡跑两个 worker：

```bash
export GPUS=2,2,4,4,0,0,3,3     # 同一个卡号重复，脚本会起 8 个 worker
```

这样做的前提是单 worker 显存占用 × 2 仍显著小于 48 GB（ESMFold 单 worker 约 5–8 GB，够）。
但注意：

- **不建议**给 `lm-design` 这么配（它单进程显存波动较大，且 MCMC 是长任务）；
- 卡内争抢会让单条跑动变慢，总吞吐不一定涨；建议先在一个 target 上试跑对比。

---

## 5. 内存与磁盘（503 GB / `/dev/shm`）

| 用途 | 建议 | 理由 |
|:--|:--|:--|
| `TMPDIR` | `/dev/shm/$USER/repro_tmp` | 快；但**别把大库解压到这里**（见下） |
| `OUT_ROOT`（实验产物） | **磁盘**，例如 `/data/repro` | 论文一 full 约 15 GB、论文二 full 约 5 GB；放内存盘重启会丢 |
| `DATA_ROOT`（数据集） | **磁盘** | UniRef90 解压后 ~80 GB，PDB 快照几百 GB |
| UniRef90 解压 | **磁盘**，别放 `/dev/shm` | 解压后 ~80 GB 会长期占内存；`/dev/shm` 通常是 RAM 的一半（≈250 GB），挤占会影响正在跑的 GPU 任务 |
| jackhmmer 索引 | 与 fasta 同目录 | 会生成 `.ssi` 等索引文件，第二次检索快很多 |

脚本已经在 `common.sh` 里加了检查：一旦发现 `OUT_ROOT` 或 `DATA_ROOT` 落在 `/dev/shm` 下会**直接警告**。

```bash
# 推荐布局
export DATA_ROOT=/data/esm_data        # 磁盘，装数据集与权重
export OUT_ROOT=/data/repro            # 磁盘，装实验产物
export TMPDIR=/dev/shm/$USER/repro_tmp # 内存盘，只放临时文件
```

---

## 6. 共享机器的礼貌与安全

你的机器上有别人的任务（GPU 1 跑到 91%、GPU 5 占 12 GB），脚本做了两件事：

1. **不碰忙碌的卡**：默认只看利用率 ≤ 20% 且空闲显存 ≥ 19 GB 的卡；
2. **不抢显存**：单 worker 最多用掉一张卡的一小部分，且 OOM 时会自己降档重试而不是崩掉。

另外几条建议：

```bash
# 固定只用自己的卡，避免调度抖动
export GPUS=2,4

# 提高门槛，连「占着显存但没在算」的卡也避开（比如 3 号卡占了 16GB、利用率 0%）
export GPU_MIN_FREE_MB=40000

# 给 worker 限 CPU 线程，避免和别人的任务抢 CPU
export OMP_NUM_THREADS=8
```

跑长任务前建议 `nvidia-smi` 看一眼，并在 `screen`/`tmux` 里跑：

```bash
tmux new -s repro
MODE=full OUT_ROOT=/data/repro bash reproduce/run_all.sh 2>&1 | tee /data/repro/run.log
```

---

## 7. 快速自检（5 条命令）

```bash
cd /path/to/esm/reproduce
bash 00_setup_env.sh hardware                     # 1) 硬件 + 自动调参
bash 00_setup_env.sh hardware --write             # 2) 写机器档案
DRY_RUN=1 ONLY=paperdata bash run_all.sh          # 3) 零成本链路（不碰 GPU）
DRY_RUN=1 MODE=smoke bash run_all.sh              # 4) 看多卡分片计划长什么样
MODE=smoke SKIP_DATA=1 bash run_all.sh            # 5) 真跑冒烟
```

第 4 步会打印类似：

```
  [dry-run] CUDA_VISIBLE_DEVICES=2 .../design_programs.py --task free_hallucination \
      --grid smoke --steps 200 --device cuda:0 --chunk-size 512 \
      --shard 0 --shard-total 4 --out-dir .../outputs/p1
  [dry-run] CUDA_VISIBLE_DEVICES=4 ... --shard 1 --shard-total 4 ...
  [dry-run] CUDA_VISIBLE_DEVICES=0 ... --shard 2 --shard-total 4 ...
  [dry-run] CUDA_VISIBLE_DEVICES=3 ... --shard 3 --shard-total 4 ...
```

看到 4 行、卡片号是空闲的那 4 张，就说明适配成功。

---

## 8. 与本机（笔记本电脑）的差异一览

同一套脚本在两台机器上的行为：

| 项目 | 笔记本（1 张 RTX 5070 Ti Laptop，CPU 版 torch） | 你的 A6000 服务器 |
|:--|:--|:--|
| `DEVICE` | `cpu`（torch 看不到 CUDA → 诚实回退） | `cuda:2`（自动挑最空的） |
| `N_WORKERS` | 1 | 4（空闲卡数） |
| `ESMFOLD_CHUNK` | 128 | 512 |
| 并行方式 | 单进程 | 4 进程分片 |
| 论文一可否完成 | 不可行（CPU） | **可行，8 卡时级 → 4 卡约 10 天**（见 RUNBOOK §4.3） |
| 论文二 | 只能跑公开数据复算 | 可抽样复现趋势 |

**这正是自动适配的意义**：脚本不假设任何机器，读到什么硬件就按它调参。
