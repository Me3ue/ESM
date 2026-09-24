# 项目长期记忆（/home/zzj/protein/esm）

## 环境约定
- 主用 conda 环境：**ESM** → `/home/zzj/anaconda3/envs/ESM/bin/python`
  （Python 3.10 + torch 2.14.0+cpu + 本仓库 `pip install -e .` + matplotlib）
- `conda` 可执行文件：`/home/zzj/anaconda3/bin/conda`（另有 `/home/zzj/miniconda3`）
- 本机 `/tmp` 仅 10MB tmpfs：任何大文件下载/pip 安装前先 `export TMPDIR=/home/zzj/tmp`
- **有物理 GPU**（NVIDIA GeForce RTX 5070 Ti Laptop，sm_120），但环境 `ESM` 里装的是
  **CPU 版 torch**（`torch.cuda.is_available() == False`）→ 实验默认 `--device cpu`。
  要用 GPU 必须另建环境装 CUDA 版 torch，**不要动现有 `ESM` 环境**（会破坏已跑通的一切）。
- `openfold / biotite / rich / hydra-core / omegaconf / nltk / pandas / scipy` 均未安装；
  论文复现所需依赖见 `reproduce/requirements-repro.txt`，装法见 `reproduce/00_setup_env.sh`。
- 运行仓库内脚本时清空 `PYTHONPATH`（宿主的 shim 目录会干扰）

## 项目结构约定
- 新增 Python 包需同时写入 `setup.py` 的 `sources` 字典，才能 `pip install -e .` 后被全局 import
- 实验产物统一放 `<包>/outputs/` 并在 `.gitignore` 中忽略
- 用户偏好中文文档与注释；`PROJECT_GUIDE.md` 是本项目的中文总指南（章节编号需保持连续）

## 论文复现约定（`reproduce/`）
- **论文 ↔ 示例目录的固定对应**（别记错）：
  - 《A high-level programming language for generative protein design》→ `examples/protein-programming-language/`
  - 《Language models generalize beyond natural proteins》→ `examples/lm-design/`
- `examples/protein-programming-language/` **没有命令行入口**，`programs/*.py` 只是返回
  `ProgramNode` 的构造函数。一切「跑实验」都要经由 `reproduce/paper1_programming/design_programs.py`
  （9 个任务统一驱动，`--grid paper|smoke`，自动断点续跑）。
- 论文一退火超参：30000 步、`Tmax=1`、`Tmin=1e-4`、退火率 `(Tmin/Tmax)^(1/M)`、
  变异概率 替换/插入/删除 = 60/20/20、**排除半胱氨酸**。权重按 Methods A.3 分任务给，
  `--weights paper` 用论文权重、`--weights repo` 用作者代码权重（多为全 1）。
- 论文二：`lm_design.py` 有 `assert Path.cwd().name == 'lm-design'`；批量必须走
  `reproduce/paper2_lm_design/run_lm_design_batch.py`。其结构模型是**线性投影 distogram**，
  论文的 oracle 是**外部 AlphaFold**，本工具包只能用 ESMFold 代替（数值不可直接比）。
- `examples/lm-design/paper-data/data.csv` 能精确复算论文二摘要核心数字（7/8 项），
  用 `reproduce/paper2_lm_design/paper_data_report.py`（不加载任何模型）。
- 两篇论文的正文规模分别约 **1 万 / 30 万 GPU 小时**，个人只做分层复现：
  `MODE=smoke` 打通 → 单任务 `--grid paper --seeds 10` → 才考虑全量。
- 所有复现脚本必须先 `export TMPDIR=/home/zzj/tmp` 并设 `MPLCONFIGDIR`。

## 实验约定（NMF）
- 线性层三因子非负分解：`W ≈ A @ S @ B + offset`，A/S/B 非负、S 为 r×r 方阵
- 默认整体平移 `shift=min`（实测优于逐行平移）；评估同时报“相对非负目标”和“相对原权重”两个误差
- 训练 NMF 因子时必须加信任域保护（Adam 步长与参数尺度无关，lr 过大会一步破坏分解）
