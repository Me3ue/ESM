# 数据集：获取与组织

两篇论文需要的数据全部在这里说清楚——**哪些是必须的、从哪下、多大、怎么放、怎么校验**。
配套一键脚本：`reproduce/fetch_datasets.sh`。

```bash
bash reproduce/fetch_datasets.sh plan     # 先看清单（不下载）
bash reproduce/fetch_datasets.sh core     # 必下部分：PDB + 权重 + 论文数据包（≈ 8.8 GB）
```

---

## 0. 先分清三类数据（这决定了你要不要熬夜下 150 GB）

| 类别 | 内容 | 体积 | 什么时候需要 |
|:--|:--|--:|:--|
| **A · 必需** | PDB 目标结构 + 模型权重 + 论文公开数据包 | **≈ 8.8 GB** | 跑任何实验都需要 |
| **B · 论文规模才需要** | UniRef90、PDB 全库快照、AlphaFold DB | **32 GB ~ 200 GB+** | 复现新颖性图 / TM-score 图 / 图 4D |
| **C · 只有要重训才需要** | 结构投影层训练集（Yang et al. 2020，15,051 条） | ≈ 20 GB | 论文二 App A.2.2；**不重训就完全不需要** |

> **关键结论**：论文二的结构投影层**已发布训练好的权重**（`linear_projection_model.pt`，0.5 MB），
> `lm_design` 代码会自动下载。所以「训练数据」这一类**默认不用下**。
> 同理，论文一**不训练任何东西**（能量函数 + 冻结的 ESMFold），所以论文一没有训练集。

---

## 1. 目录规范

沿用仓库既有约定（见 `PROJECT_GUIDE.md` §8.3.1）：`data/raw/` 存原始下载，`data/processed/` 存可直接喂模型的数据。

```text
data/
├── raw/                                原始下载，不修改
│   ├── pdb_targets/                    1QYS 5L33 6D0T 6MRS 6W3W 6WVS + 39 个 de novo target
│   ├── pdb_complexes/                  位点脚手架用的复合物 1Y6K 6M0J 1GHQ 5JW3 7MMO
│   ├── pdb_snapshot_2022-08/           可选：TM-score 穷举用的 PDB 快照 + entries.idx
│   ├── uniref90/                       可选：uniref90.fasta(.gz)
│   ├── alphafold_db/                   可选：AF DB 结构
│   ├── projection_train/               可选：Yang et al. 2020 的训练列表/结构
│   ├── natural_baseline/               可选：214 条天然对照
│   └── paper_data/                     论文二公开数据包（free_generations_full.db / *.hdf5）
├── processed/                          可直接被脚本吃
│   ├── lm_design_targets/              从 raw 清洗出的**单链、无配体、无水的纯蛋白 PDB** + targets.json
│   ├── scaffold_sites/paper_sites.json 5 个功能位点的 PDB/链/残基区间（论文 Methods A.3.4）
│   ├── natural_baseline_recipe.md      214 条天然对照的重建步骤
│   ├── projection_train/               可选：80/20 切分后的 train/valid 列表
│   └── swissprot/                      ⚠️ 这是 NMF 实验的数据，与这两篇论文无关，不要动
├── embeddings/                         预留（esm-extract 输出）
├── structures/                         预留（esm-fold 输出）
├── scores/                             预留
├── weights/torch/hub/checkpoints/      模型权重（通过 TORCH_HOME 指向这里）
└── DATASETS.manifest.tsv               下载清单：相对路径 / 字节数 / sha256 / URL
```

**为什么权重放在 `data/weights/torch`**：`esm.pretrained.*` 下载到 PyTorch Hub 缓存。
用 `export TORCH_HOME=<repo>/data/weights/torch` 就能把它固定在项目里，换机器时可整体拷贝。
代码里等价写法是 `torch.hub.set_dir("<repo>/data/weights/torch/hub")`（必须在加载模型**之前**调用）。

---

## 2. 论文一需要的数据

### 2.1 结论先行

论文一是**纯推理 + 优化**：能量函数由冻结的 ESMFold 提供，没有任何训练环节。所以它只需要：

| 数据 | 用途 | 来源 |
|:--|:--|:--|
| 6 个 de novo target 骨架 | 图 2D-F 固定骨架设计的模板 | RCSB |
| 5 个天然复合物 | 图 2H / 5B 功能位点脚手架的位点模板 | RCSB |
| PDB 全库快照（可选） | 图 S2C/S3C 结构新颖性 TM-score | RCSB / wwPDB |
| ESMFold v1 权重 2.6 GB | 每一步优化的结构预测 | Meta 公共文件 |
| ESM-IF1 权重 1.6 GB | 图 3C-D / 4C-D 逆折叠 roundtrip | Meta 公共文件 |
| ProteinMPNN 权重（可选） | 图 S2D 的第二套逆折叠对照 | ProteinMPNN 仓库 |
| 单序列 AlphaFold2 参数（可选） | 图 2C 的 ssAF2 pLDDT | AF2 官方 |

### 2.2 PDB 结构的具体清单

```bash
bash reproduce/fetch_datasets.sh pdb
```

**固定骨架设计的 6 个 de novo target**（Methods A.3.2）：`1QYS 5L33 6D0T 6MRS 6W3W 6WVS`
→ `data/raw/pdb_targets/`

**功能位点脚手架的 5 个复合物**（Methods A.3.4）：`1Y6K 6M0J 1GHQ 5JW3 7MMO`
→ `data/raw/pdb_complexes/`

每个位点的 PDB ID、链、残基区间已经整理成 `data/processed/scaffold_sites/paper_sites.json`：

| 位点 | PDB | 链 | 残基（闭区间） | 说明 |
|:--|:--|:--|:--|:--|
| IL10 | 1Y6K | L | 31–40 | IL-10R1 结合位点 |
| ACE2 | 6M0J | A | 5–23 | SARS-CoV-2 RBD 结合面 |
| C3d | 1GHQ | A | **104–126 与 170–184**（不连续） | 补体受体 2 |
| HA2 | 5JW3 | B | **14–21 / 33–42 / 45–49**（三段） | 流感 HA2 表位 |
| RBD | 7MMO | C | **439–450 与 498–506** | bebtelovimab 表位 |

> ⚠️ 三处容易踩的坑：
> 1. 语言运行时的 `get_atomarray_in_residue_range(atoms, start, end)` 是 **[start, end) 半开区间**，
>    论文给的是闭区间，所以 `end` 要 **+1**。
> 2. 该函数**不过滤链**（只按 `res_id` 筛），而上面 5 个位点分别在不同链上。用
>    `design_programs.py --task functional_site_scaffolding` 的通用版时，务必确认取到的是正确链的原子。
> 3. 仓库自带的 `programs/functional_site_scaffolding.py::scaffolding_ace2()` 用的是
>    `start=23, end=42`，与论文的 5–23 不是同一段。两者不可混比（`ace2_repo` vs `ace2_generic`）。

### 2.3 结构新颖性对照（图 S2C / S3C，可选）

论文用的是 **PDB 2022-08 快照 + TM-align 20210107**，对每个设计穷举找 TM-score 最高的条目。

**不要真下全库**（>200 GB）。推荐三步走：

```bash
# ① 拿条目索引（含初次发布日期），57 MB
curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
#    列：IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE, AUTHOR LIST, RESOLUTION, ...

# ② 用 Foldseek 在结构库上预筛，取 top-100 候选（秒级）
conda install -y -c bioconda foldseek
foldseek easy-search design.pdb afdb_or_pdb_db out.m8 tmp --alignment-type 1 -e 1e-3

# ③ 只对候选下载 PDB，用 TM-align 精确打分，取最大值
for id in $(cut -f2 out.m8 | head -100); do
  curl -sO https://files.rcsb.org/download/${id}.pdb
  tm-align -byresi design.pdb ${id}.pdb | grep "^TM-score"
done
```

论文用 TM-align 的 `-byresi` 模式（按设计长度归一化）。判定红线：**TM-score = 0.6**。

整库快照（如果磁盘真的够）：

```bash
rsync -av --include='*/' --include='*.ent.gz' --exclude='*' \
  rsync.wwpdb.org::ftp_data/structures/divided/pdb/ data/raw/pdb_snapshot_2022-08/
```

---

## 3. 论文二需要的数据

### 3.1 de novo target set（App A.1.1）

```bash
bash reproduce/fetch_datasets.sh pdb
# 或单独：python reproduce/paper2_lm_design/run_lm_design_batch.py --fetch-pdbs --pdb-dir <dir>
```

论文列了 **39 个** de novo target，但其中 `6DKM A` / `6DKM B` / `6DLM A` / `6DLM B`
是 4 个「PDB ID + 链」条目，对应 **2 个 PDB 文件** ⇒ **去重后只需下载 37 个 PDB**。

长度范围 **67 ≤ L ≤ 184**，覆盖 α-bundle、β-barrel、NTF2、Rossman 等 fold。
这 39 个结构在论文里同时用于固定骨架设计的模板（图 2A）和「模型是否理解 de novo 蛋白」的评估（图 S1/S2）。

**清洗**：`fetch_datasets.sh processing` 会把整条 entry 拆成**单链、去水、去配体**的纯蛋白 PDB，
放到 `data/processed/lm_design_targets/`，并写出 `targets.json`（记录用了哪条链、残基数）。
`lm_design` 的 `pdb_loader` 需要干净的 CA 坐标，多链/含配体的 entry 会出错。

### 3.2 模型权重

```bash
bash reproduce/fetch_datasets.sh weights
```

| 文件 | 体积 | 用途 |
|:--|--:|:--|
| `esm2_t33_650M_UR50D.pt` | 2.5 GB | 设计用的语言模型（App A.2.1） |
| `esm2_t33_650M_UR50D-contact-regression.pt` | — | 配套 contact 回归头 |
| `linear_projection_model.pt` | 0.5 MB | 结构投影层（图 1C，App A.2.2）**已训练好，直接用** |
| `esmfold_3B_v1.pt` | 2.6 GB | 论文一用；论文二可作 ESMFold 版 oracle（替代 AlphaFold） |
| `esm_if1_gvp4_t16_142M_UR50.pt` | 1.6 GB | 论文一 roundtrip |

### 3.3 n-gram 先验（仓库自带，不用下）

`examples/lm-design/utils/ngram_stats/{monogram,bigram,trigram,quadgram}_seg.p`
—— 来自 **UniRef50 2018_03** 的氨基酸 n-gram 频率（App A.2.3），是能量函数里的 `E_ngram` 项。仓库已置备。

### 3.4 序列新颖性库（图 2G / 4F-G）★★ 这是论文二最关键的外部数据

```bash
bash reproduce/fetch_datasets.sh seqdb                  # 当前版 UniRef90，32 GB
bash reproduce/fetch_datasets.sh seqdb --release 2021_04  # 论文原始版本，158 GB
```

| 选项 | 文件 | 下载体积 | 解压后 | 说明 |
|:--|:--|--:|--:|:--|
| 推荐 | `uniref90.fasta.gz`（当前版） | **32 GB** | ≈ 80 GB | 数值会相对论文略有偏移 |
| 论文原版 | `uniref2021_04.tar.gz` | **158 GB** | 更大 | 官方把 UniRef50/90/100 打成一个包，无法只取 90 |

**为什么必须注意版本**：论文用的 `UniRef90 2021_04` 是 ESM2 训练集的超集，
并且论文从检索结果里**剔除**了两类命中（App A.1.2）：

- `paper-data/artificial_sequence_purge_ids.txt` —— 被 UniProt 标为 artificial 的 1,027 条
- `paper-data/uniref90_jackhmmer_purge_ids.txt` —— 用 de novo target 反查 UniRef90 得到的 58,462 条命中

这两个列表**仓库自带**（`examples/lm-design/paper-data/`），`analyze_novelty.py` 默认就会读进来。
但它们是按 2021_04 编的，**换成当前版 UniRef90 后 purge 与实际库会错位**，
报告里必须注明（脚本产出的 `NOVELTY.md` 会自动写上库名与 purge 列表来源）。

检索设置也必须照抄论文的三个非默认项：`-n 1`、`--seed 0`、按 **best-domain E-value** 排序
（已在 `analyze_novelty.py` 里固化）。

### 3.5 AlphaFold DB（图 4D + oracle 备用）

论文图 4D / App A.5.2 的做法：把每条设计（25,000 条）与 ≈15k 天然蛋白一起，
对 **AlphaFold DB**（覆盖 UniProt 2021_04）检索 top-1 命中，然后比对
「序列一致性」与「预测结构的 TM-score」。

```bash
bash reproduce/fetch_datasets.sh afdb
```

需要两样东西：
1. **命中序列**：UniProt 2021_04 序列集
   （`https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_04/knowledgebase/`）
   或直接用当前版（数值会有偏移）；
2. **命中结构**：按 accession 逐个取
   `https://alphafold.ebi.ac.uk/files/AF-<UniProtID>-F1-model_v3.pdb`
   （论文当时是 **v3**，现在最新是 **v6**）。量大时先用 Foldseek 预筛，只下 top-N。

判定「远距」的阈值（论文 Fig 4D 左下象限）：**Seq-id < 0.2 且 预测结构 TM-score < 0.5**。
论文在该象限得到 49 条远距自由生成，其中 31 条实验成功。

### 3.6 天然蛋白对照集（可选，只影响基线图）

| 集合 | 规模 | 用途 | 怎么来 |
|:--|--:|:--|:--|
| 天然对照 | **214** 条 | 图 S1/S2、表 S1 的结构理解基线 | 按规则重建，见 `data/processed/natural_baseline_recipe.md` |
| 图 4D 的天然侧 | **≈15k** 条 | 与 25k 生成一起投到 t-SNE 图上 | 引用 [59] Yang et al. 2020 用的 PDB 集合 |

214 条的重建规则（App A.1.1）：PDB 发布日期 ≤ 2020-07 → 长度 `50 ≤ L < 250` →
与结构投影训练集的 sequence identity < 0.3（mmseqs2）→ 随机取 214。
论文没给随机种子，条目集合不必逐条一致。

### 3.7 结构投影层训练集（**默认不需要**）

只在你要**重新训练**投影层（App A.2.2）时才需要：
- 来源：引用 [59] Yang, Anishchenko, Park, Peng, Ovchinnikov, Baker.
  *Improved protein structure prediction using predicted interresidue orientations.*
  PNAS 117(3):1496–1503, 2020（即 trRosetta 论文）
- 规模：**15,051** 条非冗余 PDB 蛋白，**结构发布日期早于 2018-05-01**，取其中随机 80% 训练
- 训练超参：**10 epoch / batch size 4 / lr 1e-2**，ESM2 权重冻结，损失为预测 distogram 与真值
  distogram 之间的 categorical cross-entropy
- 重建路线：

```bash
# ① 拿含发布日期的条目索引
curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
# ② 筛「ACCESSION DATE < 2018-05-01」的条目
# ③ 用 mmseqs2 去冗余到单一序列一致性阈值，得到 ~15k 规模
mmseqs easy-cluster all.fasta clu tmp --min-seq-id 0.3
# ④ 按长度/分辨率再过滤，下载对应 PDB，构造 distogram（Cβ 推断 + 分 bin）
```

> 训练脚本论文**未开源**；`examples/lm-design/utils/linear_projection.py` 只有模型定义与 binning 方案。
> **不重训就完全不需要这份数据**。

### 3.8 论文二公开数据包（复算用，必下，才 56 MB）

```bash
bash reproduce/fetch_datasets.sh paperdata
```

| 文件 | 体积 | 内容 |
|:--|--:|:--|
| `free_generations_full.db` | 4.9 MB | 25k 自由生成的统计与 PDB（SQLite，表 `free_generations_full`） |
| `design_lm_data_2022_v1.hdf5` | 51.5 MB | 276 条送检蛋白的长表：SEC 曲线、产率、jackhmmer 结果 |
| `paper-data/data.csv`（**仓库自带**） | 68 KB | 276 条蛋白的汇总表 → 直接喂 `paper_data_report.py` |
| `paper-data/*_purge_ids.txt`（**仓库自带**） | 690 KB | 两个 purge 列表 |

```python
import pandas as pd
df = pd.read_sql('free_generations_full', 'sqlite:///<DATA_ROOT>/raw/paper_data/free_generations_full.db')
lf = pd.read_hdf('<DATA_ROOT>/raw/paper_data/design_lm_data_2022_v1.hdf5')
```

> 这两份是**论文的产出数据**（用来复算结论），不是训练输入。

### 3.9 Rosetta 相关（App A.4.2-4.4 的过滤指标，可选）

论文用 Rosetta 做四过滤器，脚本本身不调用 Rosetta，但阈值要照抄：

| 指标 | 过滤器 | 阈值 |
|:--|:--|:--|
| 堆积质量 | `Rosetta PackStat`（≈ RosettaHoles，100 次重复取平均） | **packing > 0.55** |
| 二级结构形状互补 | `SSShapeComplementarity`（`loops=true helices=true`） | **shape complementarity > 0.6** |
| 球状性 | 理想半径 = `2.24 * L^0.392`，与之比对 | **相对 Rg < 1.5**、**相对 SASA < 3** |
| 聚集倾向 | 平均 SAP（spatial aggregation propensity） | **SAP ≤ 0.4**（含 ≥25% β 链时放宽到 0.5） |

Rosetta 需自行安装并获取 license。Packing 与 Shape Complementarity 要**各算两次**
（Amber 松弛后的结构、以及再经 `beta_nov16` 最小化后的结构），两者**逻辑或**。

---

## 4. 磁盘预算

| 档位 | 内容 | 占用 |
|:--|:--|--:|
| **最小可跑** | core（PDB + 权重 + 论文数据包） | **≈ 8.8 GB** |
| 加序列新颖性 | + 当前版 UniRef90 | **≈ 120 GB**（含解压） |
| 加 AF DB 对照 | + swissprot 结构 tar + 逐 accession 结构 | **+ 数 GB ~ 数十 GB** |
| 加 TM-score 穷举 | + PDB 全库快照 + Foldseek 索引 | **+ 200 GB** |
| 加投影层重训 | + Yang et al. 训练集 | **+ 20 GB** |
| 实验产物（论文一 full） | 5,930 条 × (result.json + PDB + fasta) | **≈ 15 GB** |
| 实验产物（论文二 full） | 42,660 条 × (metrics + trajectory) | **≈ 5 GB** |

**建议**：起步给 **150 GB**；要复现新颖性图给 **300 GB**；要 TM-score 穷举或重训投影层给 **1 TB**。

---

## 5. 校验

```bash
bash reproduce/fetch_datasets.sh verify
```

会逐行读 `data/DATASETS.manifest.tsv`（相对路径 / 字节数 / sha256 / URL），
核对存在性、体积与哈希。manifest 由下载脚本自动追加，也可手工补录。

另外两个抽查点：

```bash
# ① PDB 目标数（37 个唯一 PDB）
ls data/raw/pdb_targets/*.pdb | wc -l
# ② 清洗后的单链 target 长度应落在论文声明的 67~184
python -c "import json;d=json.load(open('data/processed/lm_design_targets/targets.json'));\
print(min(x['n_residues'] for x in d), max(x['n_residues'] for x in d))"
```

### 本工具包已实测的结果

`fetch_datasets.sh pdb` 完整跑通过一次，结论：

- 落盘 **42 个 PDB 文件** = 37 个唯一 de novo / 固定骨架 target + 5 个位点复合物
  （`1QYS 5L33 6D0T 6MRS 6W3W 6WVS` 同时属于两个集合，去重后仍为 37）；
- 逐条取最长链统计残基数：**min = 68、max = 183**，完全落在论文 App A.1.1 声明的
  **67 ≤ L ≤ 184** 区间内，**无越界条目** ⇒ 说明「39 个条目、37 个文件」的对应关系读对了；
- `fetch_datasets.sh verify` 对全部 42 个文件做了体积 + sha256 校验，**全部通过**。

具体的长度分布（用于和论文 Fig 2A 的横轴顺序对照）：

| 最短 | | | 最长 | |
|--:|:--|:--|--:|:--|
| 6MRR | 68 | | 7MCD | 183 |
| 6DLM | 74 | | 6WVS | 182 |
| 2N2U / 6MRS | 77 | | 4KYZ | 164 |
| 6E5C | 78 | | 4KY3 | 157 |

---

## 6. 常见坑

| 现象 | 原因 / 处理 |
|:--|:--|
| 权重反复下载 / 找不到权重 | `TORCH_HOME` 没设。`export TORCH_HOME=<repo>/data/weights/torch`，或在加载模型前 `torch.hub.set_dir(...)` |
| `lm_design` 报 PDB 解析错 | target 是多链或含配体。跑 `fetch_datasets.sh processing` 取单链，或用 `pdb_loader` 的 `allow_missing_residue_coords` |
| 39 个 target 只下到 37 个 | **正常**。6DKM/6DLM 各有 A/B 链两个条目，共用 1 个 PDB 文件 |
| 新颖性结果和论文差很多 | 库版本不同（当前版 vs 2021_04）或 purge 列表没生效。检查 `NOVELTY.md` 里写的库名与 purge 列表 |
| 位点脚手架的 RMSD 一直很差 | 取错链或用了闭区间。见 §2.2 的三处坑 |
| `/tmp` 空间不足 | 大文件下载前 `export TMPDIR=<有空间的分区>`（本工具包已默认处理） |
| TF/数仓类数据（Atlas） | 本工具包**不用** ESM Atlas；如需可参考仓库 `scripts/atlas/` 的 URL 清单，规模为 TB 级 |
| 和 NMF 实验抢数据 | `data/processed/swissprot/` 与 `data/raw/swissprot_reviewed.fasta` 属于 NMF 实验，**不要动** |

---

## 7. 一页速查

```bash
# 全部必需数据（≈ 8.8 GB）
bash reproduce/fetch_datasets.sh core

# 论文二序列新颖性（推荐当前版 UniRef90）
bash reproduce/fetch_datasets.sh seqdb
python reproduce/paper2_lm_design/analyze_novelty.py --fasta-dir outputs/p2 \
    --db data/raw/uniref90/uniref90.fasta --db-name uniref90_current \
    --out outputs/p2/novelty

# 论文二公开数据零成本复算（不需要上面任何大文件）
python reproduce/paper2_lm_design/paper_data_report.py

# 校验
bash reproduce/fetch_datasets.sh verify
```
