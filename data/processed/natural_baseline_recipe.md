# 论文二「天然蛋白对照集」重建规则（App A.1.1）

论文原文：*A small (N = 214) set of natural proteins with structures in the PDB was selected to
serve as a baseline comparison ... The set is composed of PDBs available on July 2020 that have
sequence identity < 0.3 to the dataset used to train the structure projection, according to
mmseqs2. A length filter of 50 ≤ L < 250 was applied.*

重建步骤：

1. 取 PDB 发布日期 ≤ 2020-07 的条目：
   `curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx`
   （第 3 列 `ACCESSION DATE` 即初次发布日期）
2. 长度过滤 50 ≤ L < 250（可由清理后的单链 PDB 统计残基数）
3. 用 mmseqs2 与「结构投影训练集」比对，保留 sequence identity < 0.3 的条目：
   `mmseqs easy-search candidate.fasta projection_train.fasta out.m8 tmp --min-seq-id 0.3`
4. 随机取 214 条（论文未给随机种子，条目集合不必逐条一致）

论文用这 214 条做的是 Fig S1 / S2 / Table S1（结构理解基线），
**不参与任何设计实验**，所以不做也不影响主结论复现。
