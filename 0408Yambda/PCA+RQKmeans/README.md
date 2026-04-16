# PCA + RQKMeans SID 对比实验

本目录用于比较 Yambda item semantic ID 的几种构造方式：

1. `rawRQKmeans`: 直接在 `normalized_embed` 128 维空间上做 RQKMeans。
2. `pca32_nowhiten`: PCA 降到 32 维，不 whiten，再做 RQKMeans。
3. `pca64_nowhiten`: PCA 降到 64 维，不 whiten，再做 RQKMeans。
4. `pca96_nowhiten`: PCA 降到 96 维，不 whiten，再做 RQKMeans。
5. `pca64_whiten`: PCA 降到 64 维，whiten，再做 RQKMeans。
6. `RQVAE`: 加载已有 RQVAE checkpoint 编码，作为 neural quantization 对照。

固定公平性设置：

```text
embedding_column = normalized_embed
n_levels = 4
codebook_size = 256
kmeans_backend = faiss.Kmeans
```

当前数据规模已确认：

```text
/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet
rows = 7,721,749
embedding_dim = 128
row_groups = 30
```

## 目录结构

```text
common/
  io.py          # parquet 流式读取、reservoir sample、json/io
  pca.py         # streaming PCA: mean + covariance + eigh
  rqkmeans.py    # FAISS Residual KMeans
  metrics.py     # utilization, entropy, collision, recon, locality

rawRQKmeans/
  run_raw_rqkmeans.py

PCA_RQKmeans/
  run_pca_rqkmeans.py

RQVAE/
  model.py
  encoder.py
  run_rqvae_diagnostics.py

configs/
  variants.json

run_all.py
artifacts/
```

## 快速 smoke test

先用小样本确认代码链路：

```bash
cd /Users/Toryck/Coding/Regret_Yambda/0408Yambda/PCA+RQKmeans

python run_all.py \
  --max_rows 2000 \
  --train_rows 1000 \
  --kmeans_iter 2 \
  --batch_size 512 \
  --locality_sample_size 500 \
  --locality_query_size 20 \
  --locality_topk 3 \
  --quiet_faiss \
  --only raw pca32_nowhiten
```

## 全量运行

全量语义 ID 评估：

```bash
cd /Users/Toryck/Coding/Regret_Yambda/0408Yambda/PCA+RQKmeans

python run_all.py \
  --embeddings /Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet \
  --max_rows 0 \
  --train_rows 0 \
  --batch_size 65536 \
  --kmeans_iter 50 \
  --locality_sample_size 200000 \
  --locality_query_size 5000 \
  --locality_topk 10 \
  --quiet_faiss \
  --only raw pca32_nowhiten pca64_nowhiten pca96_nowhiten pca64_whiten
```

说明：

- `--max_rows 0` 表示编码和评估全量 7,721,749 条。
- `--train_rows 0` 表示 codebook 训练也使用当前有效全集。
- 如果全量 KMeans 太慢，可以先设 `--train_rows 500000`，但 raw 和 PCA 变体必须用同一个值。

RQVAE 需要 checkpoint：

```bash
python run_all.py \
  --embeddings /Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet \
  --max_rows 0 \
  --rqvae_ckpt /path/to/best_entropy_e20.pth \
  --only rqvae
```

## 单独运行

Raw RQKMeans：

```bash
python rawRQKmeans/run_raw_rqkmeans.py \
  --embeddings /Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet \
  --out_dir artifacts/rawRQKmeans \
  --max_rows 0 \
  --train_rows 0 \
  --quiet_faiss
```

PCA-64 no-whiten：

```bash
python PCA_RQKmeans/run_pca_rqkmeans.py \
  --pca_dim 64 \
  --variant_name pca64_nowhiten \
  --out_root artifacts/PCA_RQKmeans \
  --max_rows 0 \
  --train_rows 0 \
  --quiet_faiss
```

PCA-64 whiten：

```bash
python PCA_RQKmeans/run_pca_rqkmeans.py \
  --pca_dim 64 \
  --whiten \
  --variant_name pca64_whiten \
  --out_root artifacts/PCA_RQKmeans \
  --max_rows 0 \
  --train_rows 0 \
  --quiet_faiss
```

RQVAE：

```bash
python RQVAE/run_rqvae_diagnostics.py \
  --rqvae_ckpt /path/to/best_entropy_e20.pth \
  --out_dir artifacts/RQVAE \
  --max_rows 0
```

## 输出文件

每个变体目录会写：

```text
codebook*.npz              # RQKMeans codebook 或 RQVAE latent codebooks
pca.npz                    # 仅 PCA 变体有
pca.meta.json              # 仅 PCA 变体有
item_ids.npy               # 与 sid.npy 行对齐的原始 item_id
sid.npy                    # shape = [num_items, 4]
locality_sample_embeddings.npy
locality_sample_sid.npy
metrics.json
meta.json
```

跑完多个变体后生成汇总表：

```bash
python summarize_metrics.py --artifacts artifacts
```

默认会跳过 `smoke_*` 调试目录。如果要汇总 smoke test：

```bash
python summarize_metrics.py --artifacts artifacts --include_smoke
```

输出：

```text
artifacts/summary/sid_variant_comparison.md
artifacts/summary/sid_variant_comparison.json
artifacts/summary/sid_variant_comparison.csv
```

如果加 `--write_dense_mapping`，额外输出：

```text
dense_item2sid.npy         # row 0 为 padding
dense2orig_item_id.npy     # row 0 为 -1
```

## 指标解释

`metrics.json` 包含：

```text
sid_distribution.per_level
```

每层码字利用率、熵、Gini、最大 token 占比。

```text
collision.prefix
collision.full_sid
```

prefix/full SID 碰撞。重点看 `full_sid.excess_collision_rate`。

```text
reconstruction
```

Raw RQKMeans 是原空间重构误差。

PCA 变体包含三项：

- `quantized_original_space`: PCA + RQKMeans 量化后 inverse PCA 回 128 维的误差。
- `pca_only_original_space`: 只做 PCA 压缩再 inverse PCA 的误差，用于区分 PCA 损失和 RQ 损失。
- `quantized_pca_space`: PCA 空间内部的 RQ 重构误差。

RQVAE 如果 checkpoint 有 decoder，则报告 `decode_from_codes` 到原空间的误差；否则只报告 SID 统计和 locality。

```text
sampled_locality
```

在随机采样池里用 FAISS 找 embedding 近邻，再看这些近邻的 SID 是否共享前缀。

重点看：

- `mean_sid_hamming`
- `mean_common_prefix_len`
- `same_level1_rate`
- `same_level12_rate`
- `same_full_sid_rate`

## 公平性注意

Raw 和 PCA 变体是严格控制变量对比：

```text
同一个 parquet
同一个 embedding column
同一个 FAISS RQKMeans 实现
同一个 n_levels/codebook_size
同一个 seed/kmeans_iter/train_rows
同一套 metrics
```

RQVAE 是 neural baseline，不是同构对照。报告里建议写：

```text
RQKMeans variants are controlled comparisons; RQVAE is a neural quantization baseline.
```
