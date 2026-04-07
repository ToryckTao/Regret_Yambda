#!/usr/bin/env python3
"""
将 embeddings.parquet 预处理为 .npz 文件，大幅加速后续训练时的数据加载。
772 万条数据，约 2-3 分钟跑完，生成的 npz 约 4GB。
"""
import argparse
import time
import numpy as np
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Convert embeddings.parquet to .npz")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/embeddings.parquet",
        help="Path to embeddings.parquet"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/embeddings.npz",
        help="Output .npz path"
    )
    parser.add_argument(
        "--col",
        type=str,
        default="normalized_embed",
        choices=["embed", "normalized_embed"],
        help="Which embedding column to use"
    )
    parser.add_argument("--dtype", type=str, default="float32", help="numpy dtype")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    pf = pq.ParquetFile(args.input)
    n_rows = pf.metadata.num_rows
    print(f"[preprocess] Total rows: {n_rows}")

    embeddings_list = []
    item_ids = []
    dtype = np.dtype(args.dtype)

    for i, row_group in enumerate(pf.iter_batches(columns=['item_id', args.col])):
        df = row_group.to_pandas()
        for _, row in df.iterrows():
            arr = np.array(row[args.col], dtype=dtype)
            embeddings_list.append(arr)
            item_ids.append(row['item_id'])

        done = len(embeddings_list)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_rows - done) / rate if rate > 0 else 0
        print(f"  row_group {i}: {done}/{n_rows} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    embeddings = np.stack(embeddings_list, axis=0)  # (N, D)
    item_ids = np.array(item_ids, dtype=np.uint32)

    np.savez(args.output, embeddings=embeddings, item_ids=item_ids)

    elapsed = time.time() - t0
    size_gb = embeddings.nbytes / 1e9
    print(f"[preprocess] Done in {elapsed:.1f}s")
    print(f"  embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  item_ids shape:   {item_ids.shape}, dtype: {item_ids.dtype}")
    print(f"  file size:        ~{size_gb:.2f} GB")
    print(f"  saved to:         {args.output}")


if __name__ == "__main__":
    main()
