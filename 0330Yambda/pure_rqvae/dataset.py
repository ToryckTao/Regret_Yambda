"""
数据加载模块：支持两种来源
  1. .npz 文件（推荐，生产级速度）
  2. .parquet 文件（首次需运行 preprocess.py 转换）
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """
    加载 yambda embeddings，支持两种格式：
      - .npz: { 'embeddings': (N, D), 'item_ids': (N,) }
      - .parquet: 嵌套 List<Double> 列，需要逐行展平

    Returns:
        (embedding, index)  —  index 是全局序号，用于对齐下游序列
    """

    def __init__(self, npz_path=None, parquet_path=None, use_normalized=True):
        if npz_path is not None:
            self._load_npz(npz_path)
        elif parquet_path is not None:
            self._load_parquet(parquet_path, use_normalized)
        else:
            raise ValueError("必须指定 npz_path 或 parquet_path")

        print(f"[EmbDataset] Loaded {len(self.embeddings)} items, dim={self.dim}")

    def _load_npz(self, path):
        data = np.load(path)
        self.embeddings = data['embeddings'].astype(np.float32)
        self.item_ids = data['item_ids']
        self.dim = self.embeddings.shape[1]

    def _load_parquet(self, path, use_normalized):
        import pyarrow.parquet as pq

        col = 'normalized_embed' if use_normalized else 'embed'
        pf = pq.ParquetFile(path)

        embeddings_list = []
        item_ids_list = []

        for i, row_group in enumerate(pf.iter_batches(batch_size=None, columns=['item_id', col])):
            df = row_group.to_pandas()
            for _, row in df.iterrows():
                arr = np.array(row[col], dtype=np.float32)
                embeddings_list.append(arr)
                item_ids_list.append(row['item_id'])
            print(f"  parquet row_group {i}: loaded {len(embeddings_list)} rows")

        self.embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
        self.item_ids = np.array(item_ids_list, dtype=np.uint32)
        self.dim = self.embeddings.shape[1]

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx]), idx

    def __len__(self):
        return len(self.embeddings)
