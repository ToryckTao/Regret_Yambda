"""
数据加载模块：同时加载 Semantic Embeddings 和 CF Embeddings。

Returns:
    (semantic_emb, cf_emb, index) — 供 LETTER 模型训练使用
    其中 cf_emb 来自预训练的 CF 模型（e.g., SASRec）。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class DualEmbDataset(Dataset):
    """
    同时加载两类 embedding：
      1. Semantic Embeddings（来自 LLM，如 LLaMA）— 用于 RQ-VAE
      2. CF Embeddings（来自预训练 CF 模型）— 用于 Collaborative Regularization

    两个 npz 的 item_ids 顺序必须一致（由 cf_trainer.py 保证）。
    """

    def __init__(
        self,
        npz_path: str,               # embeddings.npz 路径
        cf_npz_path: str = None,   # cf_embeddings.npz 路径
        cf_dim: int = 64,          # CF embedding 维度（cf_trainer.py 输出）
    ):
        # ── Semantic Embeddings ───────────────────────────────────────────
        sem_data = np.load(npz_path)
        self.sem_embeddings = sem_data['embeddings'].astype(np.float32)
        self.sem_item_ids = sem_data['item_ids']
        self.dim = self.sem_embeddings.shape[1]

        # ── CF Embeddings ─────────────────────────────────────────────────
        if cf_npz_path is not None:
            cf_data = np.load(cf_npz_path)
            cf_embs = cf_data['cf_embeddings'].astype(np.float32)
            cf_item_ids = cf_data['item_ids']

            # 对齐顺序（item_ids 一致性由 cf_trainer.py 保证）
            if len(cf_item_ids) != len(self.sem_item_ids):
                raise ValueError(
                    f"CF embeddings ({len(cf_item_ids)}) 和 "
                    f"Semantic embeddings ({len(self.sem_item_ids)}) 数量不一致"
                )

            # 如果维度不匹配，做投影或截断
            if cf_embs.shape[1] != cf_dim:
                if cf_embs.shape[1] > cf_dim:
                    cf_embs = cf_embs[:, :cf_dim]
                else:
                    pad = np.zeros((len(cf_embs), cf_dim - cf_embs.shape[1]), dtype=np.float32)
                    cf_embs = np.concatenate([cf_embs, pad], axis=1)

            self.cf_embeddings = cf_embs
        else:
            # 无 CF embeddings → Collaborative Regularization 被禁用
            self.cf_embeddings = None

        print(f"[DualEmbDataset] Loaded {len(self.sem_embeddings)} items")
        print(f"  Semantic dim: {self.dim}, CF dim: "
              f"{self.cf_embeddings.shape[1] if self.cf_embeddings is not None else 'N/A'}")

    def __getitem__(self, idx):
        sem = torch.from_numpy(self.sem_embeddings[idx])
        if self.cf_embeddings is not None:
            cf = torch.from_numpy(self.cf_embeddings[idx])
            return sem, cf, idx
        return sem, torch.zeros(1), idx  # placeholder cf

    def __len__(self):
        return len(self.sem_embeddings)


class EmbDataset(Dataset):
    """
    仅加载 Semantic Embeddings（向后兼容 pure_rqvae）。
    Returns: (embedding, index)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.embeddings = data['embeddings'].astype(np.float32)
        self.item_ids = data['item_ids']
        self.dim = self.embeddings.shape[1]
        print(f"[EmbDataset] Loaded {len(self.embeddings)} items, dim={self.dim}")

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx]), idx

    def __len__(self):
        return len(self.embeddings)
