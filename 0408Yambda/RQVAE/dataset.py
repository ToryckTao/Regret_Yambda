import numpy as np
import torch
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """仅加载 Semantic Embeddings。Returns: (embedding, index)"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.item_ids = data["item_ids"]
        self.dim = self.embeddings.shape[1]
        print(f"[EmbDataset] Loaded {len(self.embeddings)} items, dim={self.dim}")

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx]), idx

    def __len__(self):
        return len(self.embeddings)
