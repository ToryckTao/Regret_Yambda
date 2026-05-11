from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RegretEntry:
    sid_path: tuple[int, ...]
    phi: float
    regret_type_id: int
    item_id: int


class RegretMemoryPool:
    """Per-user sliding window of failed semantic paths."""

    def __init__(
        self,
        pool_size: int,
        sid_levels: int,
        gamma: float = 0.9,
        phi_scale: float = 1.0,
        phi_clip: float = 2.0,
        reward_threshold: float = 0.0,
        negative_type_ids: tuple[int, ...] = (1, 2, 3),
        min_phi: float = 1e-6,
    ) -> None:
        self.pool_size = int(pool_size)
        self.sid_levels = int(sid_levels)
        self.gamma = float(gamma)
        self.phi_scale = float(phi_scale)
        self.phi_clip = float(phi_clip)
        self.reward_threshold = float(reward_threshold)
        self.negative_type_ids = set(int(x) for x in negative_type_ids)
        self.min_phi = float(min_phi)
        self._pool: dict[int, list[RegretEntry]] = {}
        self.n_updates = 0
        self.n_inserted = 0

    def __len__(self) -> int:
        return sum(len(items) for items in self._pool.values())

    def _decay_user(self, user_id: int) -> None:
        entries = self._pool.get(user_id)
        if not entries:
            return
        kept: list[RegretEntry] = []
        for entry in entries:
            phi = float(entry.phi) * self.gamma
            if phi >= self.min_phi:
                kept.append(
                    RegretEntry(
                        sid_path=entry.sid_path,
                        phi=phi,
                        regret_type_id=entry.regret_type_id,
                        item_id=entry.item_id,
                    )
                )
        if kept:
            self._pool[user_id] = kept[: self.pool_size]
        else:
            self._pool.pop(user_id, None)

    def get(self, user_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_user_ids = [int(x) for x in user_ids.detach().cpu().view(-1).tolist()]
        tokens = torch.zeros(
            len(batch_user_ids),
            self.pool_size,
            self.sid_levels,
            dtype=torch.long,
            device=device,
        )
        phis = torch.zeros(len(batch_user_ids), self.pool_size, dtype=torch.float32, device=device)
        for row_idx, user_id in enumerate(batch_user_ids):
            for col_idx, entry in enumerate(self._pool.get(user_id, [])[: self.pool_size]):
                tokens[row_idx, col_idx] = torch.tensor(entry.sid_path[: self.sid_levels], dtype=torch.long, device=device)
                phis[row_idx, col_idx] = float(entry.phi)
        return tokens, phis

    def load_snapshot(
        self,
        user_id: int,
        sid_paths: torch.Tensor,
        phis: torch.Tensor,
        regret_type_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> int:
        """Load a precomputed per-user memory snapshot without applying online decay.

        This is used for random-access offline training: the split stage can scan
        each user chronologically and attach the user's historical B_rev before
        each row. Loading the snapshot here avoids order bias from shuffled
        dataloaders.
        """
        entries: list[RegretEntry] = []
        for sid_path, phi, regret_type_id, item_id in zip(sid_paths, phis, regret_type_ids, item_ids):
            phi_f = float(phi)
            item_id_i = int(item_id)
            if phi_f <= 0.0 or item_id_i <= 0:
                continue
            sid_tuple = tuple(int(x) for x in sid_path.detach().cpu().view(-1).tolist()[: self.sid_levels])
            if len(sid_tuple) != self.sid_levels:
                continue
            entries.append(
                RegretEntry(
                    sid_path=sid_tuple,
                    phi=min(self.phi_clip, max(0.0, phi_f)),
                    regret_type_id=int(regret_type_id),
                    item_id=item_id_i,
                )
            )
        if entries:
            self._pool[int(user_id)] = entries[: self.pool_size]
        else:
            self._pool.pop(int(user_id), None)
        return len(entries[: self.pool_size])

    def should_insert(self, regret_type_id: int, reward: float) -> bool:
        return int(regret_type_id) in self.negative_type_ids or float(reward) < self.reward_threshold

    def compute_phi(self, regret_strength: float, reward: float) -> float:
        raw_phi = float(regret_strength)
        if raw_phi <= 0.0:
            raw_phi = max(0.0, self.reward_threshold - float(reward))
        return min(self.phi_clip, max(0.0, raw_phi * self.phi_scale))

    def update(
        self,
        user_id: int,
        sid_path: torch.Tensor,
        regret_type_id: int,
        regret_strength: float,
        reward: float,
        item_id: int,
    ) -> bool:
        user_id = int(user_id)
        self.n_updates += 1
        self._decay_user(user_id)
        if self.pool_size <= 0 or not self.should_insert(int(regret_type_id), float(reward)):
            return False
        phi = self.compute_phi(float(regret_strength), float(reward))
        if phi <= 0.0:
            return False
        sid_tuple = tuple(int(x) for x in sid_path.detach().cpu().view(-1).tolist()[: self.sid_levels])
        entry = RegretEntry(
            sid_path=sid_tuple,
            phi=phi,
            regret_type_id=int(regret_type_id),
            item_id=int(item_id),
        )
        entries = [entry] + self._pool.get(user_id, [])
        self._pool[user_id] = entries[: self.pool_size]
        self.n_inserted += 1
        return True

    def active_user_count(self) -> int:
        return len(self._pool)

    def summary(self) -> dict[str, float]:
        phis = [entry.phi for entries in self._pool.values() for entry in entries]
        return {
            "active_users": float(self.active_user_count()),
            "entries": float(len(phis)),
            "n_updates": float(self.n_updates),
            "n_inserted": float(self.n_inserted),
            "mean_phi": float(sum(phis) / len(phis)) if phis else 0.0,
            "max_phi": float(max(phis)) if phis else 0.0,
        }
