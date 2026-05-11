from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from regret_core.data.schema import EVENT_TYPE_TO_ID, PRIOR_STAT_COLUMNS, REGRET_TYPE_TO_ID


def list_parquet_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.parquet"))


def pad_left(values: list, length: int, pad_value):
    values = list(values)[-length:]
    if len(values) < length:
        values = [pad_value] * (length - len(values)) + values
    return values


def play_bucket_id(play_ratio: float) -> int:
    if play_ratio <= 0.0:
        return 0
    if play_ratio <= 0.2:
        return 1
    if play_ratio <= 0.8:
        return 2
    return 3


class TransitionIterableDataset(IterableDataset):
    """Streaming transition dataset backed by parquet shards and dense feature mmap."""

    def __init__(
        self,
        parquet_path: str | Path,
        dense_item_features_npy: str | Path,
        max_seq_len: int = 50,
        max_rows: int = 0,
        reward_column: str = "reward_scaled",
        batch_size: int = 2048,
        shuffle_files: bool = False,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
        sample_across_files: bool = False,
        balance_users: bool = False,
        balance_negative_types: bool = False,
        negative_share: float = 0.5,
        regret_memory_size: int = 20,
    ) -> None:
        super().__init__()
        self.files = list_parquet_files(parquet_path)
        if not self.files:
            raise FileNotFoundError(f"No parquet shards found under {parquet_path}")
        self.features = np.load(dense_item_features_npy, mmap_mode="r")
        self.max_seq_len = int(max_seq_len)
        self.max_rows = int(max_rows)
        self.reward_column = reward_column
        self.batch_size = int(batch_size)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.sample_across_files = bool(sample_across_files)
        self.balance_users = bool(balance_users)
        self.balance_negative_types = bool(balance_negative_types)
        self.negative_share = float(min(max(negative_share, 0.0), 1.0))
        self.regret_memory_size = int(regret_memory_size)
        self._iteration = 0
        self.item_dim = int(self.features.shape[1])
        self.prior_dim = len(PRIOR_STAT_COLUMNS)

    def _make_record(self, row: dict) -> dict[str, torch.Tensor]:
        history_ids = [int(x) for x in (row.get("history_item_ids") or [])]
        history_feedbacks = [float(x) for x in (row.get("history_feedbacks") or [])]
        history_event_type_ids = [int(x) for x in (row.get("history_event_type_ids") or [])]
        history_ids = pad_left(history_ids, self.max_seq_len, 0)
        history_feedbacks = pad_left(history_feedbacks, self.max_seq_len, 0.0)
        history_event_type_ids = pad_left(history_event_type_ids, self.max_seq_len, 0)
        history_ids_np = np.asarray(history_ids, dtype=np.int64)
        history_event_type_ids_np = np.asarray(history_event_type_ids, dtype=np.int64)
        user_id = int(row.get("user_id", 0) or 0)
        target_dense_item_id = int(row["target_dense_item_id"])
        prior_stats = np.asarray([float(row.get(col, 0.0) or 0.0) for col in PRIOR_STAT_COLUMNS], dtype=np.float32)
        regret_memory_item_ids = [
            int(x) for x in (row.get("regret_memory_item_ids") or [])
        ]
        regret_memory_phis = [
            float(x) for x in (row.get("regret_memory_phis") or [])
        ]
        regret_memory_type_ids = [
            int(x) for x in (row.get("regret_memory_type_ids") or [])
        ]
        regret_memory_item_ids = pad_left(regret_memory_item_ids, self.regret_memory_size, 0)
        regret_memory_phis = pad_left(regret_memory_phis, self.regret_memory_size, 0.0)
        regret_memory_type_ids = pad_left(regret_memory_type_ids, self.regret_memory_size, 0)
        history_mask = (history_ids_np > 0).astype(np.float32)
        effective_like = float(row.get("effective_like", row.get("has_like", 0.0)) or 0.0)
        effective_dislike = float(row.get("effective_dislike", row.get("has_dislike", 0.0)) or 0.0)
        effective_unlike = float(row.get("effective_unlike", row.get("has_unlike", 0.0)) or 0.0)
        effective_undislike = float(row.get("effective_undislike", row.get("has_undislike", 0.0)) or 0.0)
        has_like = float(row.get("has_like", effective_like) or 0.0)
        has_dislike = float(row.get("has_dislike", effective_dislike) or 0.0)
        has_unlike = float(row.get("has_unlike", effective_unlike) or 0.0)
        has_undislike = float(row.get("has_undislike", effective_undislike) or 0.0)
        feedback_targets = np.asarray(
            [effective_like, effective_dislike, effective_unlike, effective_undislike],
            dtype=np.float32,
        )
        play_target = float(np.clip(float(row.get("max_play_ratio", 0.0) or 0.0), 0.0, 1.0))
        listen_target = float((int(row.get("n_listen", 0) or 0) > 0))
        play_bucket = int(play_bucket_id(play_target))
        play_ordinal_targets = np.asarray(
            [
                float(play_bucket >= 1),
                float(play_bucket >= 2),
                float(play_bucket >= 3),
            ],
            dtype=np.float32,
        )
        play_ordinal_valid_mask = np.asarray(
            [
                1.0,
                float(play_bucket >= 1),
                float(play_bucket >= 2),
            ],
            dtype=np.float32,
        )
        has_like_history = bool(np.any(history_event_type_ids_np == EVENT_TYPE_TO_ID["like"]))
        has_dislike_history = bool(np.any(history_event_type_ids_np == EVENT_TYPE_TO_ID["dislike"]))
        feedback_valid_mask = np.asarray(
            [
                listen_target,
                listen_target,
                float(has_like_history or has_like > 0.0 or has_unlike > 0.0 or effective_unlike > 0.0),
                float(has_dislike_history or has_dislike > 0.0 or has_undislike > 0.0 or effective_undislike > 0.0),
            ],
            dtype=np.float32,
        )
        return {
            "history_ids": torch.tensor(history_ids_np, dtype=torch.long),
            "history_features": torch.tensor(self.features[history_ids_np].astype(np.float32), dtype=torch.float32),
            "history_feedbacks": torch.tensor(history_feedbacks, dtype=torch.float32),
            "history_event_type_ids": torch.tensor(history_event_type_ids_np, dtype=torch.long),
            "history_mask": torch.tensor(history_mask, dtype=torch.float32),
            "target_dense_item_id": torch.tensor(target_dense_item_id, dtype=torch.long),
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "action_features": torch.tensor(self.features[target_dense_item_id].astype(np.float32), dtype=torch.float32),
            "prior_stats": torch.tensor(prior_stats, dtype=torch.float32),
            "reward": torch.tensor(float(row[self.reward_column]), dtype=torch.float32),
            "listen_target": torch.tensor(listen_target, dtype=torch.float32),
            "play_target": torch.tensor(play_target, dtype=torch.float32),
            "play_bucket_id": torch.tensor(play_bucket, dtype=torch.long),
            "play_ordinal_targets": torch.tensor(play_ordinal_targets, dtype=torch.float32),
            "play_ordinal_valid_mask": torch.tensor(play_ordinal_valid_mask, dtype=torch.float32),
            "feedback_targets": torch.tensor(feedback_targets, dtype=torch.float32),
            "feedback_valid_mask": torch.tensor(feedback_valid_mask, dtype=torch.float32),
            "feedback_label": torch.tensor(float(row.get("feedback_label", 0.0) or 0.0), dtype=torch.float32),
            "regret_type_id": torch.tensor(
                int(REGRET_TYPE_TO_ID.get(str(row.get("regret_type", "none")), 0)),
                dtype=torch.long,
            ),
            "regret_strength": torch.tensor(float(row.get("regret_strength", 0.0) or 0.0), dtype=torch.float32),
            "regret_memory_item_ids": torch.tensor(regret_memory_item_ids, dtype=torch.long),
            "regret_memory_phis": torch.tensor(regret_memory_phis, dtype=torch.float32),
            "regret_memory_type_ids": torch.tensor(regret_memory_type_ids, dtype=torch.long),
        }

    def _iter_rows(self, files: list[Path], rng: np.random.Generator) -> Iterator[dict]:
        columns = [
            "history_item_ids",
            "history_feedbacks",
            "history_event_type_ids",
            "user_id",
            "target_dense_item_id",
            "reward_scaled",
            "reward_raw",
            "n_listen",
            "max_play_ratio",
            "effective_like",
            "effective_dislike",
            "effective_unlike",
            "effective_undislike",
            "has_like",
            "has_dislike",
            "has_unlike",
            "has_undislike",
            "feedback_label",
            "regret_type",
            "regret_strength",
            "regret_memory_item_ids",
            "regret_memory_phis",
            "regret_memory_type_ids",
            *PRIOR_STAT_COLUMNS,
        ]
        rows_per_file = 0
        if self.sample_across_files and self.max_rows > 0:
            rows_per_file = int(math.ceil(self.max_rows / max(len(files), 1)))
        for file_path in files:
            pf = pq.ParquetFile(file_path)
            schema_names = set(getattr(pf, "schema_arrow", pf.schema).names)
            present_columns = [col for col in columns if col in schema_names]
            if self.balance_negative_types and "regret_type" in present_columns:
                table = pf.read(columns=present_columns)
                row_indices = self._negative_type_balanced_row_indices(
                    table.column("regret_type").to_pylist(),
                    rng,
                    rows_per_file if rows_per_file > 0 else table.num_rows,
                    self.negative_share,
                )
                for row in table.take(row_indices).to_pylist():
                    yield row
                continue
            if self.balance_users and "user_id" in present_columns:
                table = pf.read(columns=present_columns)
                row_indices = self._balanced_row_indices(table.column("user_id").to_pylist(), rng, rows_per_file)
                for row in table.take(row_indices).to_pylist():
                    yield row
                continue
            if rows_per_file > 0:
                table = pf.read(columns=present_columns)
                take_n = min(rows_per_file, table.num_rows)
                row_indices = rng.choice(table.num_rows, size=take_n, replace=False)
                row_indices.sort()
                for row in table.take(row_indices).to_pylist():
                    yield row
                continue
            emitted_from_file = 0
            stop_file = False
            for batch in pf.iter_batches(batch_size=self.batch_size, columns=present_columns):
                for row in batch.to_pylist():
                    yield row
                    emitted_from_file += 1
                    if rows_per_file > 0 and emitted_from_file >= rows_per_file:
                        stop_file = True
                        break
                if stop_file:
                    break

    @staticmethod
    def _balanced_row_indices(user_ids: list, rng: np.random.Generator, max_take: int) -> np.ndarray:
        by_user: dict[int, list[int]] = {}
        for idx, user_id in enumerate(user_ids):
            try:
                user_key = int(user_id)
            except (TypeError, ValueError):
                # If user_id is missing in a row, treat that row as its own bucket
                # instead of crashing the balanced sampler.
                user_key = -(idx + 1)
            by_user.setdefault(user_key, []).append(idx)
        active_users = list(by_user.keys())
        for idxs in by_user.values():
            rng.shuffle(idxs)
        rng.shuffle(active_users)
        selected: list[int] = []
        target_n = int(max_take) if max_take > 0 else 0
        while active_users and (target_n <= 0 or len(selected) < target_n):
            next_active: list[int] = []
            for user_id in active_users:
                user_rows = by_user[user_id]
                if not user_rows:
                    continue
                selected.append(user_rows.pop())
                if target_n > 0 and len(selected) >= target_n:
                    break
                if user_rows:
                    next_active.append(user_id)
            if target_n > 0 and len(selected) >= target_n:
                break
            rng.shuffle(next_active)
            active_users = next_active
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _sample_indices(bucket: list[int], target_n: int, rng: np.random.Generator) -> list[int]:
        if target_n <= 0 or not bucket:
            return []
        if len(bucket) >= target_n:
            choice = rng.choice(np.asarray(bucket, dtype=np.int64), size=target_n, replace=False)
            return [int(x) for x in choice.tolist()]
        choice = rng.choice(np.asarray(bucket, dtype=np.int64), size=target_n, replace=True)
        return [int(x) for x in choice.tolist()]

    @staticmethod
    def _negative_type_balanced_row_indices(
        regret_types: list,
        rng: np.random.Generator,
        max_take: int,
        negative_share: float,
    ) -> np.ndarray:
        by_type: dict[int, list[int]] = {type_id: [] for type_id in REGRET_TYPE_TO_ID.values()}
        for idx, regret_type in enumerate(regret_types):
            type_id = REGRET_TYPE_TO_ID.get(str(regret_type), 0)
            by_type.setdefault(type_id, []).append(idx)

        total_rows = len(regret_types)
        target_n = int(max_take) if max_take > 0 else total_rows
        if target_n <= 0:
            return np.asarray([], dtype=np.int64)

        negative_type_ids = [1, 2, 3]
        available_negative_ids = [type_id for type_id in negative_type_ids if by_type.get(type_id)]
        if not available_negative_ids:
            return np.asarray(
                TransitionIterableDataset._sample_indices(by_type.get(0, []), target_n, rng),
                dtype=np.int64,
            )

        n_negative = int(round(target_n * float(negative_share)))
        n_negative = min(max(n_negative, len(available_negative_ids)), target_n)
        n_none = max(target_n - n_negative, 0)

        selected: list[int] = []
        selected.extend(TransitionIterableDataset._sample_indices(by_type.get(0, []), n_none, rng))

        per_type = n_negative // len(available_negative_ids)
        remainder = n_negative % len(available_negative_ids)
        for offset, type_id in enumerate(available_negative_ids):
            take_n = per_type + (1 if offset < remainder else 0)
            selected.extend(TransitionIterableDataset._sample_indices(by_type[type_id], take_n, rng))

        if len(selected) < target_n:
            pool = selected if selected else list(range(total_rows))
            extra = rng.choice(np.asarray(pool, dtype=np.int64), size=target_n - len(selected), replace=True)
            selected.extend(int(x) for x in extra.tolist())
        rng.shuffle(selected)
        return np.asarray(selected[:target_n], dtype=np.int64)

    def _iter_shuffled_rows(self, rows: Iterator[dict], rng: np.random.Generator) -> Iterator[dict]:
        buffer_size = int(self.shuffle_buffer_size)
        if buffer_size <= 1:
            yield from rows
            return
        buffer: list[dict] = []
        for row in rows:
            if len(buffer) < buffer_size:
                buffer.append(row)
                continue
            idx = int(rng.integers(0, len(buffer)))
            yield buffer[idx]
            buffer[idx] = row
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        files = list(self.files)
        if self.shuffle_files:
            rng.shuffle(files)
        emitted = 0
        rows = self._iter_shuffled_rows(self._iter_rows(files, rng), rng)
        for row in rows:
            yield self._make_record(row)
            emitted += 1
            if self.max_rows > 0 and emitted >= self.max_rows:
                return
