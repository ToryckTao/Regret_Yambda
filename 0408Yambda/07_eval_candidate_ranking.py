#!/usr/bin/env python3
"""
候选集排序评估。

用途：
- 在没有原文完整 simulator 设定时，给 Yambda-HSRL baseline 做离线 sanity check
- 每条样本构造 1 个真实 target + N 个随机负样本
- 用 Actor/HPN 的 SID token 分布给候选 item 打分
- 输出 HR@K / NDCG@K / MRR / target rank / token accuracy

注意：
- 这不是替代论文里的 Total Reward / Depth
- 它只回答：在离线 held-out target 上，Actor 是否能把真实 item 排到候选集前面
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YAMBA_DATA_DIR = Path(os.environ.get("YAMBA_DATA_DIR", "/Users/Toryck/Coding/DATASET/Yambda"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapter.bootstrap import install_hsrl_adapter  # noqa: E402

install_hsrl_adapter()

from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore  # noqa: E402
from utils import set_random_seed  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Evaluate candidate ranking for Yambda SID actor")
    parser.add_argument(
        "--eval_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/test.tsv"),
        help="评估 TSV，通常用 test 或 val split",
    )
    parser.add_argument(
        "--embeddings_parquet",
        type=str,
        default=str(DEFAULT_YAMBA_DATA_DIR / "embeddings.parquet"),
        help="Yambda embeddings.parquet 路径",
    )
    parser.add_argument(
        "--orig2dense_npy",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/mappings/yambda_orig2dense_item_id.npy"),
        help="orig_item_id -> dense_item_id 映射路径",
    )
    parser.add_argument(
        "--dense_item2sid_npy",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/mappings/yambda_dense_item2sid.npy"),
        help="dense_item_id -> SID 数组",
    )
    parser.add_argument(
        "--actor_checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/yambda_sid_actor"),
        help="Actor checkpoint。支持 06 保存的 state_dict 或 04 保存的 dict checkpoint",
    )
    parser.add_argument("--embedding_column", type=str, default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_eval_rows", type=int, default=0, help="最多评估多少行，0 表示全量")
    parser.add_argument("--num_negatives", type=int, default=99, help="每条样本随机负样本数")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument(
        "--save_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/candidate_ranking.meta.json"),
        help="评估结果输出路径",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """输入：设备字符串。输出：torch.device。"""
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def parse_list_cell(cell: object) -> list[int]:
    """输入：TSV 中的 list 字符串。输出：Python list[int]。"""
    if isinstance(cell, list):
        return [int(x) for x in cell]
    if isinstance(cell, str):
        if cell == "":
            return []
        return [int(x) for x in ast.literal_eval(cell)]
    if pd.isna(cell):
        return []
    return [int(cell)]


def load_eval_df(path: Path, max_rows: int) -> pd.DataFrame:
    """输入：评估 TSV 路径。输出：DataFrame。"""
    nrows = None if max_rows <= 0 else max_rows
    return pd.read_table(path, sep="\t", engine="python", nrows=nrows)


def collect_needed_dense_ids(df: pd.DataFrame) -> set[int]:
    """输入：评估 DataFrame。输出：history 中实际需要查 embedding 的 dense item id。"""
    needed: set[int] = set()
    for _, row in df.iterrows():
        for iid in parse_list_cell(row["user_mid_history"]):
            if iid > 0:
                needed.add(iid)
    return needed


def build_feature_cache(
    parquet_path: Path,
    embedding_column: str,
    orig2dense: np.ndarray,
    needed_dense_ids: set[int],
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """
    输入：
    - embeddings.parquet
    - orig2dense 映射
    - 评估历史里需要的 dense id

    输出：
    - feature_cache: dense_item_id -> item vector
    - cache_stats
    """
    feature_cache: dict[int, np.ndarray] = {0: np.zeros(128, dtype=np.float32)}
    remaining = set(int(x) for x in needed_dense_ids if x > 0)
    if not remaining:
        return feature_cache, {"needed": 0, "cached": 0, "missing": 0}

    pf = pq.ParquetFile(parquet_path)
    rows_seen = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=4096, columns=["item_id", embedding_column]), start=1):
        pyd = batch.to_pydict()
        for orig_item_id, vec in zip(pyd["item_id"], pyd[embedding_column]):
            rows_seen += 1
            orig_item_id = int(orig_item_id)
            if orig_item_id >= len(orig2dense):
                continue
            dense_item_id = int(orig2dense[orig_item_id])
            if dense_item_id in remaining:
                feature_cache[dense_item_id] = np.asarray(vec, dtype=np.float32)
                remaining.remove(dense_item_id)
        if batch_idx % 50 == 0:
            print(f"[cache] scanned_rows={rows_seen:,}, cached={len(feature_cache)-1:,}, remaining={len(remaining):,}")
        if not remaining:
            break
    return feature_cache, {
        "needed": int(len(needed_dense_ids)),
        "cached": int(len(feature_cache) - 1),
        "missing": int(len(remaining)),
    }


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    """输入：dense_item2sid 数组。输出：SID 层数和 vocab size。"""
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def sample_candidates(
    target_dense_id: int,
    n_item: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    输入：target item、catalog size、负样本数。
    输出：候选列表，第 0 个固定为正样本 target。
    """
    candidates = [int(target_dense_id)]
    used = {int(target_dense_id), 0}
    while len(candidates) < num_negatives + 1:
        neg = int(rng.integers(1, n_item + 1))
        if neg in used:
            continue
        used.add(neg)
        candidates.append(neg)
    return candidates


@dataclass
class RankingRecord:
    """单条候选排序评估样本。"""
    history_ids: list[int]
    target_dense_item_id: int
    candidate_ids: list[int]
    target_sid: list[int]


class RankingDataset(Dataset):
    """
    把评估 TSV 转成候选排序样本。

    输出张量：
    - history_features: [H, 128]
    - candidate_ids: [1 + num_negatives]
    - candidate_sid: [1 + num_negatives, L]
    - target_sid: [L]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dense_item2sid: np.ndarray,
        feature_cache: dict[int, np.ndarray],
        max_seq_len: int,
        num_negatives: int,
        seed: int,
    ) -> None:
        self.feature_cache = feature_cache
        self.max_seq_len = max_seq_len
        self.records: list[RankingRecord] = []
        rng = np.random.default_rng(seed)
        n_item = int(dense_item2sid.shape[0] - 1)

        for _, row in df.iterrows():
            target_dense_id = int(row["target_dense_item_id"])
            history_ids = parse_list_cell(row["user_mid_history"])[-max_seq_len:]
            if len(history_ids) < max_seq_len:
                history_ids = [0] * (max_seq_len - len(history_ids)) + history_ids
            candidates = sample_candidates(target_dense_id, n_item, num_negatives, rng)
            target_sid = dense_item2sid[target_dense_id].tolist()
            self.records.append(
                RankingRecord(
                    history_ids=history_ids,
                    target_dense_item_id=target_dense_id,
                    candidate_ids=candidates,
                    target_sid=[int(x) for x in target_sid],
                )
            )
        self.dense_item2sid = dense_item2sid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        history_features = np.stack(
            [self.feature_cache.get(iid, self.feature_cache[0]) for iid in rec.history_ids],
            axis=0,
        ).astype(np.float32)
        candidate_ids = np.asarray(rec.candidate_ids, dtype=np.int64)
        candidate_sid = np.asarray(self.dense_item2sid[candidate_ids], dtype=np.int64)
        return {
            "history_features": torch.tensor(history_features, dtype=torch.float32),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "candidate_sid": torch.tensor(candidate_sid, dtype=torch.long),
            "target_sid": torch.tensor(rec.target_sid, dtype=torch.long),
        }


class DummyYambdaEnvSpec:
    """给 SIDPolicy_credit 提供最小 environment spec。"""

    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def load_actor(actor: SIDPolicy_credit, checkpoint_path: Path, device: torch.device) -> bool:
    """输入：actor 和 checkpoint 路径。输出：是否成功加载。"""
    if not checkpoint_path.exists():
        print(f"[actor] checkpoint not found, use random init: {checkpoint_path}")
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    print(f"[actor] loaded {checkpoint_path}, missing={len(missing)}, unexpected={len(unexpected)}")
    return True


def score_candidates(sid_logits: list[torch.Tensor], candidate_sid: torch.Tensor) -> torch.Tensor:
    """
    输入：
    - sid_logits: list of [B, V]
    - candidate_sid: [B, C, L]

    输出：
    - scores: [B, C]
    """
    scores = torch.zeros(candidate_sid.shape[:2], dtype=torch.float32, device=candidate_sid.device)
    for level, logits in enumerate(sid_logits):
        log_probs = torch.log_softmax(logits, dim=-1)
        idx = candidate_sid[:, :, level]
        scores = scores + torch.gather(log_probs, 1, idx)
    return scores


def evaluate(
    actor: SIDPolicy_credit,
    loader: DataLoader,
    device: torch.device,
    k_list: list[int],
) -> dict[str, float]:
    """输入：actor、loader、设备、K 列表。输出：排序指标。"""
    actor.eval()
    n = 0
    mrr_sum = 0.0
    mean_rank_sum = 0.0
    hr_sum = {k: 0.0 for k in k_list}
    ndcg_sum = {k: 0.0 for k in k_list}
    token_correct = None
    full_correct = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="[eval]", ncols=80)
        for batch in pbar:
            history_features = batch["history_features"].to(device)
            candidate_sid = batch["candidate_sid"].to(device)
            target_sid = batch["target_sid"].to(device)
            output = actor({"history_features": history_features})
            sid_logits = output["sid_logits"]
            scores = score_candidates(sid_logits, candidate_sid)

            target_score = scores[:, 0:1]
            ranks = (scores > target_score).sum(dim=1) + 1
            batch_size = int(ranks.shape[0])
            n += batch_size
            mean_rank_sum += float(ranks.float().sum().item())
            mrr_sum += float((1.0 / ranks.float()).sum().item())

            for k in k_list:
                hit = (ranks <= k).float()
                hr_sum[k] += float(hit.sum().item())
                ndcg = torch.where(ranks <= k, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks.float()))
                ndcg_sum[k] += float(ndcg.sum().item())

            if token_correct is None:
                token_correct = [0] * len(sid_logits)
            full_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            for level, logits in enumerate(sid_logits):
                pred = torch.argmax(logits, dim=-1)
                correct = pred == target_sid[:, level]
                token_correct[level] += int(correct.sum().item())
                full_mask &= correct
            full_correct += int(full_mask.sum().item())
            pbar.set_postfix_str(f"n={n:,} mr={mean_rank_sum/max(n,1):.1f} mrr={mrr_sum/max(n,1):.4f}", refresh=True)
        pbar.close()

    metrics: dict[str, float] = {
        "n_eval": float(n),
        "mean_rank": mean_rank_sum / max(n, 1),
        "mrr": mrr_sum / max(n, 1),
        "full_path_acc": full_correct / max(n, 1),
    }
    for level, correct in enumerate(token_correct or []):
        metrics[f"token_acc_l{level+1}"] = correct / max(n, 1)
    for k in k_list:
        metrics[f"hr@{k}"] = hr_sum[k] / max(n, 1)
        metrics[f"ndcg@{k}"] = ndcg_sum[k] / max(n, 1)
    return metrics


def main() -> None:
    """主入口：构造候选集并评估 Actor 排序能力。"""
    args = parse_args()
    save_meta = Path(args.save_meta)
    save_meta.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[device] using {device}")
    set_random_seed(args.seed)

    eval_df = load_eval_df(Path(args.eval_file), args.max_eval_rows)
    print(f"[data] eval_rows={len(eval_df):,}")
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    print(f"[sid] sid_levels={sid_levels}, sid_vocab_size={sid_vocab_size}")

    needed_dense_ids = collect_needed_dense_ids(eval_df)
    print(f"[cache] unique_history_dense_ids_needed={len(needed_dense_ids):,}")
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r")
    feature_cache, cache_stats = build_feature_cache(
        Path(args.embeddings_parquet),
        args.embedding_column,
        orig2dense,
        needed_dense_ids,
    )
    print(f"[cache] stats={cache_stats}")

    dataset = RankingDataset(
        eval_df,
        dense_item2sid=dense_item2sid,
        feature_cache=feature_cache,
        max_seq_len=args.max_seq_len,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    env_spec = DummyYambdaEnvSpec(
        n_item=dense_item2sid.shape[0] - 1,
        item_dim=128,
        max_seq_len=args.max_seq_len,
    )
    policy_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
    )
    actor = SIDPolicy_credit(policy_args, env_spec).to(device)
    actor_loaded = load_actor(actor, Path(args.actor_checkpoint), device)
    metrics = evaluate(actor, loader, device, args.k_list)
    print(f"[metrics] {metrics}")

    meta = {
        "eval_file": args.eval_file,
        "actor_checkpoint": args.actor_checkpoint,
        "actor_loaded": actor_loaded,
        "num_negatives": int(args.num_negatives),
        "candidate_size": int(args.num_negatives + 1),
        "max_eval_rows": int(args.max_eval_rows),
        "sid_levels": int(sid_levels),
        "sid_vocab_size": int(sid_vocab_size),
        "cache_stats": cache_stats,
        "metrics": metrics,
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] eval meta saved to {save_meta}")


if __name__ == "__main__":
    main()
