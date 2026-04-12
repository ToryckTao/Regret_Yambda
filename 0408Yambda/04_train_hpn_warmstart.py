#!/usr/bin/env python3
"""
最小 HPN warm-start 训练入口。

目标：
1. 不先接完整 actor-critic
2. 先验证原版 HSRL 的 SIDPolicy_credit 能否在 Yambda 上学会 target SID
3. 输入是历史 item 向量序列，输出是分层 SID token

训练信号：
- history_features -> sid_logits
- target_dense_item_id -> target_sid
- token-level cross entropy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
    parser = argparse.ArgumentParser(description="Warm-start HPN on Yambda split TSV files")
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/train.tsv"),
        help="训练集 TSV 路径",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/val.tsv"),
        help="验证集 TSV 路径",
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
        help="dense_item_id -> SID 的数组路径",
    )
    parser.add_argument(
        "--embedding_column",
        type=str,
        default="normalized_embed",
        choices=["embed", "normalized_embed"],
        help="用于构造 history_features 的向量列",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="历史窗口长度 H",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=0,
        help="训练集最多读取多少行。0 表示全量",
    )
    parser.add_argument(
        "--max_val_rows",
        type=int,
        default=0,
        help="验证集最多读取多少行。0 表示全量",
    )
    parser.add_argument(
        "--train_positive_only",
        action="store_true",
        help="只用 feedback_label=1 的训练样本做 warm-start",
    )
    parser.add_argument(
        "--min_train_reward",
        type=float,
        default=-1e9,
        help="训练集 reward 下限，小于该值的样本会被过滤",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="权重衰减",
    )
    parser.add_argument(
        "--sasrec_n_layer",
        type=int,
        default=2,
        help="原版 HPN/SASRec encoder 层数",
    )
    parser.add_argument(
        "--sasrec_d_model",
        type=int,
        default=64,
        help="原版 HPN hidden 维度",
    )
    parser.add_argument(
        "--sasrec_d_forward",
        type=int,
        default=128,
        help="Transformer FFN 维度",
    )
    parser.add_argument(
        "--sasrec_n_head",
        type=int,
        default=4,
        help="多头注意力头数",
    )
    parser.add_argument(
        "--sasrec_dropout",
        type=float,
        default=0.1,
        help="dropout",
    )
    parser.add_argument(
        "--sid_temp",
        type=float,
        default=1.0,
        help="SID softmax 温度",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker 数量。Mac 上建议先用 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="训练设备",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/hpn_warmstart.pt"),
        help="最佳模型保存路径",
    )
    parser.add_argument(
        "--save_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/hpn_warmstart.meta.json"),
        help="训练元信息保存路径",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """输入：设备字符串。输出：torch.device。"""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_list_cell(cell: object) -> list[int]:
    """输入：TSV 里的 list 字符串。输出：Python list[int]。"""
    if isinstance(cell, list):
        return [int(x) for x in cell]
    if isinstance(cell, str):
        if cell == "":
            return []
        value = ast.literal_eval(cell)
        return [int(x) for x in value]
    if pd.isna(cell):
        return []
    return [int(cell)]


def load_split_df(path: Path, max_rows: int) -> pd.DataFrame:
    """输入：split TSV 路径。输出：DataFrame。"""
    nrows = None if max_rows <= 0 else max_rows
    return pd.read_table(path, sep="\t", engine="python", nrows=nrows)


def collect_needed_dense_ids(df: pd.DataFrame) -> set[int]:
    """
    输入：split DataFrame
    输出：该数据集实际用到的 dense item id 集合
    """
    needed: set[int] = set()
    for _, row in df.iterrows():
        target_dense = int(row["target_dense_item_id"])
        if target_dense > 0:
            needed.add(target_dense)
        for iid in parse_list_cell(row["user_mid_history"]):
            if iid > 0:
                needed.add(iid)
        for iid in parse_list_cell(row.get("next_user_mid_history", "[]")):
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
    - embeddings parquet
    - orig2dense 映射
    - 本轮训练需要的 dense id 集合

    输出：
    - feature_cache: dense_item_id -> vec(128,)
    - cache_stats: 命中统计
    """
    feature_cache: dict[int, np.ndarray] = {0: np.zeros(128, dtype=np.float32)}
    if not needed_dense_ids:
        return feature_cache, {"needed": 0, "cached": 1, "missing": 0}

    remaining = set(int(x) for x in needed_dense_ids if x > 0)
    pf = pq.ParquetFile(parquet_path)
    rows_seen = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=4096, columns=["item_id", embedding_column]), start=1):
        pyd = batch.to_pydict()
        item_ids = pyd["item_id"]
        vectors = pyd[embedding_column]
        for orig_item_id, vec in zip(item_ids, vectors):
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

    if remaining:
        print(f"[cache] warning: {len(remaining):,} dense item ids not found in embeddings parquet")

    return feature_cache, {
        "needed": int(len(needed_dense_ids)),
        "cached": int(len(feature_cache) - 1),
        "missing": int(len(remaining)),
    }


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    """
    输入：dense_item2sid 数组。
    输出：
    - sid_levels
    - sid_vocab_size（假设各层相同）
    """
    valid_sid = dense_item2sid[1:]
    sid_levels = int(valid_sid.shape[1])
    vocab_sizes = [int(valid_sid[:, l].max()) + 1 for l in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Per-level vocab sizes are inconsistent: {vocab_sizes}")
    return sid_levels, vocab_sizes[0]


@dataclass
class SampleRecord:
    """单条 HPN 监督样本。"""
    history_ids: list[int]
    target_dense_item_id: int
    target_sid: list[int]
    reward: float
    feedback_label: int


class YambdaHPNDataset(Dataset):
    """
    把 split TSV 转成 HPN 监督样本。

    输入张量：
    - history_features: [H, 128]
    - target_sid: [L]
    - reward: scalar
    - feedback_label: scalar
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dense_item2sid: np.ndarray,
        feature_cache: dict[int, np.ndarray],
        max_seq_len: int,
        positive_only: bool = False,
        min_reward: float = -1e9,
    ) -> None:
        self.feature_cache = feature_cache
        self.max_seq_len = max_seq_len
        self.sid_levels = dense_item2sid.shape[1]
        self.records: list[SampleRecord] = []

        for _, row in df.iterrows():
            reward = float(row["user_clicks"])
            feedback_label = int(row.get("feedback_label", 0))
            if positive_only and feedback_label <= 0:
                continue
            if reward < min_reward:
                continue

            history_ids = parse_list_cell(row["user_mid_history"])
            history_ids = history_ids[-max_seq_len:]
            if len(history_ids) < max_seq_len:
                history_ids = [0] * (max_seq_len - len(history_ids)) + history_ids

            target_dense_item_id = int(row["target_dense_item_id"])
            target_sid = dense_item2sid[target_dense_item_id].tolist()
            if any(int(x) < 0 for x in target_sid):
                continue

            self.records.append(
                SampleRecord(
                    history_ids=history_ids,
                    target_dense_item_id=target_dense_item_id,
                    target_sid=[int(x) for x in target_sid],
                    reward=reward,
                    feedback_label=feedback_label,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        history_features = np.stack(
            [self.feature_cache.get(iid, self.feature_cache[0]) for iid in rec.history_ids],
            axis=0,
        ).astype(np.float32)
        return {
            "history_ids": torch.tensor(rec.history_ids, dtype=torch.long),
            "history_features": torch.tensor(history_features, dtype=torch.float32),
            "target_dense_item_id": torch.tensor(rec.target_dense_item_id, dtype=torch.long),
            "target_sid": torch.tensor(rec.target_sid, dtype=torch.long),
            "reward": torch.tensor(rec.reward, dtype=torch.float32),
            "feedback_label": torch.tensor(rec.feedback_label, dtype=torch.float32),
        }


class DummyYambdaEnvSpec:
    """
    给原版 SIDPolicy_credit 提供最小环境规格。

    它只需要：
    - action_space['item_id']
    - action_space['item_feature']
    - observation_space['history']
    """

    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def run_epoch(
    model: SIDPolicy_credit,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """
    输入：
    - model
    - loader
    - device
    - optimizer: 为 None 时表示验证

    输出：
    - loss / token_acc / full_path_acc 等统计
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_token_correct = None
    total_full_correct = 0
    sid_levels = model.sid_levels

    for batch_idx, batch in enumerate(loader, start=1):
        history_features = batch["history_features"].to(device)
        target_sid = batch["target_sid"].to(device)

        if batch_idx == 1:
            print(
                f"[batch-shape] history_features={tuple(history_features.shape)} "
                f"target_sid={tuple(target_sid.shape)}"
            )

        output = model({"history_features": history_features})
        sid_logits = output["sid_logits"]

        losses = []
        token_correct = []
        full_correct_mask = torch.ones(target_sid.shape[0], dtype=torch.bool, device=device)
        for l in range(sid_levels):
            logits_l = sid_logits[l]
            target_l = target_sid[:, l]
            losses.append(F.cross_entropy(logits_l, target_l))

            pred_l = torch.argmax(logits_l, dim=-1)
            correct_l = (pred_l == target_l)
            token_correct.append(int(correct_l.sum().item()))
            full_correct_mask &= correct_l

        loss = sum(losses) / len(losses)
        if "reg" in output:
            loss = loss + 1e-6 * output["reg"]

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = target_sid.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        if total_token_correct is None:
            total_token_correct = [0] * sid_levels
        for l in range(sid_levels):
            total_token_correct[l] += token_correct[l]
        total_full_correct += int(full_correct_mask.sum().item())

    token_acc = [
        (correct / total_samples) if total_samples > 0 else 0.0
        for correct in (total_token_correct or [0] * sid_levels)
    ]
    return {
        "loss": (total_loss / total_samples) if total_samples > 0 else 0.0,
        "full_path_acc": (total_full_correct / total_samples) if total_samples > 0 else 0.0,
        **{f"token_acc_l{l+1}": acc for l, acc in enumerate(token_acc)},
        "n_sample": float(total_samples),
    }


def main() -> None:
    """主入口：加载数据、建立 feature cache、训练原版 HPN。"""
    args = parse_args()
    save_path = Path(args.save_path)
    save_meta = Path(args.save_meta)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[device] using {device}")
    set_random_seed(args.seed)

    train_df = load_split_df(Path(args.train_file), args.max_train_rows)
    val_df = load_split_df(Path(args.val_file), args.max_val_rows)
    print(f"[data] train_rows={len(train_df):,}, val_rows={len(val_df):,}")

    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    print(f"[sid] sid_levels={sid_levels}, sid_vocab_size={sid_vocab_size}")

    needed_dense_ids = collect_needed_dense_ids(train_df) | collect_needed_dense_ids(val_df)
    print(f"[cache] unique_dense_ids_needed={len(needed_dense_ids):,}")
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r")
    feature_cache, cache_stats = build_feature_cache(
        parquet_path=Path(args.embeddings_parquet),
        embedding_column=args.embedding_column,
        orig2dense=orig2dense,
        needed_dense_ids=needed_dense_ids,
    )
    print(f"[cache] stats={cache_stats}")

    train_dataset = YambdaHPNDataset(
        df=train_df,
        dense_item2sid=dense_item2sid,
        feature_cache=feature_cache,
        max_seq_len=args.max_seq_len,
        positive_only=args.train_positive_only,
        min_reward=args.min_train_reward,
    )
    val_dataset = YambdaHPNDataset(
        df=val_df,
        dense_item2sid=dense_item2sid,
        feature_cache=feature_cache,
        max_seq_len=args.max_seq_len,
        positive_only=False,
        min_reward=-1e9,
    )
    print(f"[dataset] train_samples={len(train_dataset):,}, val_samples={len(val_dataset):,}")
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after filtering.")

    item_dim = next(v.shape[0] for k, v in feature_cache.items() if k != 0)
    env_spec = DummyYambdaEnvSpec(
        n_item=dense_item2sid.shape[0] - 1,
        item_dim=int(item_dim),
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
    model = SIDPolicy_credit(policy_args, env_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_full_acc={train_metrics['full_path_acc']:.4f} "
            f"val_full_acc={val_metrics['full_path_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "policy_args": vars(policy_args),
                    "cache_stats": cache_stats,
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )
            print(f"[save] best model -> {save_path}")

    meta = {
        "train_file": args.train_file,
        "val_file": args.val_file,
        "embeddings_parquet": args.embeddings_parquet,
        "orig2dense_npy": args.orig2dense_npy,
        "dense_item2sid_npy": args.dense_item2sid_npy,
        "embedding_column": args.embedding_column,
        "max_seq_len": int(args.max_seq_len),
        "train_positive_only": bool(args.train_positive_only),
        "min_train_reward": float(args.min_train_reward),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "device": str(device),
        "sid_levels": int(sid_levels),
        "sid_vocab_size": int(sid_vocab_size),
        "cache_stats": cache_stats,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "save_path": str(save_path),
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] training meta saved to {save_meta}")


if __name__ == "__main__":
    main()
