#!/usr/bin/env python3
"""
训练 YambdaUserResponse，作为后续 RL 环境里的近似用户反馈模型。

当前文件做的事情：
1. 读取 03_split_data.py 产出的 train / val / test TSV
2. 从 embeddings.parquet 按需缓存本轮样本用到的 item 向量
3. 训练 YambdaUserResponse: history + exposure item -> reward(user_clicks)
4. 写出 checkpoint / log / meta，供下一阶段接环境时使用

注意：
- 这里不再使用 YambdaDataReader 全量读取 item_meta.tsv，因为 item_meta 会非常大。
- 这一步训练的是 user response，不产生推荐动作；推荐动作后面由 Actor + Facade 给出。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

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

from model.YambdaUserResponse import YambdaUserResponse  # type: ignore  # noqa: E402
import utils  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Train a lightweight YambdaUserResponse model")
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/train.tsv"),
        help="03_split_data.py 产出的训练 TSV",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/val.tsv"),
        help="03_split_data.py 产出的验证 TSV",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/test.tsv"),
        help="03_split_data.py 产出的测试 TSV。当前训练不读取，只写入 meta/log 方便后续环境复用",
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
        "--embedding_column",
        type=str,
        default="normalized_embed",
        choices=["embed", "normalized_embed"],
        help="用于训练 user response 的 item 向量列",
    )
    parser.add_argument("--embedding_dim", type=int, default=128, help="item embedding 维度")
    parser.add_argument("--max_seq_len", type=int, default=50, help="历史窗口长度 H")
    parser.add_argument("--max_train_rows", type=int, default=0, help="训练 TSV 最多读取行数，0 表示全量")
    parser.add_argument("--max_val_rows", type=int, default=0, help="验证 TSV 最多读取行数，0 表示全量")
    parser.add_argument("--train_sample_size", type=int, default=0, help="训练集抽样条数，0 表示不额外抽样")
    parser.add_argument("--val_sample_size", type=int, default=0, help="验证集抽样条数，0 表示不额外抽样")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epoch", type=int, default=5, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_worker", type=int, default=0, help="DataLoader worker 数量。Mac 上建议先用 0")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="训练设备",
    )
    parser.add_argument("--reader", type=str, default="YambdaFeatureCacheReader", help="后续环境日志里的 reader 名")
    parser.add_argument("--model", type=str, default="YambdaUserResponse", help="后续环境日志里的 model 名")

    parser = YambdaUserResponse.parse_model_args(parser)
    parser.set_defaults(
        loss="mse",
        l2_coef=0.0,
        feature_dim=32,
        attn_n_head=4,
        hidden_dims=[128],
        dropout_rate=0.2,
        model_path=str(PROJECT_ROOT / "artifacts/env/yambda_user_env.model"),
    )
    parser.add_argument(
        "--feature_cache_npz",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/yambda_user_env.feature_cache.npz"),
        help="按需 item 向量缓存输出，后续轻量 reader 可复用",
    )
    parser.add_argument(
        "--effective_train_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/yambda_user_env.train.tsv"),
        help="实际用于训练 user response 的 TSV 输出。后续轻量 reader 会读取它",
    )
    parser.add_argument(
        "--effective_val_file",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/yambda_user_env.val.tsv"),
        help="实际用于验证 user response 的 TSV 输出。后续轻量 reader 会读取它",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/log/yambda_user_env.model.log"),
        help="环境模型描述日志路径",
    )
    parser.add_argument(
        "--save_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/yambda_user_env.meta.json"),
        help="训练元信息输出路径",
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
    """输入：TSV 中的 list 字符串。输出：Python list[int]。"""
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
    """输入：split TSV 路径和最大行数。输出：DataFrame。"""
    nrows = None if max_rows <= 0 else max_rows
    return pd.read_table(path, sep="\t", engine="python", nrows=nrows)


def maybe_subsample_df(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """输入：DataFrame、抽样数、随机种子。输出：抽样后的 DataFrame。"""
    if sample_size <= 0 or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def collect_needed_dense_ids(df: pd.DataFrame) -> set[int]:
    """
    输入：split DataFrame。
    输出：该数据集实际需要查向量的 dense item id 集合。
    """
    needed: set[int] = set()
    for _, row in df.iterrows():
        target_dense = int(row["target_dense_item_id"])
        if target_dense > 0:
            needed.add(target_dense)
        for iid in parse_list_cell(row["user_mid_history"]):
            if iid > 0:
                needed.add(iid)
    return needed


def build_feature_cache(
    parquet_path: Path,
    embedding_column: str,
    orig2dense: np.ndarray,
    needed_dense_ids: set[int],
    embedding_dim: int,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """
    输入：
    - embeddings.parquet
    - orig2dense 映射
    - 本轮训练需要的 dense item id

    输出：
    - feature_cache: dense_item_id -> item vector
    - cache_stats: 缓存统计
    """
    feature_cache: dict[int, np.ndarray] = {0: np.zeros(embedding_dim, dtype=np.float32)}
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

    if remaining:
        print(f"[cache] warning: {len(remaining):,} dense item ids not found in embeddings parquet")
    return feature_cache, {
        "needed": int(len(needed_dense_ids)),
        "cached": int(len(feature_cache) - 1),
        "missing": int(len(remaining)),
    }


def save_feature_cache_npz(feature_cache: dict[int, np.ndarray], out_path: Path) -> None:
    """输入：feature_cache。输出：压缩 npz，包含 dense_ids 和 features。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dense_ids = np.asarray(sorted(k for k in feature_cache.keys() if k != 0), dtype=np.int64)
    features = np.stack([feature_cache[int(i)] for i in dense_ids], axis=0).astype(np.float32)
    np.savez_compressed(out_path, dense_ids=dense_ids, features=features)


def save_effective_split(df: pd.DataFrame, out_path: Path) -> None:
    """输入：实际使用的 DataFrame。输出：后续 reader 可复用的 TSV。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


@dataclass
class URMSample:
    """单条 user response 训练样本。"""
    history_ids: list[int]
    target_dense_item_id: int
    reward: float


class YambdaURMDataset(Dataset):
    """
    把 split TSV 转成 YambdaUserResponse 的训练样本。

    输出张量：
    - history: [H]
    - history_features: [H, 128]
    - exposure: [1]
    - exposure_features: [1, 128]
    - feedback: [1]
    - user_profile: [1]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cache: dict[int, np.ndarray],
        max_seq_len: int,
    ) -> None:
        self.feature_cache = feature_cache
        self.max_seq_len = max_seq_len
        self.records: list[URMSample] = []

        for _, row in df.iterrows():
            history_ids = parse_list_cell(row["user_mid_history"])[-max_seq_len:]
            if len(history_ids) < max_seq_len:
                history_ids = [0] * (max_seq_len - len(history_ids)) + history_ids
            self.records.append(
                URMSample(
                    history_ids=history_ids,
                    target_dense_item_id=int(row["target_dense_item_id"]),
                    reward=float(row["user_clicks"]),
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
        exposure_features = self.feature_cache.get(rec.target_dense_item_id, self.feature_cache[0]).reshape(1, -1)
        hist_length = int(sum(1 for iid in rec.history_ids if iid != 0))
        history_mask = np.asarray([0] * (self.max_seq_len - hist_length) + [1] * hist_length, dtype=np.float32)
        return {
            "history": torch.tensor(rec.history_ids, dtype=torch.long),
            "history_features": torch.tensor(history_features, dtype=torch.float32),
            "history_length": torch.tensor(hist_length, dtype=torch.long),
            "history_mask": torch.tensor(history_mask, dtype=torch.float32),
            "exposure": torch.tensor([rec.target_dense_item_id], dtype=torch.long),
            "exposure_features": torch.tensor(exposure_features, dtype=torch.float32),
            "feedback": torch.tensor([rec.reward], dtype=torch.float32),
            "user_profile": torch.zeros(1, dtype=torch.float32),
        }


class ReaderStub:
    """只给 YambdaUserResponse 提供 get_statistics() 所需信息。"""

    def __init__(self, item_vec_size: int, max_seq_len: int, n_train: int) -> None:
        self.item_vec_size = item_vec_size
        self.max_seq_len = max_seq_len
        self.n_train = n_train

    def get_statistics(self) -> dict[str, int]:
        """输入：无。输出：模型初始化需要的数据统计。"""
        return {
            "length": self.n_train,
            "n_item": 0,
            "item_vec_size": self.item_vec_size,
            "user_portrait_len": 0,
            "max_seq_len": self.max_seq_len,
            "n_feedback": 2,
        }


def evaluate_mse(model: YambdaUserResponse, loader: DataLoader, device: torch.device) -> float:
    """输入：模型、验证 loader、设备。输出：验证 MSE。"""
    total_se = 0.0
    total_n = 0
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            wrapped_batch = utils.wrap_batch(batch_data, device=str(device))
            out_dict = model.forward(wrapped_batch)
            preds = out_dict["preds"].view(-1).detach().cpu().numpy()
            target = wrapped_batch["feedback"].view(-1).detach().cpu().numpy()
            total_se += float(np.square(preds - target).sum())
            total_n += int(target.shape[0])
    return total_se / max(total_n, 1)


def write_env_log(log_path: Path, args: argparse.Namespace) -> None:
    """
    输入：日志路径和训练参数。
    输出：环境后续读取的两行头信息。

    注意：
    - 这里写的是 YambdaFeatureCacheReader；下一阶段需要在环境里接这个轻量 reader。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    class_args = Namespace(model=args.model, reader=args.reader)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"{class_args}\n")
        f.write(f"{args}\n")


def append_log_line(log_path: Path, text: str) -> None:
    """输入：日志路径和一行文本。输出：追加写入日志。"""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def main() -> None:
    """主入口：训练 YambdaUserResponse 并保存 checkpoint / log / meta。"""
    args = parse_args()
    model_path = Path(args.model_path)
    log_path = Path(args.log_path)
    save_meta = Path(args.save_meta)
    feature_cache_path = Path(args.feature_cache_npz)
    effective_train_file = Path(args.effective_train_file)
    effective_val_file = Path(args.effective_val_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    args.device = str(device)
    print(f"[device] using {device}")
    utils.set_random_seed(args.seed)

    train_df = load_split_df(Path(args.train_file), args.max_train_rows)
    val_df = load_split_df(Path(args.val_file), args.max_val_rows)
    train_df = maybe_subsample_df(train_df, args.train_sample_size, args.seed)
    val_df = maybe_subsample_df(val_df, args.val_sample_size, args.seed + 1)
    save_effective_split(train_df, effective_train_file)
    save_effective_split(val_df, effective_val_file)
    args.train_file = str(effective_train_file)
    args.val_file = str(effective_val_file)
    args.test_file = str(effective_val_file)
    args.max_train_rows = 0
    args.max_val_rows = 0
    print(f"[data] train_rows={len(train_df):,}, val_rows={len(val_df):,}")

    needed_dense_ids = collect_needed_dense_ids(train_df) | collect_needed_dense_ids(val_df)
    print(f"[cache] unique_dense_ids_needed={len(needed_dense_ids):,}")
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r")
    feature_cache, cache_stats = build_feature_cache(
        parquet_path=Path(args.embeddings_parquet),
        embedding_column=args.embedding_column,
        orig2dense=orig2dense,
        needed_dense_ids=needed_dense_ids,
        embedding_dim=args.embedding_dim,
    )
    save_feature_cache_npz(feature_cache, feature_cache_path)
    print(f"[cache] stats={cache_stats}")
    print(f"[cache] saved -> {feature_cache_path}")

    train_dataset = YambdaURMDataset(train_df, feature_cache, args.max_seq_len)
    val_dataset = YambdaURMDataset(val_df, feature_cache, args.max_seq_len)
    print(f"[dataset] train_samples={len(train_dataset):,}, val_samples={len(val_dataset):,}")
    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty.")

    reader_stub = ReaderStub(
        item_vec_size=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        n_train=len(train_dataset),
    )
    model = YambdaUserResponse(args, reader_stub, str(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer

    write_env_log(log_path, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=args.n_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=args.n_worker,
    )

    best_val_mse = float("inf")
    history = []
    for epo in range(1, args.epoch + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"[epoch {epo}] train", ncols=80)
        for step_idx, batch_data in enumerate(pbar, start=1):
            optimizer.zero_grad()
            wrapped_batch = utils.wrap_batch(batch_data, device=str(device))
            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict["loss"]
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            pbar.set_postfix_str(f"loss={np.mean(epoch_losses[-50:]):.6f}", refresh=True)
        pbar.close()

        val_mse = evaluate_mse(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        record = {
            "epoch": epo,
            "train_loss": train_loss,
            "val_mse": float(val_mse),
        }
        history.append(record)
        print(f"[epoch {epo}] train_loss={train_loss:.6f} val_mse={val_mse:.6f}")
        append_log_line(log_path, f"[epoch {epo}] train_loss={train_loss:.6f} val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = float(val_mse)
            model.save_checkpoint()
            print(f"[save] best checkpoint -> {model_path}.checkpoint")
            append_log_line(log_path, f"[save] best_checkpoint={model_path}.checkpoint val_mse={val_mse:.6f}")

    meta = {
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "embeddings_parquet": args.embeddings_parquet,
        "orig2dense_npy": args.orig2dense_npy,
        "embedding_column": args.embedding_column,
        "feature_cache_npz": args.feature_cache_npz,
        "effective_train_file": args.effective_train_file,
        "effective_val_file": args.effective_val_file,
        "model_path": args.model_path,
        "log_path": args.log_path,
        "reader": args.reader,
        "model": args.model,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "epoch": int(args.epoch),
        "max_seq_len": int(args.max_seq_len),
        "max_train_rows": int(args.max_train_rows),
        "max_val_rows": int(args.max_val_rows),
        "train_sample_size": int(args.train_sample_size),
        "val_sample_size": int(args.val_sample_size),
        "cache_stats": cache_stats,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "best_val_mse": float(best_val_mse),
        "history": history,
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] training meta saved to {save_meta}")


if __name__ == "__main__":
    main()
