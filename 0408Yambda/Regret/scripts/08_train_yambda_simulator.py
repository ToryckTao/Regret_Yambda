#!/usr/bin/env python3
"""主线 simulator 训练：从 transition parquet 监督学习用户反馈近似模型。

输入是用户历史状态和候选 item，输出 listen/play/feedback/negative 等反馈估计。
当前文件已合并旧 User-response trainer，删除 scripts/User-response 后仍可独立运行。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2] if SCRIPT_PATH.parent.name == "User-response" else SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.schema import EVENT_TYPE_TO_ID, ID_TO_REGRET_TYPE, PRIOR_STAT_COLUMNS, REGRET_TYPE_TO_ID  # noqa: E402
from regret_core.data.schema import RewardWeights  # noqa: E402
from regret_core.data.transition_dataset import TransitionIterableDataset, list_parquet_files, play_bucket_id  # noqa: E402
from regret_core.env.user_response_env import RegretUserResponseEnv  # noqa: E402
from regret_core.model.user_response import RegretUserResponse  # noqa: E402


DEFAULT_TRAIN_PATH = PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans/train"
DEFAULT_VAL_PATH = PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans/val"
DEFAULT_TEST_PATH = PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans/test"
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item_features.npy"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "artifacts/env/regret_user_response.pt"
DEFAULT_SAVE_META = PROJECT_ROOT / "artifacts/env/regret_user_response.meta.json"
DEFAULT_ENV_META = PROJECT_ROOT / "artifacts/env/regret_user_response.env.json"
FEEDBACK_LABELS = ["like", "dislike", "unlike", "undislike"]
REGRET_TYPE_NAMES = [ID_TO_REGRET_TYPE[idx] for idx in sorted(ID_TO_REGRET_TYPE)]
NEGATIVE_TYPE_LABELS = REGRET_TYPE_NAMES[1:]
PLAY_BUCKET_LABELS = ["zero", "low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Regret user response model")
    parser.add_argument("--transition_root", default="", help="Directory containing train/ val/ test/ transition shards.")
    parser.add_argument("--train_path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val_path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--test_path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--item_features_npy", default="", help="Alias for --dense_item_features_npy.")
    parser.add_argument("--dense_item_features_npy", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=200000)
    parser.add_argument("--max_test_rows", type=int, default=1000)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--shuffle_train_files", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=0)
    parser.add_argument("--train_sample_across_files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_balance_by_user", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_balance_negative_types", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_negative_share", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--history_heuristic_init", type=float, default=0.2)
    parser.add_argument("--decouple_reward_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--best_metric", default="auto", choices=["auto", "mse", "simulator_loss", "loss", "coarse_error"])
    parser.add_argument("--reward_weight_none", type=float, default=1.0)
    parser.add_argument("--reward_weight_low_play", type=float, default=1.0)
    parser.add_argument("--reward_weight_dislike", type=float, default=1.0)
    parser.add_argument("--reward_weight_unlike", type=float, default=1.0)
    parser.add_argument("--reward_loss_weight", type=float, default=1.0)
    parser.add_argument("--listen_loss_weight", type=float, default=0.1)
    parser.add_argument("--listen_pos_weight", type=float, default=1.0)
    parser.add_argument("--listen_loss_weight_decay", type=float, default=1.0)
    parser.add_argument("--play_loss_weight", type=float, default=0.2)
    parser.add_argument("--play_loss_weight_decay", type=float, default=0.7)
    parser.add_argument("--play_bucket_weight_zero", type=float, default=8.0)
    parser.add_argument("--play_bucket_weight_low", type=float, default=12.0)
    parser.add_argument("--play_bucket_weight_mid", type=float, default=3.0)
    parser.add_argument("--play_bucket_weight_high", type=float, default=1.0)
    parser.add_argument("--feedback_loss_weight", type=float, default=0.2)
    parser.add_argument("--feedback_loss_weight_decay", type=float, default=0.5)
    parser.add_argument("--feedback_pos_weight_like", type=float, default=1.0)
    parser.add_argument("--feedback_pos_weight_dislike", type=float, default=1.0)
    parser.add_argument("--feedback_pos_weight_unlike", type=float, default=1.0)
    parser.add_argument("--feedback_pos_weight_undislike", type=float, default=1.0)
    parser.add_argument("--feedback_threshold", type=float, default=0.5)
    parser.add_argument("--negative_pos_weight", type=float, default=1.0)
    parser.add_argument("--negative_type_focal_gamma", type=float, default=2.0)
    parser.add_argument("--init_head_bias_from_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior_estimate_rows", type=int, default=200000)
    parser.add_argument("--regret_loss_weight", type=float, default=0.0)
    parser.add_argument("--regret_class_weight_none", type=float, default=1.0)
    parser.add_argument("--regret_class_weight_low_play", type=float, default=1.0)
    parser.add_argument("--regret_class_weight_dislike", type=float, default=1.0)
    parser.add_argument("--regret_class_weight_unlike", type=float, default=1.0)
    parser.add_argument("--eval_only_checkpoint", default="", help="If set, load checkpoint and run val/test evaluation only.")
    parser.add_argument("--eval_test_after_train", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="", help="Directory for regret_user_response.pt/meta/env metadata.")
    parser.add_argument("--save_path", default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--save_meta", default=str(DEFAULT_SAVE_META))
    parser.add_argument("--env_meta", default=str(DEFAULT_ENV_META))
    args = parser.parse_args()
    if args.transition_root:
        root = Path(args.transition_root)
        args.train_path = str(root / "train")
        args.val_path = str(root / "val")
        args.test_path = str(root / "test")
    if args.item_features_npy:
        args.dense_item_features_npy = args.item_features_npy
    if args.out_dir:
        out_dir = Path(args.out_dir)
        args.save_path = str(out_dir / "regret_user_response.pt")
        args.save_meta = str(out_dir / "regret_user_response.meta.json")
        args.env_meta = str(out_dir / "regret_user_response.env.json")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def load_reward_config(args: argparse.Namespace) -> tuple[dict, str]:
    candidates: list[Path] = []
    if args.transition_root:
        candidates.append(Path(args.transition_root) / "split.meta.json")
    train_path = Path(args.train_path)
    if train_path.name in {"train", "val", "test"}:
        candidates.append(train_path.parent / "split.meta.json")
    else:
        candidates.append(train_path / "split.meta.json")
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reward_config = meta.get("reward_weights")
        if isinstance(reward_config, dict):
            return dict(reward_config), str(meta_path)
    return RewardWeights().__dict__.copy(), "default"


def build_loss_weight_tensors(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    listen_pos_weight = torch.tensor(float(args.listen_pos_weight), dtype=torch.float32, device=device)
    play_bucket_weights = torch.tensor(
        [
            args.play_bucket_weight_zero,
            args.play_bucket_weight_low,
            args.play_bucket_weight_mid,
            args.play_bucket_weight_high,
        ],
        dtype=torch.float32,
        device=device,
    )
    feedback_pos_weights = torch.tensor(
        [
            args.feedback_pos_weight_like,
            args.feedback_pos_weight_dislike,
            args.feedback_pos_weight_unlike,
            args.feedback_pos_weight_undislike,
        ],
        dtype=torch.float32,
        device=device,
    )
    negative_pos_weight = torch.tensor(float(args.negative_pos_weight), dtype=torch.float32, device=device)
    negative_type_weights = torch.tensor(
        [
            args.regret_class_weight_low_play,
            args.regret_class_weight_dislike,
            args.regret_class_weight_unlike,
        ],
        dtype=torch.float32,
        device=device,
    )
    return listen_pos_weight, play_bucket_weights, feedback_pos_weights, negative_pos_weight, negative_type_weights


def clipped_logit(prob: float, eps: float = 1e-4) -> float:
    prob = float(min(max(prob, eps), 1.0 - eps))
    return float(math.log(prob / (1.0 - prob)))


def estimate_train_label_priors(train_path: str, max_rows: int) -> dict[str, Any]:
    files = list_parquet_files(train_path)
    columns = [
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
    ]
    total_rows = 0
    listen_sum = 0.0
    play_bucket_counts = np.zeros(len(PLAY_BUCKET_LABELS), dtype=np.float64)
    feedback_sums = np.zeros(len(FEEDBACK_LABELS), dtype=np.float64)
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        present_columns = [col for col in columns if col in pf.schema.names]
        for batch in pf.iter_batches(batch_size=4096, columns=present_columns):
            rows = batch.to_pylist()
            for row in rows:
                listen_sum += float((int(row.get("n_listen", 0) or 0) > 0))
                play_bucket_counts[play_bucket_id(float(row.get("max_play_ratio", 0.0) or 0.0))] += 1.0
                feedback_sums[0] += float(row.get("effective_like", row.get("has_like", 0.0)) or 0.0)
                feedback_sums[1] += float(row.get("effective_dislike", row.get("has_dislike", 0.0)) or 0.0)
                feedback_sums[2] += float(row.get("effective_unlike", row.get("has_unlike", 0.0)) or 0.0)
                feedback_sums[3] += float(row.get("effective_undislike", row.get("has_undislike", 0.0)) or 0.0)
                total_rows += 1
                if max_rows > 0 and total_rows >= max_rows:
                    break
            if max_rows > 0 and total_rows >= max_rows:
                break
        if max_rows > 0 and total_rows >= max_rows:
            break
    denom = max(float(total_rows), 1.0)
    play_bucket_probs = (play_bucket_counts / play_bucket_counts.sum().clip(min=1.0)).tolist()
    feedback_rates = (feedback_sums / denom).tolist()
    priors = {
        "rows": int(total_rows),
        "listen_rate": float(listen_sum / denom),
        "play_bucket_probs": [float(x) for x in play_bucket_probs],
        "feedback_rates": [float(x) for x in feedback_rates],
        "negative_rate": 0.0,
        "negative_type_probs": [1.0 / max(len(NEGATIVE_TYPE_LABELS), 1)] * len(NEGATIVE_TYPE_LABELS),
    }
    regret_counts = np.zeros(len(REGRET_TYPE_NAMES), dtype=np.float64)
    columns = ["regret_type"]
    total_regret_rows = 0
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        present_columns = [col for col in columns if col in pf.schema.names]
        if not present_columns:
            continue
        for batch in pf.iter_batches(batch_size=4096, columns=present_columns):
            rows = batch.to_pylist()
            for row in rows:
                regret_counts[REGRET_TYPE_TO_ID.get(str(row.get("regret_type", "none")), 0)] += 1.0
                total_regret_rows += 1
                if max_rows > 0 and total_regret_rows >= max_rows:
                    break
            if max_rows > 0 and total_regret_rows >= max_rows:
                break
        if max_rows > 0 and total_regret_rows >= max_rows:
            break
    if total_regret_rows > 0:
        negative_total = float(regret_counts[1:].sum())
        priors["negative_rate"] = negative_total / float(total_regret_rows)
        if negative_total > 0:
            priors["negative_type_probs"] = [float(x / negative_total) for x in regret_counts[1:]]
    return priors


def initialize_head_bias_from_priors(model: RegretUserResponse, priors: dict[str, Any]) -> None:
    with torch.no_grad():
        model.listen_head[-1].bias.fill_(clipped_logit(float(priors["listen_rate"])))
        play_bucket_probs = np.asarray(priors["play_bucket_probs"], dtype=np.float32)
        play_bucket_probs = np.clip(play_bucket_probs, 1e-4, None)
        play_bucket_probs = play_bucket_probs / play_bucket_probs.sum()
        p_gt_zero = float(play_bucket_probs[1:].sum())
        p_gt_low = float(play_bucket_probs[2:].sum())
        p_gt_mid = float(play_bucket_probs[3])
        cond_probs = [
            p_gt_zero,
            (p_gt_low / p_gt_zero) if p_gt_zero > 1e-6 else 0.5,
            (p_gt_mid / p_gt_low) if p_gt_low > 1e-6 else 0.5,
        ]
        play_bias = torch.tensor(
            [clipped_logit(prob) for prob in cond_probs],
            dtype=torch.float32,
            device=model.play_head[-1].bias.device,
        )
        model.play_head[-1].bias.copy_(play_bias)
        feedback_bias = [clipped_logit(float(rate)) for rate in priors["feedback_rates"]]
        model.feedback_head[-1].bias.copy_(
            torch.tensor(feedback_bias, dtype=torch.float32, device=model.feedback_head[-1].bias.device)
        )
        model.negative_head[-1].bias.fill_(clipped_logit(float(priors["negative_rate"])))
        negative_type_probs = np.asarray(priors["negative_type_probs"], dtype=np.float32)
        negative_type_probs = np.clip(negative_type_probs, 1e-4, None)
        negative_type_probs = negative_type_probs / negative_type_probs.sum()
        negative_type_bias = torch.log(
            torch.tensor(negative_type_probs, dtype=torch.float32, device=model.negative_type_head[-1].bias.device)
        )
        model.negative_type_head[-1].bias.copy_(negative_type_bias)


def build_binary_stats() -> dict[str, float]:
    return {"n": 0.0, "prob_sum": 0.0, "pred_pos": 0.0, "true_pos": 0.0, "tp": 0.0}


def build_threshold_stats() -> dict[str, float]:
    return {"n": 0.0, "score_sum": 0.0, "pred_pos": 0.0, "true_pos": 0.0, "tp": 0.0}


def build_multiclass_stats(num_classes: int) -> dict[str, Any]:
    return {
        "n": 0.0,
        "correct": 0.0,
        "true_counts": [0.0] * num_classes,
        "pred_counts": [0.0] * num_classes,
        "tp_counts": [0.0] * num_classes,
        "prob_sums": [0.0] * num_classes,
    }


def build_type_stats() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in REGRET_TYPE_NAMES:
        out[name] = {
            "n": 0.0,
            "mse_sum": 0.0,
            "mae_sum": 0.0,
            "pred_sum": 0.0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
            "play_prob_sum": 0.0,
            "play_target_sum": 0.0,
            "play_mae_sum": 0.0,
            "feedback": {label: build_binary_stats() for label in FEEDBACK_LABELS},
        }
    return out


def safe_ratio(num: float, denom: float) -> float | None:
    return float(num / denom) if denom > 0 else None


def format_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "na"
    return f"{value:.{digits}f}"


def update_binary_bucket(
    bucket: dict[str, float],
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    valid_mask: torch.Tensor | None = None,
) -> None:
    if valid_mask is not None:
        valid = valid_mask >= 0.5
        if not bool(valid.any()):
            return
        probs = probs[valid]
        targets = targets[valid]
    pred_pos = (probs >= threshold).float()
    bucket["n"] += float(targets.numel())
    bucket["prob_sum"] += float(probs.sum().item())
    bucket["pred_pos"] += float(pred_pos.sum().item())
    bucket["true_pos"] += float(targets.sum().item())
    bucket["tp"] += float((pred_pos * targets).sum().item())


def finalize_binary_bucket(bucket: dict[str, float]) -> dict[str, float | None]:
    n = float(bucket["n"])
    pred_pos = float(bucket["pred_pos"])
    true_pos = float(bucket["true_pos"])
    tp = float(bucket["tp"])
    tn = max(n - pred_pos - true_pos + tp, 0.0)
    precision = safe_ratio(tp, pred_pos)
    recall = safe_ratio(tp, true_pos)
    specificity = safe_ratio(tn, n - true_pos)
    acc = safe_ratio(tp + tn, n)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced_acc = None
    if recall is not None and specificity is not None:
        balanced_acc = 0.5 * (recall + specificity)
    return {
        "n": int(n),
        "prob_mean": safe_ratio(bucket["prob_sum"], n),
        "pred_rate": safe_ratio(pred_pos, n),
        "true_rate": safe_ratio(true_pos, n),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "acc": acc,
        "f1": f1,
        "balanced_acc": balanced_acc,
    }


def update_threshold_bucket(
    bucket: dict[str, float],
    scores: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> None:
    pred_pos = (scores >= float(threshold)).float()
    bucket["n"] += float(targets.numel())
    bucket["score_sum"] += float(scores.sum().item())
    bucket["pred_pos"] += float(pred_pos.sum().item())
    bucket["true_pos"] += float(targets.sum().item())
    bucket["tp"] += float((pred_pos * targets).sum().item())


def finalize_threshold_bucket(bucket: dict[str, float]) -> dict[str, float | None]:
    n = float(bucket["n"])
    pred_pos = float(bucket["pred_pos"])
    true_pos = float(bucket["true_pos"])
    tp = float(bucket["tp"])
    tn = max(n - pred_pos - true_pos + tp, 0.0)
    precision = safe_ratio(tp, pred_pos)
    recall = safe_ratio(tp, true_pos)
    specificity = safe_ratio(tn, n - true_pos)
    acc = safe_ratio(tp + tn, n)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    balanced_acc = None
    if recall is not None and specificity is not None:
        balanced_acc = 0.5 * (recall + specificity)
    return {
        "n": int(n),
        "score_mean": safe_ratio(bucket["score_sum"], n),
        "pred_rate": safe_ratio(pred_pos, n),
        "true_rate": safe_ratio(true_pos, n),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "acc": acc,
        "f1": f1,
        "balanced_acc": balanced_acc,
    }


def update_multiclass_bucket(bucket: dict[str, Any], probs: torch.Tensor, targets: torch.Tensor) -> None:
    preds = probs.argmax(dim=-1)
    bucket["n"] += float(targets.shape[0])
    bucket["correct"] += float((preds == targets).float().sum().item())
    probs_cpu = probs.detach().cpu()
    preds_cpu = preds.detach().cpu()
    targets_cpu = targets.detach().cpu()
    for class_id in range(probs_cpu.shape[1]):
        bucket["prob_sums"][class_id] += float(probs_cpu[:, class_id].sum().item())
        bucket["true_counts"][class_id] += float((targets_cpu == class_id).sum().item())
        bucket["pred_counts"][class_id] += float((preds_cpu == class_id).sum().item())
        bucket["tp_counts"][class_id] += float(((preds_cpu == class_id) & (targets_cpu == class_id)).sum().item())


def finalize_multiclass_bucket(bucket: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    n = float(bucket["n"])
    out = {
        "n": int(n),
        "acc": safe_ratio(bucket["correct"], n),
        "classes": {},
    }
    for class_id, label in enumerate(labels):
        out["classes"][label] = {
            "prob_mean": safe_ratio(bucket["prob_sums"][class_id], n),
            "true_rate": safe_ratio(bucket["true_counts"][class_id], n),
            "pred_rate": safe_ratio(bucket["pred_counts"][class_id], n),
            "precision": safe_ratio(bucket["tp_counts"][class_id], bucket["pred_counts"][class_id]),
            "recall": safe_ratio(bucket["tp_counts"][class_id], bucket["true_counts"][class_id]),
        }
    return out


def finalize_type_buckets(type_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for type_name, bucket in type_stats.items():
        n = float(bucket["n"])
        if n <= 0:
            out[type_name] = {"n": 0}
            continue
        target_mean = bucket["target_sum"] / n
        mse = bucket["mse_sum"] / n
        mean_baseline_mse = max(bucket["target_sq_sum"] / n - target_mean * target_mean, 0.0)
        out[type_name] = {
            "n": int(n),
            "mse": mse,
            "mae": bucket["mae_sum"] / n,
            "pred_mean": bucket["pred_sum"] / n,
            "target_mean": target_mean,
            "mean_baseline_mse": mean_baseline_mse,
            "mse_gain_vs_type_mean": (1.0 - mse / mean_baseline_mse) if mean_baseline_mse > 1e-8 else None,
            "play_prob_mean": bucket["play_prob_sum"] / n,
            "play_target_mean": bucket["play_target_sum"] / n,
            "play_mae": bucket["play_mae_sum"] / n,
            "feedback": {label: finalize_binary_bucket(bucket["feedback"][label]) for label in FEEDBACK_LABELS},
        }
    return out


def format_feedback_summary(feedback_stats: dict[str, dict[str, float | None]]) -> str:
    parts = []
    for label in FEEDBACK_LABELS:
        stats = feedback_stats[label]
        parts.append(
            f"{label}:true={format_optional(stats['true_rate'])}"
            f" prob={format_optional(stats['prob_mean'])}"
            f" pred={format_optional(stats['pred_rate'])}"
            f" rec={format_optional(stats['recall'])}"
            f" prec={format_optional(stats['precision'])}"
        )
    return " | ".join(parts)


def format_binary_summary(label_name: str, stats: dict[str, float | None]) -> str:
    return (
        f"{label_name}:true={format_optional(stats['true_rate'])}"
        f" prob={format_optional(stats['prob_mean'])}"
        f" pred={format_optional(stats['pred_rate'])}"
        f" acc={format_optional(stats.get('acc'))}"
        f" bacc={format_optional(stats.get('balanced_acc'))}"
        f" rec={format_optional(stats['recall'])}"
        f" prec={format_optional(stats['precision'])}"
        f" f1={format_optional(stats.get('f1'))}"
    )


def format_multiclass_summary(detail: dict[str, Any], labels: list[str] | None = None) -> str:
    parts = [f"acc={format_optional(detail.get('acc'))}"]
    label_list = labels or list(detail.get("classes", {}).keys())
    for label in label_list:
        stats = detail["classes"][label]
        parts.append(
            f"{label}:true={format_optional(stats['true_rate'])}"
            f" prob={format_optional(stats['prob_mean'])}"
            f" pred={format_optional(stats['pred_rate'])}"
            f" rec={format_optional(stats['recall'])}"
        )
    return " | ".join(parts)


def format_type_summary(type_stats: dict[str, dict[str, Any]]) -> str:
    parts = []
    for type_name in REGRET_TYPE_NAMES:
        stats = type_stats.get(type_name, {})
        n = int(stats.get("n", 0) or 0)
        if n <= 0:
            continue
        parts.append(
            f"{type_name}:n={n}"
            f" tgt={format_optional(stats.get('target_mean'))}"
            f" pred={format_optional(stats.get('pred_mean'))}"
            f" mae={format_optional(stats.get('mae'))}"
        )
    return " | ".join(parts)


def format_threshold_summary(label_name: str, stats: dict[str, float | None]) -> str:
    return (
        f"{label_name}:true={format_optional(stats['true_rate'])}"
        f" score={format_optional(stats['score_mean'])}"
        f" pred={format_optional(stats['pred_rate'])}"
        f" acc={format_optional(stats.get('acc'))}"
        f" bacc={format_optional(stats.get('balanced_acc'))}"
        f" rec={format_optional(stats['recall'])}"
        f" prec={format_optional(stats['precision'])}"
        f" f1={format_optional(stats.get('f1'))}"
    )


def compute_type_gap(type_stats: dict[str, dict[str, Any]]) -> float | None:
    none_stats = type_stats.get("none", {})
    none_n = int(none_stats.get("n", 0) or 0)
    if none_n <= 0:
        return None
    none_pred = float(none_stats.get("pred_mean", 0.0) or 0.0)
    neg_total = 0
    neg_pred_sum = 0.0
    for type_name in NEGATIVE_TYPE_LABELS:
        stats = type_stats.get(type_name, {})
        n = int(stats.get("n", 0) or 0)
        if n <= 0:
            continue
        neg_total += n
        neg_pred_sum += n * float(stats.get("pred_mean", 0.0) or 0.0)
    if neg_total <= 0:
        return None
    return none_pred - (neg_pred_sum / float(neg_total))


def aggregate_coarse_score(
    reward_sign_detail: dict[str, float | None],
    play_engage_detail: dict[str, float | None],
    negative_detail: dict[str, float | None],
) -> tuple[float, float]:
    components: list[float] = []
    for value in [
        reward_sign_detail.get("acc"),
        play_engage_detail.get("acc"),
        negative_detail.get("balanced_acc"),
    ]:
        if value is not None:
            components.append(float(value))
    if not components:
        return 0.0, 1.0
    coarse_score = float(sum(components) / len(components))
    return coarse_score, float(1.0 - coarse_score)


def make_loader(
    path: str,
    max_rows: int,
    split_name: str,
    args: argparse.Namespace,
) -> DataLoader:
    is_train = split_name == "train"
    dataset = TransitionIterableDataset(
        path,
        args.dense_item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=max_rows,
        batch_size=args.read_batch_size,
        shuffle_files=is_train and args.shuffle_train_files,
        shuffle_buffer_size=args.shuffle_buffer_size if is_train else 0,
        seed=args.seed,
        sample_across_files=is_train and args.train_sample_across_files,
        balance_users=is_train and args.train_balance_by_user,
        balance_negative_types=is_train and args.train_balance_negative_types,
        negative_share=args.train_negative_share if is_train else 0.0,
    )
    print(
        f"[data] {split_name}: files={len(dataset.files)} max_rows={max_rows} "
        f"shuffle_files={dataset.shuffle_files} "
        f"shuffle_buffer={dataset.shuffle_buffer_size} "
        f"sample_across_files={dataset.sample_across_files} "
        f"balance_users={dataset.balance_users} "
        f"balance_negative_types={dataset.balance_negative_types} "
        f"negative_share={dataset.negative_share:.2f}"
    )
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def run_epoch(
    model: RegretUserResponse,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    label: str,
    reward_type_weights: torch.Tensor,
    reward_loss_weight: float,
    listen_loss_weight: float,
    listen_pos_weight: torch.Tensor,
    play_loss_weight: float,
    play_bucket_weights: torch.Tensor,
    feedback_loss_weight: float,
    feedback_pos_weights: torch.Tensor,
    feedback_threshold: float,
    negative_pos_weight: torch.Tensor,
    negative_type_weights: torch.Tensor,
    negative_type_focal_gamma: float,
    regret_loss_weight: float,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    expected_rows = 0
    expected_steps = None
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        expected_rows = int(getattr(dataset, "max_rows", 0) or 0)
    if expected_rows > 0:
        batch_size_hint = int(getattr(loader, "batch_size", 0) or 0)
        if batch_size_hint > 0:
            expected_steps = int(math.ceil(expected_rows / batch_size_hint))
    total_loss = 0.0
    total_simulator_loss = 0.0
    total_mae = 0.0
    total_n = 0
    pred_sum = 0.0
    target_sum = 0.0
    target_sq_sum = 0.0
    total_listen_loss = 0.0
    total_listen_acc = 0.0
    total_play_loss = 0.0
    total_play_mae = 0.0
    total_play_bucket_acc = 0.0
    total_positive_gate = 0.0
    total_feedback_loss = 0.0
    total_feedback_acc = 0.0
    total_negative_loss = 0.0
    total_negative_acc = 0.0
    total_negative_type_loss = 0.0
    total_negative_type_acc = 0.0
    total_regret_loss = 0.0
    total_regret_acc = 0.0
    listen_stats = build_binary_stats()
    play_bucket_stats = build_multiclass_stats(len(PLAY_BUCKET_LABELS))
    play_engage_stats = build_threshold_stats()
    feedback_stats = {label_name: build_binary_stats() for label_name in FEEDBACK_LABELS}
    negative_stats = build_binary_stats()
    reward_sign_stats = build_threshold_stats()
    negative_type_stats = build_multiclass_stats(len(NEGATIVE_TYPE_LABELS))
    type_stats = build_type_stats()
    pbar = tqdm(loader, desc=f"[epoch {epoch}] {label}", ncols=110, total=expected_steps)
    for batch in pbar:
        batch = move_batch(batch, device)
        sample_weights = reward_type_weights[batch["regret_type_id"].long()] if is_train else None
        out = model.loss(
            batch,
            reward_loss_weight=reward_loss_weight,
            reward_sample_weights=sample_weights,
            listen_loss_weight=listen_loss_weight,
            listen_pos_weight=listen_pos_weight,
            play_loss_weight=play_loss_weight,
            play_bucket_weights=play_bucket_weights,
            feedback_loss_weight=feedback_loss_weight,
            feedback_pos_weights=feedback_pos_weights,
            negative_pos_weight=negative_pos_weight,
            negative_type_weights=negative_type_weights,
            negative_type_focal_gamma=negative_type_focal_gamma,
            regret_loss_weight=regret_loss_weight,
        )
        loss = out["loss"]
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_size = int(batch["reward"].shape[0])
        rewards = batch["reward"].detach()
        preds = out["preds"].detach()
        play_prob = out["play_prob"].detach()
        play_bucket_probs = out["play_bucket_probs"].detach()
        play_target = batch["play_target"].detach()
        play_bucket_target = batch["play_bucket_id"].detach()
        positive_gate = out["positive_gate"].detach()
        feedback_probs = out["feedback_probs"].detach()
        feedback_targets = batch["feedback_targets"].detach()
        feedback_valid_mask = batch.get("feedback_valid_mask")
        if feedback_valid_mask is not None:
            feedback_valid_mask = feedback_valid_mask.detach()
        negative_prob = out["negative_prob"].detach()
        negative_type_probs = out["negative_type_probs"].detach()
        listen_prob = out["listen_prob"].detach()
        listen_target = batch["listen_target"].detach()
        regret_type_id = batch["regret_type_id"].detach()
        negative_target = (regret_type_id > 0).float()
        negative_type_target = (regret_type_id - 1).clamp_min(0)
        negative_mask = regret_type_id > 0
        sq_err = (preds - rewards) * (preds - rewards)
        abs_err = (preds - rewards).abs()

        total_loss += float(out["mse"].item()) * batch_size
        total_simulator_loss += float(out["simulator_loss"].item()) * batch_size
        total_mae += float(out["mae"].item()) * batch_size
        total_listen_loss += float(out["listen_loss"].item()) * batch_size
        total_listen_acc += float(out["listen_acc"].item()) * batch_size
        total_play_loss += float(out["play_loss"].item()) * batch_size
        total_play_mae += float(out["play_mae"].item()) * batch_size
        total_play_bucket_acc += float(out["play_bucket_acc"].item()) * batch_size
        total_positive_gate += float(positive_gate.sum().item())
        total_feedback_loss += float(out["feedback_loss"].item()) * batch_size
        total_feedback_acc += float(out["feedback_acc"].item()) * batch_size
        total_negative_loss += float(out["negative_loss"].item()) * batch_size
        total_negative_acc += float(out["negative_acc"].item()) * batch_size
        total_negative_type_loss += float(out["negative_type_loss"].item()) * batch_size
        total_negative_type_acc += float(out["negative_type_acc"].item()) * batch_size
        total_regret_loss += float(out["regret_loss"].item()) * batch_size
        total_regret_acc += float(out["regret_acc"].item()) * batch_size
        total_n += batch_size
        pred_sum += float(preds.sum().item())
        target_sum += float(rewards.sum().item())
        target_sq_sum += float((rewards * rewards).sum().item())

        update_binary_bucket(listen_stats, listen_prob, listen_target, feedback_threshold)
        update_multiclass_bucket(play_bucket_stats, play_bucket_probs, play_bucket_target)
        update_binary_bucket(negative_stats, negative_prob, negative_target, feedback_threshold)
        update_threshold_bucket(reward_sign_stats, preds, (rewards > 0.0).float(), 0.0)
        update_threshold_bucket(
            play_engage_stats,
            play_prob,
            (play_target > float(model.low_play_threshold)).float(),
            float(model.low_play_threshold),
        )
        if bool(negative_mask.any()):
            update_multiclass_bucket(negative_type_stats, negative_type_probs[negative_mask], negative_type_target[negative_mask])
        for idx, label_name in enumerate(FEEDBACK_LABELS):
            update_binary_bucket(
                feedback_stats[label_name],
                feedback_probs[:, idx],
                feedback_targets[:, idx],
                feedback_threshold,
                feedback_valid_mask[:, idx] if feedback_valid_mask is not None else None,
            )
        for type_id, type_name in enumerate(REGRET_TYPE_NAMES):
            mask = regret_type_id == type_id
            if not bool(mask.any()):
                continue
            bucket = type_stats[type_name]
            mask_float_n = float(mask.sum().item())
            bucket["n"] += mask_float_n
            bucket["mse_sum"] += float(sq_err[mask].sum().item())
            bucket["mae_sum"] += float(abs_err[mask].sum().item())
            bucket["pred_sum"] += float(preds[mask].sum().item())
            bucket["target_sum"] += float(rewards[mask].sum().item())
            bucket["target_sq_sum"] += float((rewards[mask] * rewards[mask]).sum().item())
            bucket["play_prob_sum"] += float(play_prob[mask].sum().item())
            bucket["play_target_sum"] += float(play_target[mask].sum().item())
            bucket["play_mae_sum"] += float((play_prob[mask] - play_target[mask]).abs().sum().item())
            for idx, label_name in enumerate(FEEDBACK_LABELS):
                update_binary_bucket(
                    bucket["feedback"][label_name],
                    feedback_probs[mask, idx],
                    feedback_targets[mask, idx],
                    feedback_threshold,
                    feedback_valid_mask[mask, idx] if feedback_valid_mask is not None else None,
                )

        postfix = (
            f"mse={total_loss/max(total_n,1):.5f} "
            f"sim={total_simulator_loss/max(total_n,1):.5f} "
            f"play_bacc={total_play_bucket_acc/max(total_n,1):.3f} "
            f"fb_acc={total_feedback_acc/max(total_n,1):.3f} "
            f"play_mae={total_play_mae/max(total_n,1):.3f} "
            f"mae={total_mae/max(total_n,1):.5f} "
            f"rtype_acc={total_regret_acc/max(total_n,1):.3f}"
        )
        if expected_rows > 0:
            progress_pct = 100.0 * min(total_n, expected_rows) / max(expected_rows, 1)
            postfix += f" rows={min(total_n, expected_rows)}/{expected_rows} ({progress_pct:.1f}%)"
        pbar.set_postfix_str(postfix, refresh=True)
    pbar.close()

    target_mean = target_sum / max(total_n, 1)
    mean_baseline_mse = max(target_sq_sum / max(total_n, 1) - target_mean * target_mean, 0.0)
    mse = total_loss / max(total_n, 1)
    reward_sign_detail = finalize_threshold_bucket(reward_sign_stats)
    play_engage_detail = finalize_threshold_bucket(play_engage_stats)
    negative_detail = finalize_binary_bucket(negative_stats)
    by_type = finalize_type_buckets(type_stats)
    coarse_score, coarse_error = aggregate_coarse_score(reward_sign_detail, play_engage_detail, negative_detail)
    metrics: dict[str, Any] = {
        "n": float(total_n),
        "mse": mse,
        "simulator_loss": total_simulator_loss / max(total_n, 1),
        "mae": total_mae / max(total_n, 1),
        "pred_mean": pred_sum / max(total_n, 1),
        "target_mean": target_mean,
        "mean_baseline_mse": mean_baseline_mse,
        "mse_gain_vs_mean": 1.0 - mse / mean_baseline_mse if mean_baseline_mse > 0 else 0.0,
        "listen_loss": total_listen_loss / max(total_n, 1),
        "listen_acc": total_listen_acc / max(total_n, 1),
        "play_loss": total_play_loss / max(total_n, 1),
        "play_mae": total_play_mae / max(total_n, 1),
        "play_bucket_acc": total_play_bucket_acc / max(total_n, 1),
        "positive_gate_mean": total_positive_gate / max(total_n, 1),
        "feedback_loss": total_feedback_loss / max(total_n, 1),
        "feedback_acc": total_feedback_acc / max(total_n, 1),
        "negative_loss": total_negative_loss / max(total_n, 1),
        "negative_acc": total_negative_acc / max(total_n, 1),
        "negative_type_loss": total_negative_type_loss / max(total_n, 1),
        "negative_type_acc": total_negative_type_acc / max(total_n, 1),
        "regret_loss": total_regret_loss / max(total_n, 1),
        "regret_acc": total_regret_acc / max(total_n, 1),
        "listen_detail": finalize_binary_bucket(listen_stats),
        "play_bucket_detail": finalize_multiclass_bucket(play_bucket_stats, PLAY_BUCKET_LABELS),
        "play_engage_detail": play_engage_detail,
        "feedback_detail": {label_name: finalize_binary_bucket(feedback_stats[label_name]) for label_name in FEEDBACK_LABELS},
        "negative_detail": negative_detail,
        "reward_sign_detail": reward_sign_detail,
        "negative_type_detail": finalize_multiclass_bucket(negative_type_stats, NEGATIVE_TYPE_LABELS),
        "by_type": by_type,
        "coarse_score": coarse_score,
        "coarse_error": coarse_error,
        "none_vs_negative_gap": compute_type_gap(by_type),
    }
    return metrics


def print_split_summary(split_name: str, metrics: dict[str, Any]) -> None:
    print(
        f"[{split_name}] mse={metrics['mse']:.6f} "
        f"sim_loss={metrics['simulator_loss']:.6f} "
        f"gain={metrics['mse_gain_vs_mean']:.3f} "
        f"play_mae={metrics['play_mae']:.3f} "
        f"play_bacc={metrics['play_bucket_acc']:.3f} "
        f"coarse={metrics['coarse_score']:.3f} "
        f"gap={format_optional(metrics.get('none_vs_negative_gap'))} "
        f"pos_gate={metrics['positive_gate_mean']:.3f} "
        f"fb_acc={metrics['feedback_acc']:.3f} "
        f"neg_acc={metrics['negative_acc']:.3f} "
        f"negtype_acc={metrics['negative_type_acc']:.3f} "
        f"rtype_acc={metrics['regret_acc']:.3f}"
    )
    print(f"[{split_name}] listen: {format_binary_summary('listen', metrics['listen_detail'])}")
    print(f"[{split_name}] play buckets: {format_multiclass_summary(metrics['play_bucket_detail'], PLAY_BUCKET_LABELS)}")
    print(f"[{split_name}] play engage: {format_threshold_summary('play>low', metrics['play_engage_detail'])}")
    print(f"[{split_name}] feedback: {format_feedback_summary(metrics['feedback_detail'])}")
    print(f"[{split_name}] reward sign: {format_threshold_summary('reward>0', metrics['reward_sign_detail'])}")
    print(f"[{split_name}] negative: {format_binary_summary('negative', metrics['negative_detail'])}")
    print(f"[{split_name}] negative types: {format_multiclass_summary(metrics['negative_type_detail'], NEGATIVE_TYPE_LABELS)}")
    print(f"[{split_name}] by_type: {format_type_summary(metrics['by_type'])}")


def build_model_args_from_transition(args: argparse.Namespace, reward_config: dict[str, Any]) -> dict[str, Any]:
    feature_shape = np.load(args.dense_item_features_npy, mmap_mode="r").shape
    item_dim = int(feature_shape[1])
    return {
        "item_dim": item_dim,
        "hidden_dim": int(args.hidden_dim),
        "event_vocab_size": max(EVENT_TYPE_TO_ID.values()) + 1,
        "prior_dim": len(PRIOR_STAT_COLUMNS),
        "regret_num_classes": len(REGRET_TYPE_TO_ID),
        "dropout": float(args.dropout),
        "history_heuristic_init": float(args.history_heuristic_init),
        "reward_config": reward_config,
        "decouple_reward_model": bool(args.decouple_reward_model),
    }


def resolve_best_metric_name(args: argparse.Namespace) -> str:
    if str(args.best_metric) != "auto":
        return str(args.best_metric)
    return "coarse_error" if bool(args.decouple_reward_model) else "mse"


def evaluate_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    reward_type_weights: torch.Tensor,
    listen_pos_weight: torch.Tensor,
    play_bucket_weights: torch.Tensor,
    feedback_pos_weights: torch.Tensor,
    negative_pos_weight: torch.Tensor,
    negative_type_weights: torch.Tensor,
) -> dict[str, Any]:
    checkpoint = torch.load(args.eval_only_checkpoint, map_location=device, weights_only=False)
    if not args.item_features_npy and checkpoint.get("data_args", {}).get("dense_item_features_npy"):
        args.dense_item_features_npy = checkpoint["data_args"]["dense_item_features_npy"]
    if checkpoint.get("data_args", {}).get("max_seq_len"):
        args.max_seq_len = int(checkpoint["data_args"]["max_seq_len"])
    model_args = dict(checkpoint["model_args"])
    model_args.setdefault("decouple_reward_model", False)
    model = RegretUserResponse(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[eval] history_heuristic_gate={torch.sigmoid(model.history_heuristic_gate_logit.detach()).item():.4f}")
    print(f"[eval] reward_mode={'decoupled_rule' if model.decouple_reward_model else 'learned_scorer'}")
    val_loader = make_loader(args.val_path, args.max_val_rows, "val", args)
    val_metrics = run_epoch(
        model,
        val_loader,
        device,
        None,
        0,
        "val",
        reward_type_weights,
        args.reward_loss_weight,
        args.listen_loss_weight,
        listen_pos_weight,
        args.play_loss_weight,
        play_bucket_weights,
        args.feedback_loss_weight,
        feedback_pos_weights,
        args.feedback_threshold,
        negative_pos_weight,
        negative_type_weights,
        args.negative_type_focal_gamma,
        args.regret_loss_weight,
    )
    print_split_summary("eval val", val_metrics)
    result: dict[str, Any] = {"val": val_metrics}
    test_path = Path(args.test_path)
    if test_path.exists():
        test_loader = make_loader(args.test_path, args.max_test_rows, "test", args)
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            None,
            0,
            "test",
            reward_type_weights,
            args.reward_loss_weight,
            args.listen_loss_weight,
            listen_pos_weight,
            args.play_loss_weight,
            play_bucket_weights,
            args.feedback_loss_weight,
            feedback_pos_weights,
            args.feedback_threshold,
            negative_pos_weight,
            negative_type_weights,
            args.negative_type_focal_gamma,
            args.regret_loss_weight,
        )
        print_split_summary("eval test", test_metrics)
        result["test"] = test_metrics
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    save_path = Path(args.save_path)
    save_meta = Path(args.save_meta)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)

    reward_config, reward_config_source = load_reward_config(args)
    print(f"[config] reward source: {reward_config_source}")
    print(f"[config] reward version: {reward_config.get('reward_version', 'unknown')}")
    print(f"[config] reward model: {'decoupled_rule' if args.decouple_reward_model else 'learned_scorer'}")
    best_metric_name = resolve_best_metric_name(args)
    print(f"[config] best metric: {best_metric_name}")

    reward_type_weights = torch.tensor(
        [
            args.reward_weight_none,
            args.reward_weight_low_play,
            args.reward_weight_dislike,
            args.reward_weight_unlike,
        ],
        dtype=torch.float32,
        device=device,
    )
    listen_pos_weight, play_bucket_weights, feedback_pos_weights, negative_pos_weight, negative_type_weights = build_loss_weight_tensors(args, device)

    if args.eval_only_checkpoint:
        evaluate_checkpoint(
            args,
            device,
            reward_type_weights,
            listen_pos_weight,
            play_bucket_weights,
            feedback_pos_weights,
            negative_pos_weight,
            negative_type_weights,
        )
        return

    model_args = build_model_args_from_transition(args, reward_config)
    model = RegretUserResponse(**model_args).to(device)
    prior_init_stats: dict[str, Any] | None = None
    if args.init_head_bias_from_train:
        prior_init_stats = estimate_train_label_priors(args.train_path, args.prior_estimate_rows)
        initialize_head_bias_from_priors(model, prior_init_stats)
        print(
            "[init] head priors: "
            f"rows={prior_init_stats['rows']} "
            f"listen={prior_init_stats['listen_rate']:.4f} "
            f"play={','.join(f'{x:.3f}' for x in prior_init_stats['play_bucket_probs'])} "
            f"feedback={','.join(f'{x:.3f}' for x in prior_init_stats['feedback_rates'])} "
            f"negative={prior_init_stats['negative_rate']:.3f} "
            f"negtypes={','.join(f'{x:.3f}' for x in prior_init_stats['negative_type_probs'])}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(args.train_path, args.max_train_rows, "train", args)
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        listen_loss_weight = float(args.listen_loss_weight) * (float(args.listen_loss_weight_decay) ** (epoch - 1))
        play_loss_weight = float(args.play_loss_weight) * (float(args.play_loss_weight_decay) ** (epoch - 1))
        feedback_loss_weight = float(args.feedback_loss_weight) * (float(args.feedback_loss_weight_decay) ** (epoch - 1))
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            epoch,
            "train",
            reward_type_weights,
            args.reward_loss_weight,
            listen_loss_weight,
            listen_pos_weight,
            play_loss_weight,
            play_bucket_weights,
            feedback_loss_weight,
            feedback_pos_weights,
            args.feedback_threshold,
            negative_pos_weight,
            negative_type_weights,
            args.negative_type_focal_gamma,
            args.regret_loss_weight,
        )
        val_loader = make_loader(args.val_path, args.max_val_rows, "val", args)
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            epoch,
            "val",
            reward_type_weights,
            args.reward_loss_weight,
            listen_loss_weight,
            listen_pos_weight,
            play_loss_weight,
            play_bucket_weights,
            feedback_loss_weight,
            feedback_pos_weights,
            args.feedback_threshold,
            negative_pos_weight,
            negative_type_weights,
            args.negative_type_focal_gamma,
            args.regret_loss_weight,
        )
        heuristic_gate = float(torch.sigmoid(model.history_heuristic_gate_logit.detach()).item())
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "history_heuristic_gate": heuristic_gate,
            "reward_loss_weight": float(args.reward_loss_weight),
            "listen_loss_weight": listen_loss_weight,
            "play_loss_weight": play_loss_weight,
            "feedback_loss_weight": feedback_loss_weight,
        }
        history.append(record)
        print(
            f"[epoch {epoch}] "
            f"train_mse={train_metrics['mse']:.6f} "
            f"train_sim={train_metrics['simulator_loss']:.6f} "
            f"train_gain={train_metrics['mse_gain_vs_mean']:.3f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_sim={val_metrics['simulator_loss']:.6f} "
            f"val_mean_base={val_metrics['mean_baseline_mse']:.6f} "
            f"val_gain={val_metrics['mse_gain_vs_mean']:.3f} "
            f"val_coarse={val_metrics['coarse_score']:.3f} "
            f"val_gap={format_optional(val_metrics.get('none_vs_negative_gap'))} "
            f"val_play_bacc={val_metrics['play_bucket_acc']:.3f} "
            f"val_play_eng_acc={format_optional(val_metrics['play_engage_detail'].get('acc'))} "
            f"val_sign_acc={format_optional(val_metrics['reward_sign_detail'].get('acc'))} "
            f"val_pos_gate={val_metrics['positive_gate_mean']:.3f} "
            f"val_fb_acc={val_metrics['feedback_acc']:.3f} "
            f"val_neg_acc={val_metrics['negative_acc']:.3f} "
            f"val_neg_bacc={format_optional(val_metrics['negative_detail'].get('balanced_acc'))} "
            f"val_negtype_acc={val_metrics['negative_type_acc']:.3f} "
            f"val_play_mae={val_metrics['play_mae']:.3f} "
            f"val_rtype_acc={val_metrics['regret_acc']:.3f} "
            f"hist_gate={heuristic_gate:.4f} "
            f"loss_w=({args.reward_loss_weight:.3f},{listen_loss_weight:.3f},{play_loss_weight:.3f},{feedback_loss_weight:.3f})"
        )
        print(f"[epoch {epoch}] val play buckets: {format_multiclass_summary(val_metrics['play_bucket_detail'], PLAY_BUCKET_LABELS)}")
        print(f"[epoch {epoch}] val play engage: {format_threshold_summary('play>low', val_metrics['play_engage_detail'])}")
        print(f"[epoch {epoch}] val feedback: {format_feedback_summary(val_metrics['feedback_detail'])}")
        print(f"[epoch {epoch}] val reward sign: {format_threshold_summary('reward>0', val_metrics['reward_sign_detail'])}")
        print(f"[epoch {epoch}] val negative: {format_binary_summary('negative', val_metrics['negative_detail'])}")
        print(f"[epoch {epoch}] val negative types: {format_multiclass_summary(val_metrics['negative_type_detail'], NEGATIVE_TYPE_LABELS)}")
        print(f"[epoch {epoch}] val by_type: {format_type_summary(val_metrics['by_type'])}")
        current_val_metric = float(val_metrics[best_metric_name])
        if current_val_metric < best_val:
            best_val = current_val_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": model_args,
                    "data_args": {
                        "dense_item_features_npy": args.dense_item_features_npy,
                        "max_seq_len": int(args.max_seq_len),
                    },
                    "training_args": vars(args),
                    "reward_config_source": reward_config_source,
                    "best_val_metric_name": best_metric_name,
                    "best_val_metric": best_val,
                    "best_val_mse": float(val_metrics["mse"]),
                },
                save_path,
            )

    meta: dict[str, Any] = {
        "model_args": model_args,
        "training_args": vars(args),
        "reward_config_source": reward_config_source,
        "reward_model_mode": "decoupled_rule" if args.decouple_reward_model else "learned_scorer",
        "best_metric_name": best_metric_name,
        "best_metric_value": best_val,
        "lowest_val_mse_across_run": min((record["val"]["mse"] for record in history), default=None),
        "lowest_val_coarse_error_across_run": min((record["val"]["coarse_error"] for record in history), default=None),
        "highest_val_coarse_score_across_run": max((record["val"]["coarse_score"] for record in history), default=None),
        "history": history,
        "prior_init_stats": prior_init_stats,
        "final_history_heuristic_gate": float(torch.sigmoid(model.history_heuristic_gate_logit.detach()).item()),
        "save_path": str(save_path),
    }
    if args.eval_test_after_train and Path(args.test_path).exists():
        best_checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        best_model = RegretUserResponse(**best_checkpoint["model_args"]).to(device)
        best_model.load_state_dict(best_checkpoint["model_state_dict"])
        test_loader = make_loader(args.test_path, args.max_test_rows, "test", args)
        test_metrics = run_epoch(
            best_model,
            test_loader,
            device,
            None,
            0,
            "test",
            reward_type_weights,
            args.reward_loss_weight,
            args.listen_loss_weight,
            listen_pos_weight,
            args.play_loss_weight,
            play_bucket_weights,
            args.feedback_loss_weight,
            feedback_pos_weights,
            args.feedback_threshold,
            negative_pos_weight,
            negative_type_weights,
            args.negative_type_focal_gamma,
            args.regret_loss_weight,
        )
        meta["test_metrics"] = test_metrics
        print_split_summary("test", test_metrics)
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    RegretUserResponseEnv.write_env_meta(save_path, args.env_meta)
    print(f"[done] model saved to {save_path}")
    print(f"[done] meta saved to {save_meta}")


if __name__ == "__main__":
    main()
