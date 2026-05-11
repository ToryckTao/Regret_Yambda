#!/usr/bin/env python3
"""历史离线 SID 训练：直接用 transition shard 训练 actor/critic。

当前主线已经转向 simulator rollout + HSAC；本文件保留用于消融和旧实验复现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.transition_dataset import TransitionIterableDataset  # noqa: E402
from regret_core.data.schema import REGRET_TYPE_TO_ID  # noqa: E402
from regret_core.model.sid_policy import RegretAwareSIDPolicy, SIDActionQCritic, StateValueCritic, TokenRewardCritic  # noqa: E402


class OfflineTransitionEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regret SID offline transition trainer")
    parser.add_argument("--transition_root", default=str(PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans_v2_smoke_timefix"))
    parser.add_argument("--item_features_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item_features.npy"))
    parser.add_argument("--dense_item2sid_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item2sid.npy"))
    parser.add_argument("--actor_init_checkpoint", default="", help="Optional actor state_dict checkpoint for partial warm start.")
    parser.add_argument("--save_path", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_feedback"))
    parser.add_argument("--save_meta", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_feedback.meta.json"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_train_rows", type=int, default=1000000)
    parser.add_argument("--max_val_rows", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--shuffle_train_files", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=0)
    parser.add_argument("--train_sample_across_files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--value_lr", type=float, default=1e-3)
    parser.add_argument("--actor_decay", type=float, default=1e-5)
    parser.add_argument("--critic_decay", type=float, default=1e-5)
    parser.add_argument("--value_decay", type=float, default=1e-5)
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--use_history_feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_history_event_type", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_type_vocab_size", type=int, default=8)
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--critic_dropout_rate", type=float, default=0.2)
    parser.add_argument("--q_critic_type", choices=["token_context", "sid_action"], default="token_context")
    parser.add_argument("--reward_threshold", type=float, default=0.0)
    parser.add_argument("--reward_temperature", type=float, default=0.5)
    parser.add_argument("--bc_weight", type=float, default=1.0)
    parser.add_argument("--avoid_weight", type=float, default=0.25)
    parser.add_argument("--critic_weight", type=float, default=0.5)
    parser.add_argument("--value_weight", type=float, default=0.0)
    parser.add_argument("--old_surrogate_weight", type=float, default=1.0)
    parser.add_argument("--adv_loss_weight", type=float, default=0.0)
    parser.add_argument("--use_advantage_loss", action="store_true")
    parser.add_argument("--detach_critic_context", action="store_true")
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--advantage_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rrca_weight", type=float, default=0.0)
    parser.add_argument("--rrca_low_play_weight", type=float, default=1.0)
    parser.add_argument("--rrca_dislike_weight", type=float, default=1.0)
    parser.add_argument("--rrca_unlike_weight", type=float, default=1.0)
    parser.add_argument("--rrca_undislike_bonus", type=float, default=0.0)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--sara_eta", type=float, default=0.5)
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def load_actor_checkpoint(actor: RegretAwareSIDPolicy, ckpt_path: str, device: torch.device) -> bool:
    if not ckpt_path:
        return False
    path = Path(ckpt_path)
    if not path.exists():
        print(f"[init] actor checkpoint not found: {path}")
        return False
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    actor_state = actor.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in actor_state and tuple(actor_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = actor.load_state_dict(compatible, strict=False)
    print(
        f"[init] loaded_actor={path} compatible={len(compatible)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return True


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    is_train = split == "train"
    max_rows = args.max_train_rows if is_train else args.max_val_rows
    dataset = TransitionIterableDataset(
        Path(args.transition_root) / split,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=max_rows,
        batch_size=args.read_batch_size,
        shuffle_files=is_train and args.shuffle_train_files,
        shuffle_buffer_size=args.shuffle_buffer_size if is_train else 0,
        seed=args.seed,
        sample_across_files=is_train and args.train_sample_across_files,
    )
    print(
        f"[data] {split}: files={len(dataset.files)} max_rows={max_rows} "
        f"shuffle_files={dataset.shuffle_files} shuffle_buffer={dataset.shuffle_buffer_size} "
        f"sample_across_files={dataset.sample_across_files}"
    )
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def target_sid_from_dense_ids(
    dense_item2sid: np.ndarray,
    item_ids: torch.Tensor,
    device: torch.device,
    sid_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    item_ids_np = item_ids.detach().cpu().numpy().astype(np.int64)
    sid_np = np.asarray(dense_item2sid[item_ids_np], dtype=np.int64)
    sid = torch.as_tensor(sid_np, dtype=torch.long, device=device)
    valid = (sid >= 0).all(dim=1) & (sid < int(sid_vocab_size)).all(dim=1)
    return sid, valid


def move_observation(batch: dict[str, torch.Tensor], valid: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_features": batch["history_features"].to(device)[valid],
        "history_feedbacks": batch["history_feedbacks"].to(device).float()[valid],
        "history_event_type_ids": batch["history_event_type_ids"].to(device).long()[valid],
        "history_mask": batch["history_mask"].to(device).float()[valid],
    }


def compute_rrca_correction(
    batch: dict[str, torch.Tensor],
    valid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    regret_type_id = batch["regret_type_id"].to(device).long()[valid]
    regret_strength = batch["regret_strength"].to(device).float()[valid]
    correction = torch.zeros_like(regret_strength)

    low_play_id = int(REGRET_TYPE_TO_ID["low_play"])
    dislike_id = int(REGRET_TYPE_TO_ID["dislike"])
    unlike_id = int(REGRET_TYPE_TO_ID["unlike"])
    correction = correction - float(args.rrca_low_play_weight) * regret_strength * (regret_type_id == low_play_id).float()
    correction = correction - float(args.rrca_dislike_weight) * regret_strength * (regret_type_id == dislike_id).float()
    correction = correction - float(args.rrca_unlike_weight) * regret_strength * (regret_type_id == unlike_id).float()

    if "feedback_targets" in batch and float(args.rrca_undislike_bonus) != 0.0:
        undislike = batch["feedback_targets"].to(device).float()[valid][:, 3]
        correction = correction + float(args.rrca_undislike_bonus) * undislike
    return float(args.rrca_weight) * correction


def compute_loss(
    actor: RegretAwareSIDPolicy,
    critic: TokenRewardCritic | SIDActionQCritic,
    value_critic: StateValueCritic,
    batch: dict[str, torch.Tensor],
    dense_item2sid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sid_vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_sid, valid = target_sid_from_dense_ids(dense_item2sid, batch["target_dense_item_id"], device, sid_vocab_size)
    if not bool(valid.any()):
        return torch.zeros((), device=device, requires_grad=True), {"n": 0.0}

    obs = move_observation(batch, valid, device)
    reward = batch["reward"].to(device).float()[valid]
    target_sid = target_sid[valid]
    output = actor(obs)
    critic_context = output["context_list"].detach() if args.detach_critic_context else output["context_list"]
    q_feed = {"context_list": critic_context, "target_sid": target_sid}
    q_pred = critic(q_feed)["q"]
    v_pred = value_critic({"context_list": critic_context})["v"]

    temp = max(float(args.reward_temperature), 1e-6)
    threshold = float(args.reward_threshold)
    pos_weight = torch.sigmoid((reward - threshold) / temp).detach()
    neg_weight = torch.sigmoid((threshold - reward) / temp).detach()
    pos_mask = reward >= threshold
    neg_mask = ~pos_mask

    nll = torch.zeros_like(reward)
    avoid = torch.zeros_like(reward)
    entropy = torch.zeros((), device=device)
    token_acc_sum = torch.zeros((), device=device)
    full_match = torch.ones_like(reward, dtype=torch.bool)
    pos_acc_sum = torch.zeros((), device=device)
    neg_acc_sum = torch.zeros((), device=device)
    pos_acc_count = 0
    neg_acc_count = 0

    for level, logits_l in enumerate(output["sid_logits"]):
        z_l = target_sid[:, level]
        log_probs_l = F.log_softmax(logits_l, dim=-1)
        logp_l = log_probs_l.gather(1, z_l.view(-1, 1)).squeeze(1)
        prob_l = logp_l.exp().clamp(max=1.0 - 1e-6)
        nll = nll - logp_l
        avoid = avoid - torch.log1p(-prob_l)
        pred_l = logits_l.argmax(dim=-1)
        hit_l = (pred_l == z_l).float()
        token_acc_sum = token_acc_sum + hit_l.mean()
        full_match = full_match & (pred_l == z_l)
        if bool(pos_mask.any()):
            pos_acc_sum = pos_acc_sum + hit_l[pos_mask].mean()
            pos_acc_count += 1
        if bool(neg_mask.any()):
            neg_acc_sum = neg_acc_sum + hit_l[neg_mask].mean()
            neg_acc_count += 1
        probs_l = log_probs_l.exp()
        entropy = entropy - (probs_l * log_probs_l).sum(dim=-1).mean()

    n_level = max(len(output["sid_logits"]), 1)
    pos_loss = (pos_weight * nll).sum() / pos_weight.sum().clamp_min(1e-6)
    neg_loss = (neg_weight * avoid).sum() / neg_weight.sum().clamp_min(1e-6)
    critic_loss = F.mse_loss(q_pred, reward)
    value_loss = F.mse_loss(v_pred, reward)
    entropy = entropy / n_level
    old_actor_loss = float(args.bc_weight) * pos_loss + float(args.avoid_weight) * neg_loss

    advantage = (q_pred - v_pred).detach()
    rrca_correction = compute_rrca_correction(batch, valid, args, device).detach()
    corrected_advantage = advantage + rrca_correction
    adv_for_loss = corrected_advantage
    if args.advantage_normalize and adv_for_loss.numel() > 1:
        adv_for_loss = (adv_for_loss - adv_for_loss.mean()) / adv_for_loss.std(unbiased=False).clamp_min(1e-6)
    if float(args.advantage_clip) > 0:
        adv_for_loss = adv_for_loss.clamp(min=-float(args.advantage_clip), max=float(args.advantage_clip))
    logp_mean = -nll / n_level
    adv_policy_loss = -(adv_for_loss * logp_mean).mean()

    adv_weight = float(args.adv_loss_weight) if args.use_advantage_loss else 0.0
    actor_loss = (
        float(args.old_surrogate_weight) * old_actor_loss
        + adv_weight * adv_policy_loss
        - float(args.entropy_weight) * entropy
    )
    loss = actor_loss + float(args.critic_weight) * critic_loss + float(args.value_weight) * value_loss

    metrics = {
        "n": float(reward.numel()),
        "loss": float(loss.detach().cpu()),
        "actor": float(actor_loss.detach().cpu()),
        "old_actor": float(old_actor_loss.detach().cpu()),
        "adv_actor": float(adv_policy_loss.detach().cpu()),
        "critic": float(critic_loss.detach().cpu()),
        "value": float(value_loss.detach().cpu()),
        "pos": float(pos_loss.detach().cpu()),
        "avoid": float(neg_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "q_mse": float(critic_loss.detach().cpu()),
        "v_mse": float(value_loss.detach().cpu()),
        "q_mean": float(q_pred.mean().detach().cpu()),
        "v_mean": float(v_pred.mean().detach().cpu()),
        "adv_mean": float(advantage.mean().detach().cpu()),
        "adv_std": float(advantage.std(unbiased=False).detach().cpu()),
        "adv_pos_share": float((advantage > 0).float().mean().detach().cpu()),
        "rrca_mean": float(rrca_correction.mean().detach().cpu()),
        "rrca_abs_mean": float(rrca_correction.abs().mean().detach().cpu()),
        "rrca_neg_share": float((rrca_correction < 0).float().mean().detach().cpu()),
        "corrected_adv_mean": float(corrected_advantage.mean().detach().cpu()),
        "corrected_adv_std": float(corrected_advantage.std(unbiased=False).detach().cpu()),
        "corrected_adv_pos_share": float((corrected_advantage > 0).float().mean().detach().cpu()),
        "reward_mean": float(reward.mean().detach().cpu()),
        "pos_share": float(pos_mask.float().mean().detach().cpu()),
        "token_acc": float((token_acc_sum / n_level).detach().cpu()),
        "full_sid_acc": float(full_match.float().mean().detach().cpu()),
        "pos_token_acc": float((pos_acc_sum / max(pos_acc_count, 1)).detach().cpu()),
        "neg_token_acc": float((neg_acc_sum / max(neg_acc_count, 1)).detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    actor: RegretAwareSIDPolicy,
    critic: TokenRewardCritic | SIDActionQCritic,
    value_critic: StateValueCritic,
    loader: DataLoader,
    dense_item2sid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    sid_vocab_size: int,
    actor_optimizer: torch.optim.Optimizer | None,
    critic_optimizer: torch.optim.Optimizer | None,
    value_optimizer: torch.optim.Optimizer | None,
    epoch: int,
    split: str,
) -> dict[str, float]:
    is_train = actor_optimizer is not None and critic_optimizer is not None and value_optimizer is not None
    actor.train(is_train)
    critic.train(is_train)
    value_critic.train(is_train)
    totals: dict[str, float] = {}
    n_rows = 0.0
    pbar = tqdm(loader, desc=f"[epoch {epoch}] {split}", ncols=120)
    for batch in pbar:
        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_loss(actor, critic, value_critic, batch, dense_item2sid, args, device, sid_vocab_size)
            if metrics.get("n", 0.0) <= 0:
                continue
            if is_train:
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(value_critic.parameters(), args.grad_clip)
                actor_optimizer.step()
                critic_optimizer.step()
                value_optimizer.step()
        batch_n = metrics["n"]
        n_rows += batch_n
        for key, value in metrics.items():
            if key == "n":
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * batch_n
        pbar.set_postfix(
            loss=totals.get("loss", 0.0) / max(n_rows, 1.0),
            q_mse=totals.get("q_mse", 0.0) / max(n_rows, 1.0),
            v_mse=totals.get("v_mse", 0.0) / max(n_rows, 1.0),
            adv=totals.get("corrected_adv_mean", 0.0) / max(n_rows, 1.0),
            token_acc=totals.get("token_acc", 0.0) / max(n_rows, 1.0),
            full=totals.get("full_sid_acc", 0.0) / max(n_rows, 1.0),
            pos_share=totals.get("pos_share", 0.0) / max(n_rows, 1.0),
        )
    pbar.close()
    if n_rows <= 0:
        return {"n": 0.0}
    return {"n": n_rows, **{key: value / n_rows for key, value in totals.items()}}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] using {device}")
    print(
        f"[history] feedback={args.use_history_feedback} "
        f"event_type={args.use_history_event_type}"
    )

    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    print(f"[sid] sid_levels={sid_levels} sid_vocab_size={sid_vocab_size} item_dim={feature_shape[1]}")

    env = OfflineTransitionEnvSpec(
        n_item=int(dense_item2sid.shape[0] - 1),
        item_dim=int(feature_shape[1]),
        max_seq_len=args.max_seq_len,
    )
    model_args = SimpleNamespace(
        sasrec_n_layer=args.sasrec_n_layer,
        sasrec_d_model=args.sasrec_d_model,
        sasrec_d_forward=args.sasrec_d_forward,
        sasrec_n_head=args.sasrec_n_head,
        sasrec_dropout=args.sasrec_dropout,
        sid_levels=sid_levels,
        sid_vocab_sizes=sid_vocab_size,
        sid_temp=args.sid_temp,
        use_history_feedback=args.use_history_feedback,
        use_history_event_type=args.use_history_event_type,
        event_type_vocab_size=args.event_type_vocab_size,
        sara_eta=args.sara_eta,
        sara_layer_weights=args.sara_layer_weights,
    )
    actor = RegretAwareSIDPolicy(model_args, env).to(device)
    actor_init_loaded = load_actor_checkpoint(actor, args.actor_init_checkpoint, device)
    if args.q_critic_type == "sid_action":
        critic = SIDActionQCritic(args, actor).to(device)
    else:
        critic = TokenRewardCritic(args, actor).to(device)
    value_critic = StateValueCritic(args, actor).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)
    value_optimizer = torch.optim.Adam(value_critic.parameters(), lr=args.value_lr, weight_decay=args.value_decay)

    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_meta).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            actor, critic, value_critic, train_loader, dense_item2sid, args, device, sid_vocab_size,
            actor_optimizer, critic_optimizer, value_optimizer, epoch, "train",
        )
        val_metrics = run_epoch(
            actor, critic, value_critic, val_loader, dense_item2sid, args, device, sid_vocab_size,
            None, None, None, epoch, "val",
        )
        print(
            f"[epoch {epoch}] train_loss={train_metrics.get('loss', float('nan')):.5f} "
            f"train_token_acc={train_metrics.get('token_acc', float('nan')):.3f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.5f} "
            f"val_q_mse={val_metrics.get('q_mse', float('nan')):.5f} "
            f"val_v_mse={val_metrics.get('v_mse', float('nan')):.5f} "
            f"val_adv={val_metrics.get('corrected_adv_mean', float('nan')):.4f} "
            f"val_adv_pos={val_metrics.get('corrected_adv_pos_share', float('nan')):.3f} "
            f"val_token_acc={val_metrics.get('token_acc', float('nan')):.3f} "
            f"val_full_sid_acc={val_metrics.get('full_sid_acc', float('nan')):.3f} "
            f"val_pos_share={val_metrics.get('pos_share', float('nan')):.3f}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics.get("loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(actor.state_dict(), str(save_path) + "_actor")
            torch.save(critic.state_dict(), str(save_path) + "_critic")
            torch.save(value_critic.state_dict(), str(save_path) + "_value")
            print(f"[save] best checkpoint epoch={epoch} val_loss={val_loss:.5f}")

    meta = {
        "training_args": vars(args),
        "device": str(device),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "uses_history_feedback": bool(args.use_history_feedback),
        "uses_history_event_type": bool(args.use_history_event_type),
        "actor_init_loaded": actor_init_loaded,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "actor_checkpoint": str(save_path) + "_actor",
        "critic_checkpoint": str(save_path) + "_critic",
        "value_checkpoint": str(save_path) + "_value",
    }
    Path(args.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] actor saved to {save_path}_actor")
    print(f"[done] critic saved to {save_path}_critic")
    print(f"[done] value saved to {save_path}_value")
    print(f"[done] meta saved to {args.save_meta}")


if __name__ == "__main__":
    main()
