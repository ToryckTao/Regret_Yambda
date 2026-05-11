#!/usr/bin/env python3
"""主线策略训练：用 simulator rollout 训练 HSAC-style SID actor，并接入 RRCA 与 RAPI。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.schema import EVENT_TYPE_TO_ID  # noqa: E402
from regret_core.data.transition_dataset import TransitionIterableDataset  # noqa: E402
from regret_core.env.user_response_env import RegretUserResponseEnv  # noqa: E402
from regret_core.model.regret_memory import RegretMemoryPool  # noqa: E402
from regret_core.model.sid_policy import MultiLevelValueCritic, RegretAwareSIDPolicy  # noqa: E402


class OfflineTransitionEnvSpec:
    def __init__(self, n_item: int, item_dim: int, max_seq_len: int) -> None:
        self.action_space = {
            "item_id": ("nominal", n_item),
            "item_feature": ("continuous", item_dim, "normal"),
        }
        self.observation_space = {
            "history": ("sequence", max_seq_len, ("continuous", item_dim)),
        }


class SIDDecoder:
    def __init__(self, dense_item2sid: np.ndarray, sid_vocab_size: int) -> None:
        sid = np.asarray(dense_item2sid)
        self.sid_levels = int(sid.shape[1])
        self.sid_vocab_size = int(sid_vocab_size)
        self.multipliers = (self.sid_vocab_size ** np.arange(self.sid_levels - 1, -1, -1)).astype(np.int64)
        valid = (sid >= 0).all(axis=1) & (sid < self.sid_vocab_size).all(axis=1)
        valid[0] = False
        dense_ids = np.nonzero(valid)[0].astype(np.int64)
        codes = self.pack_np(sid[dense_ids].astype(np.int64))
        order = np.argsort(codes, kind="mergesort")
        self.codes = codes[order]
        self.dense_ids = dense_ids[order]
        self.sid = sid

    def pack_np(self, sid_paths: np.ndarray) -> np.ndarray:
        return (sid_paths.astype(np.int64) * self.multipliers.reshape(1, -1)).sum(axis=1)

    def lookup_code(self, code: int) -> int | None:
        idx = int(np.searchsorted(self.codes, np.int64(code), side="left"))
        if idx < self.codes.shape[0] and int(self.codes[idx]) == int(code):
            return int(self.dense_ids[idx])
        return None

    def sid_for_item(self, dense_item_id: int) -> np.ndarray:
        item_id = int(dense_item_id)
        if item_id <= 0 or item_id >= self.sid.shape[0] or (self.sid[item_id] < 0).any():
            item_id = int(self.dense_ids[0])
        return np.asarray(self.sid[item_id], dtype=np.int64)

    def decode_logits(
        self,
        logits_list: list[torch.Tensor],
        fallback_items: torch.Tensor,
        top_k: int,
        device: torch.device,
        mode: str,
        rng: np.random.Generator,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = [F.log_softmax(logits.detach(), dim=-1).cpu() for logits in logits_list]
        k = int(max(1, min(top_k, log_probs[0].shape[-1])))
        top_values: list[torch.Tensor] = []
        top_indices: list[torch.Tensor] = []
        for level_log_probs in log_probs:
            values, indices = torch.topk(level_log_probs, k=k, dim=-1)
            top_values.append(values)
            top_indices.append(indices)

        batch_size = int(log_probs[0].shape[0])
        out_items: list[int] = []
        out_sids: list[np.ndarray] = []
        fallback_np = fallback_items.detach().cpu().numpy().astype(np.int64)
        for row_idx in range(batch_size):
            candidates: list[tuple[int, np.ndarray, float]] = []
            for combo in itertools.product(*[range(k) for _ in range(self.sid_levels)]):
                sid_tuple = np.asarray(
                    [int(top_indices[level][row_idx, combo[level]]) for level in range(self.sid_levels)],
                    dtype=np.int64,
                )
                dense_item_id = self.lookup_code(int((sid_tuple * self.multipliers).sum()))
                if dense_item_id is None:
                    continue
                score = float(sum(float(top_values[level][row_idx, combo[level]]) for level in range(self.sid_levels)))
                candidates.append((dense_item_id, sid_tuple, score))
            if not candidates:
                best_item = int(fallback_np[row_idx])
                best_sid = self.sid_for_item(best_item)
            elif mode == "sample":
                scores = np.asarray([item[2] for item in candidates], dtype=np.float64)
                probs = np.exp((scores - scores.max()) / max(float(temperature), 1e-6))
                probs = probs / probs.sum()
                chosen = int(rng.choice(len(candidates), p=probs))
                best_item = int(candidates[chosen][0])
                best_sid = candidates[chosen][1]
            else:
                best_item, best_sid, _ = max(candidates, key=lambda item: item[2])
            out_items.append(best_item)
            out_sids.append(best_sid.astype(np.int64))
        return (
            torch.tensor(out_items, dtype=torch.long, device=device),
            torch.tensor(np.stack(out_sids, axis=0), dtype=torch.long, device=device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSAC simulator-rollout trainer")
    parser.add_argument("--transition_root", default=str(PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans_v2_session_run_1h_span6h_run100_replay50"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "replay_val", "replay_test"])
    parser.add_argument("--item_features_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item_features.npy"))
    parser.add_argument("--dense_item2sid_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item2sid.npy"))
    parser.add_argument("--actor_init_checkpoint", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_session_run_v2_actor"))
    parser.add_argument("--simulator_checkpoint", default=str(PROJECT_ROOT / "artifacts/user_response/raw_rqkmeans_v2_session_run_simulator_decoupled_1m_e3/regret_user_response.pt"))
    parser.add_argument("--save_prefix", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_hsac_simroll_v1"))
    parser.add_argument("--save_meta", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_hsac_simroll_v1.meta.json"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--read_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--actor_decay", type=float, default=1e-5)
    parser.add_argument("--critic_decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--target_tau", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.001)
    parser.add_argument("--critic_weight", type=float, default=1.0)
    parser.add_argument("--actor_weight", type=float, default=1.0)
    parser.add_argument("--advantage_clip", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--sample_response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative_patience", type=int, default=5)
    parser.add_argument("--decode_top_k", type=int, default=4)
    parser.add_argument("--action_mode", default="sample", choices=["sample", "argmax"])
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--use_rapi", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--sara_eta", type=float, default=0.5)
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    parser.add_argument("--regret_pool_size", type=int, default=20)
    parser.add_argument("--regret_gamma", type=float, default=0.9)
    parser.add_argument("--regret_phi_scale", type=float, default=1.0)
    parser.add_argument("--regret_phi_clip", type=float, default=2.0)
    parser.add_argument("--regret_reward_threshold", type=float, default=0.0)
    parser.add_argument(
        "--memory_signal_scope",
        default="all_failed",
        choices=["all_failed", "revision_only", "rrca", "none"],
        help="Controls which simulator feedback enters B_rev. all_failed stores low_play/dislike/unlike for stronger RAPI.",
    )
    parser.add_argument("--use_precomputed_regret_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history_memory_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--base_reward_mode",
        default="no_revision_v2",
        choices=["no_revision_v2", "paper", "simulator"],
        help=(
            "Reward used by actor-critic before RRCA. no_revision_v2 keeps the current v2 play/like/dislike scale "
            "but excludes unlike/undislike; paper uses w_listen*played_ratio + w_like*like - w_dislike*dislike; "
            "simulator uses the raw simulator reward for ablation."
        ),
    )
    parser.add_argument(
        "--rrca_apply_to",
        default="effective_reward",
        choices=["effective_reward", "advantage", "off"],
        help="effective_reward follows the paper r <- r + gamma^Delta psi; advantage keeps the older A-only correction.",
    )
    parser.add_argument(
        "--rrca_signal_scope",
        default="revision_only",
        choices=["revision_only", "all_negative"],
        help="revision_only only uses unlike/undislike as psi signals; all_negative keeps the older low_play/dislike/unlike behavior.",
    )
    parser.add_argument("--reward_w_listen", type=float, default=1.0)
    parser.add_argument("--reward_w_like", type=float, default=0.8)
    parser.add_argument("--reward_w_dislike", type=float, default=1.2)
    parser.add_argument("--reward_clip_min", type=float, default=-2.0)
    parser.add_argument("--reward_clip_max", type=float, default=2.0)
    parser.add_argument("--rrca_dislike_weight", type=float, default=1.2)
    parser.add_argument("--rrca_unlike_weight", type=float, default=0.6)
    parser.add_argument("--rrca_low_play_weight", type=float, default=1.0)
    parser.add_argument("--rrca_undislike_weight", type=float, default=0.2)
    parser.add_argument(
        "--rrca_callback_target",
        default="current",
        choices=["current", "latest_same_item"],
        help="Where to apply A' correction. current matches the revised session-run definition; latest_same_item keeps the old retrospective behavior.",
    )
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


def actor_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "history_features": obs["history_features"],
        "history_feedbacks": obs["history_feedbacks"].float(),
        "history_event_type_ids": obs["history_event_type_ids"].long(),
        "history_mask": obs["history_mask"].float(),
    }


def fallback_items_from_history(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    hist = obs["history_ids"].long()
    mask = hist > 0
    counts = mask.long().sum(dim=1).clamp_min(1) - 1
    return hist.gather(1, counts.view(-1, 1)).squeeze(1).clamp_min(1)


def build_actor(args: argparse.Namespace, dense_item2sid: np.ndarray, item_dim: int, device: torch.device) -> RegretAwareSIDPolicy:
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    env = OfflineTransitionEnvSpec(int(dense_item2sid.shape[0] - 1), int(item_dim), int(args.max_seq_len))
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
    if args.actor_init_checkpoint:
        state_dict = torch.load(args.actor_init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = actor.load_state_dict(state_dict, strict=False)
        print(f"[actor] init={args.actor_init_checkpoint} missing={len(missing)} unexpected={len(unexpected)}")
    return actor


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = TransitionIterableDataset(
        Path(args.transition_root) / args.split,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=args.episodes,
        batch_size=args.read_batch_size,
        shuffle_files=True,
        shuffle_buffer_size=0,
        seed=args.seed,
        sample_across_files=False,
        regret_memory_size=args.regret_pool_size,
    )
    print(f"[data] split={args.split} files={len(dataset.files)} episodes={args.episodes}")
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def logp_entropy_from_sid(logits_list: list[torch.Tensor], sid_paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logp_sum = torch.zeros(sid_paths.shape[0], dtype=torch.float32, device=sid_paths.device)
    entropy_sum = torch.zeros_like(logp_sum)
    for level, logits_l in enumerate(logits_list):
        z_l = sid_paths[:, level].long()
        log_probs_l = F.log_softmax(logits_l, dim=-1)
        probs_l = log_probs_l.exp()
        logp_sum = logp_sum + log_probs_l.gather(1, z_l.view(-1, 1)).squeeze(1)
        entropy_sum = entropy_sum - (probs_l * log_probs_l).sum(dim=-1)
    n_level = max(len(logits_list), 1)
    return logp_sum / n_level, entropy_sum / n_level


def init_memory_from_history(memory: RegretMemoryPool, obs: dict[str, torch.Tensor], dense_item2sid: np.ndarray, device: torch.device) -> None:
    user_ids = obs["user_id"].detach().cpu().numpy().astype(np.int64)
    hist_ids = obs["history_ids"].detach().cpu().numpy().astype(np.int64)
    feedbacks = obs["history_feedbacks"].detach().cpu().numpy().astype(np.float32)
    event_ids = obs["history_event_type_ids"].detach().cpu().numpy().astype(np.int64)
    masks = obs["history_mask"].detach().cpu().numpy() > 0
    for row_idx, user_id in enumerate(user_ids):
        for col_idx in range(hist_ids.shape[1]):
            if not masks[row_idx, col_idx]:
                continue
            item_id = int(hist_ids[row_idx, col_idx])
            if item_id <= 0 or item_id >= dense_item2sid.shape[0]:
                continue
            event_id = int(event_ids[row_idx, col_idx])
            reward = float(feedbacks[row_idx, col_idx])
            if event_id == EVENT_TYPE_TO_ID["dislike"]:
                regret_type_id, strength = 2, 1.2
            elif event_id == EVENT_TYPE_TO_ID["unlike"]:
                regret_type_id, strength = 3, 0.6
            elif reward < 0.0:
                regret_type_id, strength = 1, max(1e-6, -reward)
            else:
                continue
            sid_path = torch.tensor(np.asarray(dense_item2sid[item_id], dtype=np.int64), dtype=torch.long, device=device)
            if not bool((sid_path < 0).any()):
                memory.update(int(user_id), sid_path, regret_type_id, float(strength), reward, item_id)


def init_memory_from_snapshot(
    memory: RegretMemoryPool,
    batch: dict[str, torch.Tensor],
    dense_item2sid: np.ndarray,
    device: torch.device,
) -> int:
    item_ids = batch.get("regret_memory_item_ids")
    phis = batch.get("regret_memory_phis")
    type_ids = batch.get("regret_memory_type_ids")
    if item_ids is None or phis is None or type_ids is None:
        return 0
    item_ids = item_ids.to(device=device).long()
    phis = phis.to(device=device).float()
    type_ids = type_ids.to(device=device).long()
    if not bool((item_ids > 0).any() and (phis > 0.0).any()):
        return 0
    user_ids = batch["user_id"].detach().cpu().numpy().astype(np.int64)
    loaded = 0
    sid_levels = memory.sid_levels
    for row_idx, user_id in enumerate(user_ids):
        row_item_ids = item_ids[row_idx]
        row_phis = phis[row_idx]
        row_type_ids = type_ids[row_idx]
        row_sid_paths = torch.zeros(
            row_item_ids.shape[0],
            sid_levels,
            dtype=torch.long,
            device=device,
        )
        valid_items = row_item_ids.detach().cpu().numpy().astype(np.int64)
        for col_idx, item_id in enumerate(valid_items):
            if item_id <= 0 or item_id >= dense_item2sid.shape[0]:
                continue
            sid_path = np.asarray(dense_item2sid[int(item_id)], dtype=np.int64)
            if (sid_path < 0).any():
                continue
            row_sid_paths[col_idx] = torch.tensor(sid_path[:sid_levels], dtype=torch.long, device=device)
        loaded += memory.load_snapshot(
            user_id=int(user_id),
            sid_paths=row_sid_paths,
            phis=row_phis,
            regret_type_ids=row_type_ids,
            item_ids=row_item_ids,
        )
    return int(loaded)


def clamp_reward(reward: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return reward.clamp(min=float(args.reward_clip_min), max=float(args.reward_clip_max))


def compose_base_reward(
    info: dict[str, Any],
    raw_reward: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.base_reward_mode == "simulator":
        return raw_reward.detach()

    listen = info["listen"].detach().float()
    play_ratio = info["play_ratio"].detach().float()
    feedback = info["feedback_samples"].detach().float()
    like = feedback[:, 0]
    dislike = feedback[:, 1]

    if args.base_reward_mode == "paper":
        base_reward = (
            float(args.reward_w_listen) * play_ratio
            + float(args.reward_w_like) * like
            - float(args.reward_w_dislike) * dislike
        )
    else:
        # Keep the v2 implicit-play scale but exclude revision-only signals.
        play_term = listen * (2.0 * play_ratio - 1.0)
        base_reward = play_term + float(args.reward_w_like) * like - float(args.reward_w_dislike) * dislike
    return clamp_reward(base_reward, args)


def regret_signal(
    event_type_id: int,
    reward: float,
    args: argparse.Namespace,
) -> tuple[float, float, int] | None:
    if args.rrca_apply_to == "off":
        return None
    if args.rrca_signal_scope == "revision_only":
        if event_type_id == EVENT_TYPE_TO_ID["unlike"]:
            return -float(args.rrca_unlike_weight), float(args.rrca_unlike_weight), 3
        if event_type_id == EVENT_TYPE_TO_ID["undislike"]:
            return float(args.rrca_undislike_weight), 0.0, 0
        return None

    if event_type_id == EVENT_TYPE_TO_ID["dislike"]:
        return -float(args.rrca_dislike_weight), float(args.rrca_dislike_weight), 2
    if event_type_id == EVENT_TYPE_TO_ID["unlike"]:
        return -float(args.rrca_unlike_weight), float(args.rrca_unlike_weight), 3
    if event_type_id == EVENT_TYPE_TO_ID["undislike"]:
        return float(args.rrca_undislike_weight), 0.0, 0
    if reward < 0.0:
        strength = max(1e-6, -float(reward)) * float(args.rrca_low_play_weight)
        return -strength, strength, 1
    return None


def memory_signal(
    event_type_id: int,
    reward: float,
    args: argparse.Namespace,
) -> tuple[float, int] | None:
    if args.memory_signal_scope == "none":
        return None
    if args.memory_signal_scope == "rrca":
        signal = regret_signal(event_type_id, reward, args)
        if signal is None:
            return None
        _psi_signed, phi_penalty, regret_type_id = signal
        if regret_type_id <= 0 or phi_penalty <= 0.0:
            return None
        return float(phi_penalty), int(regret_type_id)
    if event_type_id == EVENT_TYPE_TO_ID["unlike"]:
        return float(args.rrca_unlike_weight), 3
    if args.memory_signal_scope == "revision_only":
        return None
    if event_type_id == EVENT_TYPE_TO_ID["dislike"]:
        return float(args.rrca_dislike_weight), 2
    if reward < 0.0:
        return max(1e-6, -float(reward)) * float(args.rrca_low_play_weight), 1
    return None


def find_callback_record(records: list[dict[str, torch.Tensor]], row_idx: int, item_id: int, event_type_id: int) -> int:
    if event_type_id in (EVENT_TYPE_TO_ID["unlike"], EVENT_TYPE_TO_ID["undislike"]):
        for rec_idx in range(len(records) - 2, -1, -1):
            if int(records[rec_idx]["action_items"][row_idx].detach().cpu()) == int(item_id):
                return rec_idx
    return len(records) - 1


def select_with_policy(
    actor: RegretAwareSIDPolicy,
    critic: MultiLevelValueCritic,
    obs: dict[str, torch.Tensor],
    decoder: SIDDecoder,
    args: argparse.Namespace,
    rng: np.random.Generator,
    memory: RegretMemoryPool | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feed = actor_obs(obs)
    if memory is not None:
        pool_tokens, pool_phis = memory.get(obs["user_id"].long(), obs["history_features"].device)
        out = actor.get_sara_logits(feed, pool_tokens=pool_tokens, pool_phis=pool_phis)
    else:
        out = actor(feed)
    action_items, sid_paths = decoder.decode_logits(
        out["sid_logits"],
        fallback_items=fallback_items_from_history(obs),
        top_k=args.decode_top_k,
        device=obs["history_features"].device,
        mode=args.action_mode,
        rng=rng,
        temperature=args.action_temperature,
    )
    logp, entropy = logp_entropy_from_sid(out["sid_logits"], sid_paths)
    critic_out = critic({"context_list": out["context_list"]})
    value = critic_out["q"]
    return out, action_items, sid_paths, logp, entropy, value, critic_out["v_seq"]


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    tau = float(tau)
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def aggregate_metrics(totals: dict[str, float], metrics: dict[str, float], weight: float = 1.0) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value) * float(weight)


def train_rollout_batch(
    actor: RegretAwareSIDPolicy,
    critic: MultiLevelValueCritic,
    target_critic: MultiLevelValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    env: RegretUserResponseEnv,
    batch: dict[str, torch.Tensor],
    decoder: SIDDecoder,
    dense_item2sid: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, float]:
    obs = env.reset_from_batch(batch)
    batch_size = int(obs["history_ids"].shape[0])
    active = torch.ones(batch_size, dtype=torch.bool, device=env.device)
    records: list[dict[str, torch.Tensor]] = []
    memory: RegretMemoryPool | None = None
    if args.use_rapi:
        sid_levels, _ = infer_sid_spec(dense_item2sid)
        memory = RegretMemoryPool(
            pool_size=args.regret_pool_size,
            sid_levels=sid_levels,
            gamma=args.regret_gamma,
            phi_scale=args.regret_phi_scale,
            phi_clip=args.regret_phi_clip,
            reward_threshold=args.regret_reward_threshold,
            negative_type_ids=(1, 2, 3),
        )
        loaded_snapshot = 0
        if args.use_precomputed_regret_memory:
            loaded_snapshot = init_memory_from_snapshot(memory, batch, dense_item2sid, env.device)
        if loaded_snapshot <= 0 and args.history_memory_fallback:
            init_memory_from_history(memory, obs, dense_item2sid, env.device)

    callback_count = 0.0
    callback_low_play_count = 0.0
    callback_dislike_count = 0.0
    callback_unlike_count = 0.0
    callback_undislike_count = 0.0
    memory_insert_count = 0.0
    memory_signal_count = 0.0
    memory_low_play_count = 0.0
    memory_dislike_count = 0.0
    memory_unlike_count = 0.0
    correction_pos_sum = 0.0
    correction_neg_sum = 0.0
    callback_delta_sum = 0.0
    callback_to_past_count = 0.0
    raw_reward_sum = 0.0
    base_reward_sum = 0.0
    effective_reward_sum = 0.0
    step_count = 0.0
    raw_negative_count = 0.0
    base_negative_count = 0.0
    effective_negative_count = 0.0
    for _step in range(int(args.max_steps)):
        if not bool(active.any()):
            break
        _out, action_items, sid_paths, logp, entropy, value, v_seq = select_with_policy(
            actor, critic, env.current_observation, decoder, args, rng, memory
        )
        _, reward, done, info = env.step(action_items)
        base_reward = compose_base_reward(info, reward, args)
        active_f = active.float()
        record = {
            "active": active.detach().clone(),
            "action_items": action_items.detach().clone(),
            "sid_paths": sid_paths.detach().clone(),
            "logp": logp,
            "entropy": entropy,
            "value": value,
            "v_seq": v_seq,
            "raw_reward": reward.detach(),
            "base_reward": base_reward.detach(),
            "done": done.detach().float(),
            "reward_correction": torch.zeros(batch_size, dtype=torch.float32, device=env.device),
            "advantage_correction": torch.zeros(batch_size, dtype=torch.float32, device=env.device),
        }
        records.append(record)

        event_ids = info["event_type_id"].detach().cpu().numpy().astype(np.int64)
        base_rewards_np = base_reward.detach().cpu().numpy().astype(np.float32)
        action_np = action_items.detach().cpu().numpy().astype(np.int64)
        users_np = env.current_observation["user_id"].detach().cpu().numpy().astype(np.int64)
        for row_idx in torch.nonzero(active.detach().cpu(), as_tuple=False).view(-1).tolist():
            signal = regret_signal(int(event_ids[row_idx]), float(base_rewards_np[row_idx]), args)
            target_idx = len(records) - 1
            delta_t = 0
            if signal is not None:
                psi_signed, _phi_penalty, regret_type_id = signal
                if args.rrca_callback_target == "latest_same_item":
                    target_idx = find_callback_record(records, row_idx, int(action_np[row_idx]), int(event_ids[row_idx]))
                else:
                    # The revised session-run definition updates the tuple where the revision signal occurs.
                    target_idx = len(records) - 1
                delta_t = max(0, len(records) - 1 - target_idx)
                signed_correction = (float(args.gamma) ** delta_t) * float(psi_signed)
                if args.rrca_apply_to == "effective_reward":
                    records[target_idx]["reward_correction"][row_idx] += signed_correction
                elif args.rrca_apply_to == "advantage":
                    records[target_idx]["advantage_correction"][row_idx] += signed_correction
                callback_count += 1.0
                callback_delta_sum += float(delta_t)
                if delta_t > 0:
                    callback_to_past_count += 1.0
                if regret_type_id == 1:
                    callback_low_play_count += 1.0
                elif regret_type_id == 2:
                    callback_dislike_count += 1.0
                elif regret_type_id == 3:
                    callback_unlike_count += 1.0
                elif int(event_ids[row_idx]) == EVENT_TYPE_TO_ID["undislike"]:
                    callback_undislike_count += 1.0
                if signed_correction > 0.0:
                    correction_pos_sum += float(signed_correction)
                elif signed_correction < 0.0:
                    correction_neg_sum += float(signed_correction)

            mem_signal = memory_signal(int(event_ids[row_idx]), float(base_rewards_np[row_idx]), args)
            if memory is not None and mem_signal is not None:
                phi_penalty, memory_regret_type_id = mem_signal
                if memory_regret_type_id <= 0 or phi_penalty <= 0.0:
                    continue
                target_sid = records[-1]["sid_paths"][row_idx]
                inserted = memory.update(
                    user_id=int(users_np[row_idx]),
                    sid_path=target_sid,
                    regret_type_id=int(memory_regret_type_id),
                    regret_strength=float(phi_penalty),
                    reward=float(base_rewards_np[row_idx]),
                    item_id=int(action_np[row_idx]),
                )
                memory_signal_count += 1.0
                memory_insert_count += float(inserted)
                if memory_regret_type_id == 1:
                    memory_low_play_count += 1.0
                elif memory_regret_type_id == 2:
                    memory_dislike_count += 1.0
                elif memory_regret_type_id == 3:
                    memory_unlike_count += 1.0

        with torch.no_grad():
            if memory is not None:
                pool_tokens, pool_phis = memory.get(env.current_observation["user_id"].long(), env.device)
                next_out = actor.get_sara_logits(actor_obs(env.current_observation), pool_tokens=pool_tokens, pool_phis=pool_phis)
            else:
                next_out = actor(actor_obs(env.current_observation))
            next_value = target_critic({"context_list": next_out["context_list"]})["q"]
        record["next_value"] = next_value.detach()

        raw_reward_sum += float((reward.detach() * active_f).sum().detach().cpu())
        base_reward_sum += float((base_reward.detach() * active_f).sum().detach().cpu())
        step_count += float(active_f.sum().detach().cpu())
        raw_negative_count += float(((reward.detach() < 0.0).float() * active_f).sum().detach().cpu())
        base_negative_count += float(((base_reward.detach() < 0.0).float() * active_f).sum().detach().cpu())
        active = active & ~done.bool()

    if not records:
        return {"n": 0.0}

    losses_actor: list[torch.Tensor] = []
    losses_critic: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    adv_values: list[torch.Tensor] = []
    corr_values: list[torch.Tensor] = []
    n_active = 0.0
    for rec in records:
        mask = rec["active"]
        if not bool(mask.any()):
            continue
        if args.rrca_apply_to == "effective_reward":
            step_reward = clamp_reward(rec["base_reward"] + rec["reward_correction"], args)
            correction_for_log = rec["reward_correction"]
        else:
            step_reward = rec["base_reward"]
            correction_for_log = rec["advantage_correction"]
        target = step_reward + float(args.gamma) * (1.0 - rec["done"]) * rec["next_value"]
        advantage = (target - rec["value"]).detach()
        corrected_advantage = advantage
        if args.rrca_apply_to == "advantage":
            corrected_advantage = advantage + rec["advantage_correction"]
        if float(args.advantage_clip) > 0:
            corrected_for_loss = corrected_advantage.clamp(-float(args.advantage_clip), float(args.advantage_clip))
        else:
            corrected_for_loss = corrected_advantage
        losses_actor.append(-(corrected_for_loss[mask] * rec["logp"][mask]).mean())
        losses_critic.append(F.mse_loss(rec["value"][mask], target.detach()[mask]))
        entropies.append(rec["entropy"][mask].mean())
        adv_values.append(advantage[mask].detach())
        corr_values.append(correction_for_log[mask].detach())
        mask_n = float(mask.float().sum().detach().cpu())
        n_active += mask_n
        effective_reward_sum += float(step_reward[mask].sum().detach().cpu())
        effective_negative_count += float((step_reward[mask] < 0.0).float().sum().detach().cpu())

    actor_loss_raw = torch.stack(losses_actor).mean()
    critic_loss = torch.stack(losses_critic).mean()
    entropy = torch.stack(entropies).mean()
    actor_loss = actor_loss_raw - float(args.entropy_weight) * entropy
    total_loss = float(args.actor_weight) * actor_loss + float(args.critic_weight) * critic_loss

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    total_loss.backward()
    if float(args.grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), float(args.grad_clip))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), float(args.grad_clip))
    actor_optimizer.step()
    critic_optimizer.step()
    soft_update(target_critic, critic, float(args.target_tau))

    adv_cat = torch.cat(adv_values) if adv_values else torch.zeros(1, device=env.device)
    corr_cat = torch.cat(corr_values) if corr_values else torch.zeros(1, device=env.device)
    token_weights = F.softmax(critic.token_weight.detach(), dim=0)
    metrics = {
        "n": n_active,
        "loss": float(total_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "adv_mean": float(adv_cat.mean().detach().cpu()),
        "adv_std": float(adv_cat.std(unbiased=False).detach().cpu()),
        "adv_pos_share": float((adv_cat > 0.0).float().mean().detach().cpu()),
        "correction_mean": float(corr_cat.mean().detach().cpu()),
        "correction_abs_mean": float(corr_cat.abs().mean().detach().cpu()),
        "correction_pos_per_step": correction_pos_sum / max(n_active, 1.0),
        "correction_neg_per_step": correction_neg_sum / max(n_active, 1.0),
        "callback_per_step": callback_count / max(n_active, 1.0),
        "callback_low_play_per_step": callback_low_play_count / max(n_active, 1.0),
        "callback_dislike_per_step": callback_dislike_count / max(n_active, 1.0),
        "callback_unlike_per_step": callback_unlike_count / max(n_active, 1.0),
        "callback_undislike_per_step": callback_undislike_count / max(n_active, 1.0),
        "callback_delta_mean": callback_delta_sum / max(callback_count, 1.0),
        "callback_to_past_per_step": callback_to_past_count / max(n_active, 1.0),
        "memory_signal_per_step": memory_signal_count / max(n_active, 1.0),
        "memory_low_play_per_step": memory_low_play_count / max(n_active, 1.0),
        "memory_dislike_per_step": memory_dislike_count / max(n_active, 1.0),
        "memory_unlike_per_step": memory_unlike_count / max(n_active, 1.0),
        "memory_insert_per_step": memory_insert_count / max(n_active, 1.0),
        "reward_per_step": effective_reward_sum / max(n_active, 1.0),
        "effective_reward_per_step": effective_reward_sum / max(n_active, 1.0),
        "base_reward_per_step": base_reward_sum / max(step_count, 1.0),
        "raw_reward_per_step": raw_reward_sum / max(step_count, 1.0),
        "negative_rate": effective_negative_count / max(n_active, 1.0),
        "effective_negative_rate": effective_negative_count / max(n_active, 1.0),
        "base_negative_rate": base_negative_count / max(step_count, 1.0),
        "raw_negative_rate": raw_negative_count / max(step_count, 1.0),
    }
    for idx, value in enumerate(token_weights.detach().cpu().tolist()):
        metrics[f"mlc_w{idx}"] = float(value)
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    print(f"[device] using {device}")
    print(f"[sid] levels={sid_levels} vocab={sid_vocab_size} item_dim={feature_shape[1]}")
    decoder = SIDDecoder(dense_item2sid, sid_vocab_size)
    actor = build_actor(args, dense_item2sid, int(feature_shape[1]), device)
    critic = MultiLevelValueCritic(args, actor).to(device)
    target_critic = deepcopy(critic).to(device)
    target_critic.eval()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)
    transition_path = Path(args.transition_root) / args.split
    env = RegretUserResponseEnv(
        checkpoint_path=args.simulator_checkpoint,
        transition_path=transition_path,
        dense_item_features_npy=args.item_features_npy,
        max_seq_len=args.max_seq_len,
        device=str(device),
        max_step_per_episode=args.max_steps,
        sample_response=args.sample_response,
        negative_patience=args.negative_patience,
        seed=args.seed,
    )
    history: list[dict[str, Any]] = []
    save_prefix = Path(args.save_prefix)
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_meta).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        critic.train()
        loader = make_loader(args)
        totals: dict[str, float] = {}
        n_total = 0.0
        pbar = tqdm(loader, desc=f"[epoch {epoch}] hsac", ncols=120)
        for batch in pbar:
            metrics = train_rollout_batch(
                actor, critic, target_critic, actor_optimizer, critic_optimizer,
                env, batch, decoder, dense_item2sid, args, rng,
            )
            if metrics.get("n", 0.0) <= 0:
                continue
            n = float(metrics["n"])
            n_total += n
            aggregate_metrics(totals, {k: v for k, v in metrics.items() if k != "n"}, n)
            view = {k: totals[k] / max(n_total, 1.0) for k in totals}
            pbar.set_postfix(
                loss=view.get("loss", 0.0),
                eff_r=view.get("effective_reward_per_step", 0.0),
                base_r=view.get("base_reward_per_step", 0.0),
                neg=view.get("effective_negative_rate", 0.0),
                adv=view.get("adv_mean", 0.0),
                cb=view.get("callback_per_step", 0.0),
            )
        pbar.close()
        epoch_metrics = {"n": n_total, **{k: v / max(n_total, 1.0) for k, v in totals.items()}}
        history.append({"epoch": epoch, "train": epoch_metrics})
        print(
            f"[epoch {epoch}] loss={epoch_metrics.get('loss', float('nan')):.5f} "
            f"eff_reward={epoch_metrics.get('effective_reward_per_step', float('nan')):.4f} "
            f"base_reward={epoch_metrics.get('base_reward_per_step', float('nan')):.4f} "
            f"raw_reward={epoch_metrics.get('raw_reward_per_step', float('nan')):.4f} "
            f"neg={epoch_metrics.get('negative_rate', float('nan')):.4f} "
            f"adv={epoch_metrics.get('adv_mean', float('nan')):.4f} "
            f"adv_pos={epoch_metrics.get('adv_pos_share', float('nan')):.3f} "
            f"cb={epoch_metrics.get('callback_per_step', float('nan')):.4f} "
            f"cb_low={epoch_metrics.get('callback_low_play_per_step', float('nan')):.4f} "
            f"cb_unlike={epoch_metrics.get('callback_unlike_per_step', float('nan')):.4f} "
            f"cb_undislike={epoch_metrics.get('callback_undislike_per_step', float('nan')):.4f} "
            f"cb_dt={epoch_metrics.get('callback_delta_mean', float('nan')):.3f} "
            f"mem={epoch_metrics.get('memory_insert_per_step', float('nan')):.4f} "
            f"mem_low={epoch_metrics.get('memory_low_play_per_step', float('nan')):.4f} "
            f"mem_dislike={epoch_metrics.get('memory_dislike_per_step', float('nan')):.4f} "
            f"mem_unlike={epoch_metrics.get('memory_unlike_per_step', float('nan')):.4f}"
        )

    torch.save(actor.state_dict(), str(save_prefix) + "_actor")
    torch.save(critic.state_dict(), str(save_prefix) + "_critic")
    meta = {
        "args": vars(args),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "history": history,
        "actor_checkpoint": str(save_prefix) + "_actor",
        "critic_checkpoint": str(save_prefix) + "_critic",
    }
    Path(args.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] actor saved to {save_prefix}_actor")
    print(f"[done] critic saved to {save_prefix}_critic")
    print(f"[done] meta saved to {args.save_meta}")


if __name__ == "__main__":
    main()
