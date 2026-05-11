#!/usr/bin/env python3
"""历史 replay 评估：按真实 transition 回放用户失败记忆池，测 RAPI 对 logged action 的影响。

当前主评估是 simulator rollout；本文件保留用于对照。
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.transition_dataset import TransitionIterableDataset  # noqa: E402
from regret_core.model.regret_memory import RegretMemoryPool  # noqa: E402
from regret_core.model.sid_policy import RegretAwareSIDPolicy  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Evaluate regret-pool policy intervention by log replay")
    parser.add_argument("--transition_root", default=str(PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans_v2_smoke_timefix"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "replay_val", "replay_test"])
    parser.add_argument("--item_features_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item_features.npy"))
    parser.add_argument("--dense_item2sid_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/dense_item2sid.npy"))
    parser.add_argument("--actor_checkpoint", default=str(PROJECT_ROOT / "artifacts/models/regret_sid_feedback_actor"))
    parser.add_argument("--save_meta", default=str(PROJECT_ROOT / "artifacts/models/regret_pool_replay.meta.json"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1, help="Use 1 for exact per-user sequential pool updates.")
    parser.add_argument("--read_batch_size", type=int, default=2048)
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--use_history_feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_history_event_type", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_type_vocab_size", type=int, default=8)
    parser.add_argument("--sara_eta", type=float, default=0.5)
    parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
    parser.add_argument("--regret_pool_size", type=int, default=20)
    parser.add_argument("--regret_gamma", type=float, default=0.9)
    parser.add_argument("--regret_phi_scale", type=float, default=1.0)
    parser.add_argument("--regret_phi_clip", type=float, default=2.0)
    parser.add_argument("--regret_reward_threshold", type=float, default=0.0)
    parser.add_argument("--negative_type_ids", default="1,2,3", help="Comma ids: low_play=1, dislike=2, unlike=3.")
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


def build_actor(args: argparse.Namespace, dense_item2sid: np.ndarray, item_dim: int, device: torch.device) -> RegretAwareSIDPolicy:
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    env = OfflineTransitionEnvSpec(
        n_item=int(dense_item2sid.shape[0] - 1),
        item_dim=int(item_dim),
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
    ckpt_path = Path(args.actor_checkpoint)
    if ckpt_path.exists():
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)
        missing, unexpected = actor.load_state_dict(state_dict, strict=False)
        print(f"[actor] loaded={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[actor] checkpoint not found, evaluating random actor: {ckpt_path}")
    actor.eval()
    return actor


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


def path_log_prob(logits_list: list[torch.Tensor], target_sid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = torch.zeros(target_sid.shape[0], device=target_sid.device)
    full_match = torch.ones(target_sid.shape[0], dtype=torch.bool, device=target_sid.device)
    token_hits = []
    for level, logits_l in enumerate(logits_list):
        z_l = target_sid[:, level]
        log_probs_l = F.log_softmax(logits_l, dim=-1)
        logp = logp + log_probs_l.gather(1, z_l.view(-1, 1)).squeeze(1)
        pred_l = logits_l.argmax(dim=-1)
        hit_l = pred_l == z_l
        token_hits.append(hit_l.float())
        full_match = full_match & hit_l
    token_acc = torch.stack(token_hits, dim=1).mean(dim=1)
    return logp, token_acc, full_match.float()


def make_loader(args: argparse.Namespace) -> DataLoader:
    dataset = TransitionIterableDataset(
        Path(args.transition_root) / args.split,
        args.item_features_npy,
        max_seq_len=args.max_seq_len,
        max_rows=args.max_rows,
        batch_size=args.read_batch_size,
        shuffle_files=False,
        shuffle_buffer_size=0,
        seed=args.seed,
        sample_across_files=False,
    )
    print(f"[data] split={args.split} files={len(dataset.files)} max_rows={args.max_rows} batch_size={args.batch_size}")
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


def add_weighted(totals: dict[str, float], prefix: str, values: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
    n = float(mask.float().sum().item())
    if n <= 0:
        return
    totals[f"{prefix}_n"] = totals.get(f"{prefix}_n", 0.0) + n
    for key, value in values.items():
        totals[f"{prefix}_{key}"] = totals.get(f"{prefix}_{key}", 0.0) + float(value[mask].sum().detach().cpu())


def finalize_group(totals: dict[str, float], prefix: str, keys: list[str]) -> dict[str, float]:
    n = totals.get(f"{prefix}_n", 0.0)
    out = {f"{prefix}_n": n}
    for key in keys:
        raw = totals.get(f"{prefix}_{key}", 0.0)
        out[f"{prefix}_{key}"] = raw / n if n > 0 else 0.0
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    feature_shape = np.load(args.item_features_npy, mmap_mode="r").shape
    actor = build_actor(args, dense_item2sid, int(feature_shape[1]), device)
    negative_type_ids = tuple(int(x) for x in args.negative_type_ids.split(",") if x.strip())
    memory = RegretMemoryPool(
        pool_size=args.regret_pool_size,
        sid_levels=sid_levels,
        gamma=args.regret_gamma,
        phi_scale=args.regret_phi_scale,
        phi_clip=args.regret_phi_clip,
        reward_threshold=args.regret_reward_threshold,
        negative_type_ids=negative_type_ids,
    )
    loader = make_loader(args)
    totals: dict[str, float] = {}
    keys = [
        "base_nll",
        "pool_nll",
        "suppression",
        "base_token_acc",
        "pool_token_acc",
        "base_full_acc",
        "pool_full_acc",
        "pool_active",
        "pool_phi",
    ]

    pbar = tqdm(loader, desc="[replay]", ncols=120)
    for batch in pbar:
        with torch.no_grad():
            target_sid, valid = target_sid_from_dense_ids(dense_item2sid, batch["target_dense_item_id"], device, sid_vocab_size)
            if not bool(valid.any()):
                continue
            obs = move_observation(batch, valid, device)
            target_sid = target_sid[valid]
            user_ids = batch["user_id"].to(device).long()[valid]
            reward = batch["reward"].to(device).float()[valid]
            regret_type_id = batch["regret_type_id"].to(device).long()[valid]
            regret_strength = batch["regret_strength"].to(device).float()[valid]
            item_ids = batch["target_dense_item_id"].to(device).long()[valid]

            pool_tokens, pool_phis = memory.get(user_ids, device)
            base_out = actor(obs)
            pool_out = actor.get_sara_logits(obs, pool_tokens=pool_tokens, pool_phis=pool_phis)
            base_logp, base_token_acc, base_full = path_log_prob(base_out["sid_logits"], target_sid)
            pool_logp, pool_token_acc, pool_full = path_log_prob(pool_out["sid_logits"], target_sid)
            pool_active = (pool_phis > 0).float().sum(dim=1)
            pool_phi = pool_phis.sum(dim=1)
            values = {
                "base_nll": -base_logp,
                "pool_nll": -pool_logp,
                "suppression": base_logp - pool_logp,
                "base_token_acc": base_token_acc,
                "pool_token_acc": pool_token_acc,
                "base_full_acc": base_full,
                "pool_full_acc": pool_full,
                "pool_active": pool_active,
                "pool_phi": pool_phi,
            }
            neg_mask = torch.zeros_like(reward, dtype=torch.bool)
            for type_id in negative_type_ids:
                neg_mask = neg_mask | (regret_type_id == int(type_id))
            neg_mask = neg_mask | (reward < args.regret_reward_threshold)
            pos_mask = ~neg_mask
            all_mask = torch.ones_like(neg_mask, dtype=torch.bool)
            add_weighted(totals, "all", values, all_mask)
            add_weighted(totals, "neg", values, neg_mask)
            add_weighted(totals, "pos", values, pos_mask)

            for row_idx in range(target_sid.shape[0]):
                memory.update(
                    user_id=int(user_ids[row_idx].detach().cpu()),
                    sid_path=target_sid[row_idx],
                    regret_type_id=int(regret_type_id[row_idx].detach().cpu()),
                    regret_strength=float(regret_strength[row_idx].detach().cpu()),
                    reward=float(reward[row_idx].detach().cpu()),
                    item_id=int(item_ids[row_idx].detach().cpu()),
                )

        all_n = totals.get("all_n", 0.0)
        pbar.set_postfix(
            n=int(all_n),
            neg_supp=finalize_group(totals, "neg", ["suppression"]).get("neg_suppression", 0.0),
            pos_supp=finalize_group(totals, "pos", ["suppression"]).get("pos_suppression", 0.0),
            pool_users=memory.active_user_count(),
        )
    pbar.close()

    metrics = {
        **finalize_group(totals, "all", keys),
        **finalize_group(totals, "neg", keys),
        **finalize_group(totals, "pos", keys),
        "memory": memory.summary(),
    }
    meta = {
        "args": vars(args),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "metrics": metrics,
    }
    Path(args.save_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        "[done] "
        f"all_n={metrics['all_n']:.0f} "
        f"neg_n={metrics['neg_n']:.0f} "
        f"neg_suppression={metrics['neg_suppression']:.5f} "
        f"pos_suppression={metrics['pos_suppression']:.5f}"
    )
    print(f"[done] meta saved to {args.save_meta}")


if __name__ == "__main__":
    main()
