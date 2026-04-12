#!/usr/bin/env python3
"""
最小 Yambda-HSRL SID 训练入口。

它把当前 baseline 的核心组件串起来：
- YambdaEnvironment_GPU_HAC: 用 05 的 UserResponse 给 reward
- SIDPolicy_credit: HPN / Actor
- Token_Critic: MLC / Critic
- SIDFacade_credit: semantic action -> candidate item
- DDPG: 当前仓库里已有的 actor-critic 更新器

它是正式 baseline 的单文件入口；推荐通过 run_stage.sh 读取集中配置来运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapter.bootstrap import install_hsrl_adapter  # noqa: E402

install_hsrl_adapter()

from env.YambdaEnvironment_GPU_HAC import YambdaEnvironment_GPU_HAC  # type: ignore  # noqa: E402
from model.agents.DDPG import DDPG  # type: ignore  # noqa: E402
from model.critic.Token_Critic import Token_Critic  # type: ignore  # noqa: E402
from model.facade.SIDFacade_credit import SIDFacade_credit  # type: ignore  # noqa: E402
from model.policy.SIDPolicy_credit import SIDPolicy_credit  # type: ignore  # noqa: E402
from utils import set_random_seed  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Train minimal Yambda SID HSRL baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--urm_log_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/env/log/yambda_user_env.model.log"),
    )
    parser.add_argument(
        "--dense_item2sid_npy",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/mappings/yambda_dense_item2sid.npy"),
    )
    parser.add_argument(
        "--hpn_checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/hpn_warmstart.pt"),
    )
    parser.add_argument("--slate_size", type=int, default=1)
    parser.add_argument("--max_candidate_items", type=int, default=50000)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--start_timestamp", type=int, default=2000)
    parser.add_argument("--episode_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--train_every_n_step", type=int, default=5)
    parser.add_argument("--check_episode", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--actor_decay", type=float, default=1e-5)
    parser.add_argument("--critic_decay", type=float, default=1e-5)
    parser.add_argument("--target_mitigate_coef", type=float, default=0.01)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--bc_coef", type=float, default=0.1)
    parser.add_argument("--initial_greedy_epsilon", type=float, default=0.0)
    parser.add_argument("--final_greedy_epsilon", type=float, default=0.0)
    parser.add_argument("--elbow_greedy", type=float, default=0.5)
    parser.add_argument("--sasrec_n_layer", type=int, default=2)
    parser.add_argument("--sasrec_d_model", type=int, default=64)
    parser.add_argument("--sasrec_d_forward", type=int, default=128)
    parser.add_argument("--sasrec_n_head", type=int, default=4)
    parser.add_argument("--sasrec_dropout", type=float, default=0.1)
    parser.add_argument("--sid_temp", type=float, default=1.0)
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--critic_dropout_rate", type=float, default=0.2)
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/yambda_sid"),
    )
    parser.add_argument(
        "--save_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/models/yambda_sid.meta.json"),
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """输入：设备字符串。输出：torch.device。"""
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def infer_sid_spec(dense_item2sid: np.ndarray) -> tuple[int, int]:
    """输入：dense_item2sid 数组。输出：SID 层数和每层 vocab size。"""
    valid = dense_item2sid[1:]
    sid_levels = int(valid.shape[1])
    vocab_sizes = [int(valid[:, i].max()) + 1 for i in range(sid_levels)]
    if len(set(vocab_sizes)) != 1:
        raise ValueError(f"Inconsistent per-level SID vocab sizes: {vocab_sizes}")
    return sid_levels, int(vocab_sizes[0])


def load_hpn_checkpoint(actor: SIDPolicy_credit, ckpt_path: Path, device: torch.device) -> bool:
    """输入：actor、checkpoint 路径、设备。输出：是否成功加载。"""
    if not ckpt_path.exists():
        print(f"[hpn] checkpoint not found, use random init: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    print(f"[hpn] loaded {ckpt_path}, missing={len(missing)}, unexpected={len(unexpected)}")
    return True


def main() -> None:
    """主入口：跑一个最小 Yambda SID-HSRL baseline 训练。"""
    args = parse_args()
    save_meta = Path(args.save_meta)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    save_meta.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[device] using {device}")
    set_random_seed(args.seed)

    dense_item2sid = np.load(args.dense_item2sid_npy, mmap_mode="r")
    sid_levels, sid_vocab_size = infer_sid_spec(dense_item2sid)
    print(f"[sid] sid_levels={sid_levels}, sid_vocab_size={sid_vocab_size}")

    env_args = SimpleNamespace(
        env_path="",
        reward_func="direct_score",
        max_step_per_episode=10,
        initial_temper=5,
        urm_log_path=args.urm_log_path,
        temper_sweet_point=0.9,
        temper_prob_lag=100,
        device=str(device),
    )
    env = YambdaEnvironment_GPU_HAC(env_args)

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
    actor = SIDPolicy_credit(policy_args, env).to(device)
    hpn_loaded = load_hpn_checkpoint(actor, Path(args.hpn_checkpoint), device)

    critic_args = SimpleNamespace(
        critic_hidden_dims=args.critic_hidden_dims,
        critic_dropout_rate=args.critic_dropout_rate,
    )
    critic = Token_Critic(critic_args, env, actor).to(device)

    facade_args = SimpleNamespace(
        device=str(device),
        slate_size=args.slate_size,
        buffer_size=args.buffer_size,
        start_timestamp=args.start_timestamp,
        noise_var=0.0,
        n_iter=[args.n_iter],
        q_laplace_smoothness=0.5,
        topk_rate=1.0,
        empty_start_rate=0.0,
        item2sid=args.dense_item2sid_npy,
        candidate_ids_npy="",
        max_candidate_items=args.max_candidate_items,
        candidate_seed=args.seed,
    )
    facade = SIDFacade_credit(facade_args, env, actor, critic)

    agent_args = SimpleNamespace(
        device=str(device),
        gamma=args.gamma,
        n_iter=[args.n_iter],
        train_every_n_step=args.train_every_n_step,
        initial_greedy_epsilon=args.initial_greedy_epsilon,
        final_greedy_epsilon=args.final_greedy_epsilon,
        elbow_greedy=args.elbow_greedy,
        check_episode=args.check_episode,
        with_eval=False,
        save_path=args.save_path,
        use_wandb=False,
        wandb_project="yambda_sid",
        wandb_name=None,
        episode_batch_size=args.episode_batch_size,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        actor_decay=args.actor_decay,
        critic_decay=args.critic_decay,
        target_mitigate_coef=args.target_mitigate_coef,
        entropy_coef=args.entropy_coef,
        bc_coef=args.bc_coef,
    )
    agent = DDPG(agent_args, facade)
    agent.train()

    meta = {
        "training_args": vars(args),
        "device": str(device),
        "sid_levels": sid_levels,
        "sid_vocab_size": sid_vocab_size,
        "hpn_checkpoint": args.hpn_checkpoint,
        "hpn_loaded": hpn_loaded,
        "candidate_count": int(facade.candidate_iids.numel()),
        "buffer_size": int(facade.current_buffer_size),
        "n_iter": int(args.n_iter),
        "save_path": args.save_path,
    }
    save_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] train meta saved to {save_meta}")


if __name__ == "__main__":
    main()
