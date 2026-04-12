from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import DNN
from utils import get_regularization


class Token_Critic(nn.Module):
    """MLC critic: context_list [B,L+1,d] -> q [B] and v_seq [B,L+1]."""

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[128])
        parser.add_argument("--critic_dropout_rate", type=float, default=0.2)
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net = DNN(
            self.state_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True,
        )
        self.token_weight = nn.Parameter(torch.ones(getattr(policy, "sid_levels", 3) + 1))

    def forward(self, feed_dict):
        context_tensor = feed_dict["context_list"]
        batch_size, n_level_plus_1, d_model = context_tensor.shape
        context_flat = context_tensor.view(batch_size * n_level_plus_1, d_model)
        v_flat = self.net(context_flat).view(batch_size, n_level_plus_1)
        weights = F.softmax(self.token_weight, dim=0)
        v_weighted = (v_flat * weights.unsqueeze(0)).sum(dim=1)
        return {"v_seq": v_flat, "q": v_weighted, "reg": get_regularization(self.net)}
