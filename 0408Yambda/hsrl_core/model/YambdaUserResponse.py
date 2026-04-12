from __future__ import annotations

import torch
import torch.nn as nn

from model.components import DNN
from model.general import BaseModel


class YambdaUserResponse(BaseModel):
    """UserResponse: history + exposure item -> continuous feedback reward."""

    @staticmethod
    def parse_model_args(parser):
        parser = BaseModel.parse_model_args(parser)
        parser.add_argument("--feature_dim", type=int, default=32)
        parser.add_argument("--attn_n_head", type=int, default=4)
        parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128])
        parser.add_argument("--dropout_rate", type=float, default=0.2)
        return parser

    def __init__(self, args, reader, device):
        super().__init__(args, reader, device)
        self.mse_loss = nn.MSELoss(reduction="none")

    def _define_params(self, args, reader):
        stats = reader.get_statistics()
        self.portrait_len = stats["user_portrait_len"]
        self.item_dim = stats["item_vec_size"]
        self.feature_dim = args.feature_dim

        if self.portrait_len > 0:
            self.portrait_encoding_layer = DNN(
                self.portrait_len,
                args.hidden_dims,
                args.feature_dim,
                dropout_rate=args.dropout_rate,
                do_batch_norm=False,
            )
        else:
            self.portrait_encoding_layer = nn.Linear(1, args.feature_dim, bias=False)
            self.portrait_encoding_layer.weight.data.fill_(0.0)

        self.item_emb_layer = nn.Linear(self.item_dim, args.feature_dim)
        self.seq_self_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first=True)
        self.seq_user_attn_layer = nn.MultiheadAttention(args.feature_dim, args.attn_n_head, batch_first=True)

    def get_forward(self, feed_dict):
        if self.portrait_len > 0:
            user_emb = self.portrait_encoding_layer(feed_dict["user_profile"]).view(-1, 1, self.feature_dim)
        else:
            batch_size = feed_dict["history"].shape[0]
            user_emb = torch.zeros(batch_size, 1, self.feature_dim, device=feed_dict["history"].device)

        history_item_emb = self.item_emb_layer(feed_dict["history_features"])
        seq_encoding, _ = self.seq_self_attn_layer(history_item_emb, history_item_emb, history_item_emb)
        user_interest, _ = self.seq_user_attn_layer(user_emb, seq_encoding, seq_encoding)
        exposure_item_emb = self.item_emb_layer(feed_dict["exposure_features"])

        score = torch.sum(exposure_item_emb * user_interest, dim=-1)
        reg = self.get_regularization(
            self.portrait_encoding_layer,
            self.item_emb_layer,
            self.seq_user_attn_layer,
            self.seq_self_attn_layer,
        )
        return {"preds": score, "reg": reg}

    def get_loss(self, feed_dict, out_dict):
        preds = out_dict["preds"].view(-1)
        reg = out_dict["reg"]
        target = feed_dict["feedback"].view(-1).to(torch.float)
        return torch.mean(self.mse_loss(preds, target)) + self.l2_coef * reg
