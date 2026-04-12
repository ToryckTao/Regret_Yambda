from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_regularization


class SIDPolicy_credit(nn.Module):
    """
    HPN actor: history_features [B,H,item_dim] -> SID logits list[(B,V_l)].

    context_list 输出形状为 [B,L+1,d_model]，供 Token_Critic 做多层 value。
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--sasrec_n_layer", type=int, default=2)
        parser.add_argument("--sasrec_d_model", type=int, default=64)
        parser.add_argument("--sasrec_d_forward", type=int, default=128)
        parser.add_argument("--sasrec_n_head", type=int, default=4)
        parser.add_argument("--sasrec_dropout", type=float, default=0.1)
        parser.add_argument("--sid_levels", type=int, default=3)
        parser.add_argument("--sid_vocab_sizes", type=int, default=256)
        parser.add_argument("--sid_temp", type=float, default=1.0)
        parser.add_argument("--sara_eta", type=float, default=0.5)
        parser.add_argument("--sara_layer_weights", type=str, default="0.05,0.25,0.70")
        return parser

    def __init__(self, args, environment):
        super().__init__()
        self.n_layer = args.sasrec_n_layer
        self.d_model = args.sasrec_d_model
        self.n_head = args.sasrec_n_head
        self.dropout = args.sasrec_dropout
        self.sid_temp = float(getattr(args, "sid_temp", 1.0))

        self.n_item = environment.action_space["item_id"][1]
        self.item_dim = environment.action_space["item_feature"][1]
        self.maxlen = environment.observation_space["history"][1]
        self.state_dim = self.d_model
        self.action_dim = self.d_model

        self.item_map = nn.Linear(self.item_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.maxlen, self.d_model)
        self.emb_drop = nn.Dropout(self.dropout)
        self.emb_norm = nn.LayerNorm(self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=args.sasrec_d_forward,
            nhead=self.n_head,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layer)

        self.register_buffer("pos_idx", torch.arange(self.maxlen, dtype=torch.long), persistent=False)
        full_mask = torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.register_buffer("attn_mask_full", ~full_mask, persistent=False)

        self.sid_levels = int(args.sid_levels)
        self.sid_vocab_sizes = [args.sid_vocab_sizes] * self.sid_levels
        self.sara_eta = getattr(args, "sara_eta", 0.5)
        weights_str = getattr(args, "sara_layer_weights", "0.05,0.25,0.70")
        self.sara_layer_weights = [float(w) for w in weights_str.split(",")]

        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, v) for v in self.sid_vocab_sizes])
        self.sid_token_embeds = nn.ModuleList([nn.Embedding(v, self.d_model) for v in self.sid_vocab_sizes])
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

    def encode_state(self, feed_dict):
        """输入 history_features=[B,H,item_dim]；输出 state_emb=[B,d_model]。"""
        hist = feed_dict["history_features"]
        batch_size, hist_len, _ = hist.shape
        pos = self.pos_emb(self.pos_idx[:hist_len]).unsqueeze(0).expand(batch_size, hist_len, -1)
        x = self.item_map(hist)
        x = self.emb_norm(self.emb_drop(x + pos))
        attn_mask = self.attn_mask_full[:hist_len, :hist_len]
        out_seq = self.transformer(x, mask=attn_mask)
        state = out_seq[:, -1, :]
        return {"output_seq": out_seq, "state_emb": state}

    def forward(self, feed_dict):
        enc = self.encode_state(feed_dict)
        context = enc["state_emb"]
        sid_logits = []
        context_list = [context]
        tau = self.sid_temp if self.sid_temp is not None else 1.0

        for level in range(self.sid_levels):
            logits_l = self.sid_heads[level](context)
            sid_logits.append(logits_l)
            probs_l = torch.softmax(logits_l / tau, dim=-1)
            exp_emb = probs_l @ self.sid_token_embeds[level].weight
            context = self.sid_res_norms[level](context - exp_emb)
            context_list.append(context)

        return {
            "sid_logits": sid_logits,
            "context_list": torch.stack(context_list, dim=1),
            "seq_emb": enc["output_seq"],
            "reg": get_regularization(self.item_map, self.transformer),
        }

    def get_sara_logits(self, feed_dict, pool_tokens=None, pool_phis=None):
        """保留 SARA/RAPI 后续扩展入口；baseline forward 不调用它。"""
        enc = self.encode_state(feed_dict)
        context = enc["state_emb"]
        batch_size = context.shape[0]
        sid_logits = []
        context_list = [context]
        vocab_dim = self.sid_vocab_sizes[0]
        n_level = self.sid_levels
        tau = self.sid_temp
        current_soft_prefix = torch.zeros(batch_size, n_level, vocab_dim, device=context.device)

        if pool_tokens is not None and pool_tokens.shape[0] > 0:
            pool_size = pool_tokens.shape[0]
            pool_one_hot = F.one_hot(pool_tokens, num_classes=vocab_dim).float()
        else:
            pool_tokens = None
            pool_size = 0
            pool_one_hot = None

        weights = torch.tensor(self.sara_layer_weights, device=context.device)
        for level in range(n_level):
            logits_l = self.sid_heads[level](context)
            if pool_tokens is not None:
                if level == 0:
                    similarity_vec = torch.ones(batch_size, pool_size, device=context.device)
                else:
                    current_slice = current_soft_prefix[:, :level, :]
                    pool_slice = pool_one_hot[:, :level, :]
                    hit_matrix = (current_slice.unsqueeze(1) * pool_slice.unsqueeze(0)).sum(dim=-1)
                    similarity_vec = (hit_matrix * weights[:level].view(1, 1, -1)).sum(dim=-1)
                active_penalty = similarity_vec * pool_phis.unsqueeze(0)
                penalty = torch.matmul(active_penalty, pool_one_hot[:, level, :])
                logits_l = logits_l - self.sara_eta * penalty

            sid_logits.append(logits_l)
            probs_l = F.softmax(logits_l / tau, dim=-1)
            current_soft_prefix[:, level, :] = probs_l
            exp_emb = torch.matmul(probs_l, self.sid_token_embeds[level].weight)
            context = self.sid_res_norms[level](context - exp_emb)
            context_list.append(context)

        return {
            "sid_logits": sid_logits,
            "context_list": torch.stack(context_list, dim=1),
            "output_seq": enc["output_seq"],
        }
