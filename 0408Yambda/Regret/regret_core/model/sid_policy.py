from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegretAwareSIDPolicy(nn.Module):
    """SID actor with behavior-aware history encoding.

    The state encoder keeps the standard sequential item stream, but augments
    each history position with its feedback value and event type. This does not
    change the recommendation action space: the policy still predicts a
    hierarchical SID path for the next item.
    """

    def __init__(self, args, environment) -> None:
        super().__init__()
        self.n_layer = int(args.sasrec_n_layer)
        self.d_model = int(args.sasrec_d_model)
        self.n_head = int(args.sasrec_n_head)
        self.dropout = float(args.sasrec_dropout)
        self.sid_temp = float(getattr(args, "sid_temp", 1.0))
        self.use_history_feedback = bool(getattr(args, "use_history_feedback", True))
        self.use_history_event_type = bool(getattr(args, "use_history_event_type", True))

        self.n_item = environment.action_space["item_id"][1]
        self.item_dim = environment.action_space["item_feature"][1]
        self.maxlen = environment.observation_space["history"][1]
        self.state_dim = self.d_model
        self.action_dim = self.d_model

        self.item_map = nn.Linear(self.item_dim, self.d_model)
        self.feedback_map = nn.Linear(1, self.d_model, bias=False)
        self.event_type_emb = nn.Embedding(int(getattr(args, "event_type_vocab_size", 8)), self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.d_model)
        self.emb_drop = nn.Dropout(self.dropout)
        self.emb_norm = nn.LayerNorm(self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=int(args.sasrec_d_forward),
            nhead=self.n_head,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layer)
        self.register_buffer("pos_idx", torch.arange(self.maxlen, dtype=torch.long), persistent=False)
        full_mask = torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.register_buffer("attn_mask_full", ~full_mask, persistent=False)

        self.sid_levels = int(args.sid_levels)
        self.sid_vocab_sizes = [int(args.sid_vocab_sizes)] * self.sid_levels
        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, v) for v in self.sid_vocab_sizes])
        self.sid_token_embeds = nn.ModuleList([nn.Embedding(v, self.d_model) for v in self.sid_vocab_sizes])
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

        self.sara_eta = float(getattr(args, "sara_eta", 0.5))
        weights = [float(x) for x in str(getattr(args, "sara_layer_weights", "0.05,0.25,0.70")).split(",") if x]
        if len(weights) < self.sid_levels:
            weights = weights + [weights[-1] if weights else 1.0] * (self.sid_levels - len(weights))
        self.sara_layer_weights = weights[: self.sid_levels]

    def encode_state(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hist = feed_dict["history_features"]
        batch_size, hist_len, _ = hist.shape
        pos = self.pos_emb(self.pos_idx[:hist_len]).unsqueeze(0).expand(batch_size, hist_len, -1)
        x = self.item_map(hist) + pos

        if self.use_history_feedback and "history_feedbacks" in feed_dict:
            feedback = feed_dict["history_feedbacks"].to(hist.device, dtype=hist.dtype).unsqueeze(-1)
            x = x + self.feedback_map(feedback)

        if self.use_history_event_type and "history_event_type_ids" in feed_dict:
            event_ids = feed_dict["history_event_type_ids"].to(hist.device).long()
            event_ids = event_ids.clamp(min=0, max=self.event_type_emb.num_embeddings - 1)
            x = x + self.event_type_emb(event_ids)

        x = self.emb_norm(self.emb_drop(x))
        attn_mask = self.attn_mask_full[:hist_len, :hist_len]
        # Do not pass a key padding mask here: left padding plus a causal mask
        # can create fully masked padded queries and produce NaNs. Padding still
        # has zero item/feedback and pad event embeddings, matching the previous
        # item-only SID policy behavior.
        out_seq = self.transformer(x, mask=attn_mask)
        return {"output_seq": out_seq, "state_emb": out_seq[:, -1, :]}

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        enc = self.encode_state(feed_dict)
        context = enc["state_emb"]
        sid_logits = []
        context_list = [context]
        tau = max(float(self.sid_temp), 1e-6)
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
            "reg": torch.zeros((), device=context.device),
        }

    def get_sara_logits(self, feed_dict, pool_tokens=None, pool_phis=None):
        enc = self.encode_state(feed_dict)
        context = enc["state_emb"]
        batch_size = context.shape[0]
        sid_logits = []
        context_list = [context]
        vocab_dim = self.sid_vocab_sizes[0]
        n_level = self.sid_levels
        tau = max(float(self.sid_temp), 1e-6)
        current_soft_prefix = torch.zeros(batch_size, n_level, vocab_dim, device=context.device)

        if pool_tokens is not None and pool_phis is not None and pool_tokens.numel() > 0:
            pool_tokens = pool_tokens.to(device=context.device, dtype=torch.long)
            pool_phis = pool_phis.to(device=context.device, dtype=context.dtype)
            if pool_tokens.dim() == 2:
                pool_tokens = pool_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            if pool_phis.dim() == 1:
                pool_phis = pool_phis.unsqueeze(0).expand(batch_size, -1)
            pool_tokens = pool_tokens[:, :, :n_level].clamp(min=0, max=vocab_dim - 1)
            pool_one_hot = F.one_hot(pool_tokens, num_classes=vocab_dim).to(dtype=context.dtype)
            pool_size = pool_tokens.shape[1]
        else:
            pool_tokens = None
            pool_one_hot = None
            pool_size = 0

        weights = torch.tensor(self.sara_layer_weights, device=context.device, dtype=context.dtype)
        for level in range(n_level):
            logits_l = self.sid_heads[level](context)
            if pool_tokens is not None and pool_one_hot is not None:
                if level == 0:
                    similarity_vec = torch.ones(batch_size, pool_size, device=context.device, dtype=context.dtype)
                else:
                    current_slice = current_soft_prefix[:, :level, :]
                    pool_slice = pool_one_hot[:, :, :level, :]
                    hit_matrix = (current_slice.unsqueeze(1) * pool_slice).sum(dim=-1)
                    similarity_vec = (hit_matrix * weights[:level].view(1, 1, -1)).sum(dim=-1)
                active_penalty = similarity_vec * pool_phis
                penalty = (active_penalty.unsqueeze(-1) * pool_one_hot[:, :, level, :]).sum(dim=1)
                # Paper notation applies +eta * D. Here D is the negative
                # intervention score of failed semantic overlap, so this is
                # equivalent to subtracting a positive penalty.
                intervention_score = -penalty
                logits_l = logits_l + self.sara_eta * intervention_score
            sid_logits.append(logits_l)
            probs_l = F.softmax(logits_l / tau, dim=-1)
            current_soft_prefix[:, level, :] = probs_l
            exp_emb = probs_l @ self.sid_token_embeds[level].weight
            context = self.sid_res_norms[level](context - exp_emb)
            context_list.append(context)
        return {
            "sid_logits": sid_logits,
            "context_list": torch.stack(context_list, dim=1),
            "seq_emb": enc["output_seq"],
        }


class TokenRewardCritic(nn.Module):
    """Small value head over policy context_list."""

    def __init__(self, args, policy: RegretAwareSIDPolicy) -> None:
        super().__init__()
        hidden_dims = list(getattr(args, "critic_hidden_dims", [256, 64]))
        dropout = float(getattr(args, "critic_dropout_rate", 0.2))
        layers: list[nn.Module] = []
        in_dim = policy.state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.token_weight = nn.Parameter(torch.ones(policy.sid_levels + 1))

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context_tensor = feed_dict["context_list"]
        batch_size, n_level_plus_1, d_model = context_tensor.shape
        context_flat = context_tensor.reshape(batch_size * n_level_plus_1, d_model)
        v_flat = self.net(context_flat).view(batch_size, n_level_plus_1)
        weights = F.softmax(self.token_weight, dim=0)
        q = (v_flat * weights.unsqueeze(0)).sum(dim=1)
        return {"v_seq": v_flat, "q": q, "reg": torch.zeros((), device=q.device)}


class MultiLevelValueCritic(TokenRewardCritic):
    """MLC critic: per-level values V_phi(s,l) plus learnable aggregation."""


class SIDActionQCritic(nn.Module):
    """Action-conditioned Q(s, a) critic for logged SID paths.

    Unlike TokenRewardCritic, this module receives the logged target SID tokens
    explicitly, so Q can vary across candidate actions under the same state.
    """

    def __init__(self, args, policy: RegretAwareSIDPolicy) -> None:
        super().__init__()
        hidden_dims = list(getattr(args, "critic_hidden_dims", [256, 64]))
        dropout = float(getattr(args, "critic_dropout_rate", 0.2))
        self.sid_levels = int(policy.sid_levels)
        self.sid_vocab_sizes = list(policy.sid_vocab_sizes)
        self.action_token_embeds = nn.ModuleList(
            [nn.Embedding(vocab_size, policy.state_dim) for vocab_size in self.sid_vocab_sizes]
        )
        in_dim = policy.state_dim * (self.sid_levels + 1)
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "state_emb" in feed_dict:
            state_emb = feed_dict["state_emb"]
        else:
            state_emb = feed_dict["context_list"][:, 0, :]
        target_sid = feed_dict["target_sid"].to(device=state_emb.device).long()
        action_parts = []
        for level, embed in enumerate(self.action_token_embeds):
            token_l = target_sid[:, level].clamp(min=0, max=embed.num_embeddings - 1)
            action_parts.append(embed(token_l))
        x = torch.cat([state_emb, *action_parts], dim=-1)
        q = self.net(x).squeeze(-1)
        return {"q": q, "reg": torch.zeros((), device=q.device)}


class StateValueCritic(nn.Module):
    """State-value baseline V(s) used for advantage-weighted policy updates."""

    def __init__(self, args, policy: RegretAwareSIDPolicy) -> None:
        super().__init__()
        hidden_dims = list(getattr(args, "critic_hidden_dims", [256, 64]))
        dropout = float(getattr(args, "critic_dropout_rate", 0.2))
        layers: list[nn.Module] = []
        in_dim = policy.state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "state_emb" in feed_dict:
            state_emb = feed_dict["state_emb"]
        else:
            state_emb = feed_dict["context_list"][:, 0, :]
        v = self.net(state_emb).squeeze(-1)
        return {"v": v, "reg": torch.zeros((), device=v.device)}
