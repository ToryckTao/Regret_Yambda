from __future__ import annotations

import torch.nn as nn


class DNN(nn.Module):
    """输入 [B,in_dim]；输出 [B,out_dim] 的轻量 MLP。"""

    def __init__(self, in_dim, hidden_dims, out_dim=1, dropout_rate=0.0, do_batch_norm=True):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            if do_batch_norm:
                layers.append(nn.LayerNorm([hidden_dim]))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.in_dim)
        return self.layers(inputs)
