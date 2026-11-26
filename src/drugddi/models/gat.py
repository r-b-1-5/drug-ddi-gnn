"""Graph Attention Network encoder implemented with DGL GATConv."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import GATConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use the GAT model.") from exc


class GATEncoder(nn.Module):
    """Multi-head GAT with a single-head output layer for link prediction."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        self.dropout = nn.Dropout(dropout)
        self.attention1 = GATConv(
            in_features,
            hidden_features,
            num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=False,
            allow_zero_in_degree=True,
        )
        self.attention2 = GATConv(
            hidden_features * num_heads,
            out_features,
            1,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=False,
            allow_zero_in_degree=True,
        )

    def forward(self, graph, features):
        hidden = self.attention1(graph, features).flatten(start_dim=1)
        hidden = self.dropout(F.elu(hidden))
        output = self.attention2(graph, hidden)
        return output.squeeze(1)
