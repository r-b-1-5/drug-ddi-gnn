"""GraphSAGE encoder using DGL's neighborhood aggregation layers."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import SAGEConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use the GraphSAGE model.") from exc


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder; mean aggregation matches the original notebook."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 32,
        aggregator_type: str = "mean",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_features, aggregator_type)
        self.conv2 = SAGEConv(hidden_features, out_features, aggregator_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        hidden = F.relu(self.conv1(graph, features))
        hidden = self.dropout(hidden)
        return self.conv2(graph, hidden)
