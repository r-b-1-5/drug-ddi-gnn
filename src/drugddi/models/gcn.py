"""Graph Convolutional Network encoder implemented with DGL GraphConv."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import GraphConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use the GCN model.") from exc


class GCNEncoder(nn.Module):
    """Two-layer GCN matching the baseline structure in the source notebook."""

    def __init__(self, in_features: int, hidden_features: int = 64, out_features: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        hidden = F.relu(self.conv1(graph, features))
        hidden = self.dropout(hidden)
        return self.conv2(graph, hidden)
