"""Jumping Knowledge GCN with concatenation, max, or LSTM aggregation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import GraphConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use JKNet.") from exc


class JKNetEncoder(nn.Module):
    """GCN stack with layer-wise Jumping Knowledge aggregation.

    The default ``cat`` mode preserves the aggregation style used in the source
    notebook. ``max`` and ``lstm`` are exposed as configurable alternatives.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 32,
        num_layers: int = 4,
        mode: str = "cat",
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if mode not in {"cat", "max", "lstm"}:
            raise ValueError("mode must be 'cat', 'max', or 'lstm'")
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [GraphConv(in_features, hidden_features, allow_zero_in_degree=True)]
            + [GraphConv(hidden_features, hidden_features, allow_zero_in_degree=True) for _ in range(num_layers - 1)]
        )
        if mode == "cat":
            aggregate_features = hidden_features * num_layers
        elif mode == "max":
            aggregate_features = hidden_features
        else:
            lstm_hidden = max(1, hidden_features // 2)
            self.lstm = nn.LSTM(
                input_size=hidden_features,
                hidden_size=lstm_hidden,
                bidirectional=True,
                batch_first=True,
            )
            self.attention = nn.Linear(2 * lstm_hidden, 1)
            aggregate_features = 2 * lstm_hidden
        self.output = nn.Linear(aggregate_features, out_features)

    def forward(self, graph, features):
        layer_outputs = []
        hidden = features
        for layer in self.layers:
            hidden = self.dropout(F.relu(layer(graph, hidden)))
            layer_outputs.append(hidden)

        stacked = torch.stack(layer_outputs, dim=1)  # [nodes, layers, hidden]
        if self.mode == "cat":
            aggregate = torch.cat(layer_outputs, dim=-1)
        elif self.mode == "max":
            aggregate = stacked.max(dim=1).values
        else:
            sequence, _ = self.lstm(stacked)
            attention = torch.softmax(self.attention(sequence), dim=1)
            aggregate = torch.sum(sequence * attention, dim=1)
        return self.output(aggregate)
