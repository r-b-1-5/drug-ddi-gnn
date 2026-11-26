"""Relational GCN encoder for graphs with typed edges."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import RelGraphConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use R-GCN.") from exc


class RGCNEncoder(nn.Module):
    """Two-layer relational GCN.

    Input is a homogeneous DGL graph whose edges carry integer relation IDs in
    ``edge_type`` (or supplied as the second forward argument). This matches the
    relational-message-passing requirement of DGL's R-GCN examples. It is not
    trained on the binary DDI CSV without meaningful relation labels.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if min(in_features, hidden_features, out_features, num_relations) < 1:
            raise ValueError("feature dimensions and num_relations must be >= 1")
        if num_bases is not None and num_bases < 1:
            raise ValueError("num_bases must be >= 1 when provided")
        basis_count = min(num_bases or num_relations, num_relations)
        self.layer1 = RelGraphConv(
            in_features, hidden_features, num_relations,
            regularizer="basis", num_bases=basis_count,
            activation=None, self_loop=True, dropout=dropout,
        )
        self.layer2 = RelGraphConv(
            hidden_features, out_features, num_relations,
            regularizer="basis", num_bases=basis_count,
            activation=None, self_loop=True, dropout=dropout,
        )

    def forward(self, graph, features, edge_types=None):
        if edge_types is None:
            if "etype" not in graph.edata:
                raise ValueError("R-GCN requires integer relation IDs in graph.edata['etype']")
            edge_types = graph.edata["etype"]
        hidden = F.relu(self.layer1(graph, features, edge_types))
        return self.layer2(graph, hidden, edge_types)
