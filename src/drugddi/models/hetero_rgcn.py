"""Heterogeneous R-GCN layers following DGL's relation-aware message passing API.

The implementation is a modern, compact port of DGL's ``rgcn-hetero`` example.
It accepts a DGL heterograph and a feature tensor for each node type.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    import dgl.nn as dglnn
except ImportError as exc:  # pragma: no cover - requires DGL runtime
    raise ImportError("Install the DGL dependencies to use heterogeneous R-GCN.") from exc


class HeteroRGCNLayer(nn.Module):
    """One relation-specific convolution layer for a DGL heterograph."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        relation_names,
        num_bases: Optional[int] = None,
        *,
        activation=None,
        self_loop: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.relation_names = tuple(sorted(set(relation_names)))
        if not self.relation_names:
            raise ValueError("A heterogeneous R-GCN needs at least one edge relation")
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_self_loop = self_loop
        if num_bases is not None and num_bases < 1:
            raise ValueError("num_bases must be >= 1 when provided")
        self.num_bases = min(num_bases or len(self.relation_names), len(self.relation_names))
        self.conv = dglnn.HeteroGraphConv(
            {
                relation: dglnn.GraphConv(
                    in_features, out_features, norm="right", weight=False, bias=False,
                    allow_zero_in_degree=True,
                )
                for relation in self.relation_names
            },
            aggregate="sum",
        )
        if self.num_bases < len(self.relation_names):
            self.basis = dglnn.WeightBasis(
                (in_features, out_features), self.num_bases, len(self.relation_names)
            )
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(
                torch.empty(len(self.relation_names), in_features, out_features)
            )
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
            self.basis = None
        self.bias = nn.Parameter(torch.zeros(out_features))
        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(in_features, out_features))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
        else:
            self.register_parameter("loop_weight", None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        graph = graph.local_var()
        relation_weights = self.basis() if self.basis is not None else self.weight
        module_kwargs = {
            relation: {"weight": relation_weights[index]}
            for index, relation in enumerate(self.relation_names)
        }
        outputs = self.conv(graph, dict(features), mod_kwargs=module_kwargs)
        result = {}
        for node_type in graph.ntypes:
            if node_type not in outputs:
                # A node type without incoming edges still receives its self-loop/bias.
                reference = features[node_type]
                value = reference.new_zeros((reference.shape[0], self.out_features))
            else:
                value = outputs[node_type]
            if self.use_self_loop:
                value = value + features[node_type] @ self.loop_weight
            value = value + self.bias
            if self.activation is not None:
                value = self.activation(value)
            result[node_type] = self.dropout(value)
        return result


class HeteroNodeEmbedding(nn.Module):
    """Learnable per-node-type embeddings for featureless heterogeneous graphs."""

    def __init__(self, node_counts: Mapping[str, int], embedding_dim: int):
        super().__init__()
        if not node_counts or any(int(count) < 1 for count in node_counts.values()):
            raise ValueError("node_counts must contain positive counts for every node type")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.node_types = tuple(sorted(node_counts))
        self.embeddings = nn.ParameterList()
        for node_type in self.node_types:
            parameter = nn.Parameter(torch.empty(int(node_counts[node_type]), embedding_dim))
            nn.init.xavier_uniform_(parameter, gain=nn.init.calculate_gain("relu"))
            self.embeddings.append(parameter)

    def forward(self) -> Dict[str, torch.Tensor]:
        return {
            node_type: self.embeddings[index]
            for index, node_type in enumerate(self.node_types)
        }


class HeteroRGCNEncoder(nn.Module):
    """Stacked heterogeneous R-GCN encoder for drug/gene/entity graphs.

    ``features`` must map each DGL node type to a tensor with shape
    ``[number_of_nodes_of_type, in_features]``. This class deliberately does
    not invent node features or relation labels; callers supply both the
    heterograph schema and node features.
    """

    def __init__(
        self,
        relation_names,
        in_features: int,
        hidden_features: int = 64,
        out_features: int = 32,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if min(in_features, hidden_features, out_features) < 1:
            raise ValueError("feature dimensions must be >= 1")
        dimensions = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        self.layers = nn.ModuleList(
            [
                HeteroRGCNLayer(
                    dimensions[index],
                    dimensions[index + 1],
                    relation_names,
                    num_bases,
                    activation=F.relu if index < num_layers - 1 else None,
                    self_loop=True,
                    dropout=dropout if index < num_layers - 1 else 0.0,
                )
                for index in range(num_layers)
            ]
        )

    def forward(self, graph, features: Mapping[str, torch.Tensor]):
        hidden = dict(features)
        for layer in self.layers:
            hidden = layer(graph, hidden)
        return hidden
