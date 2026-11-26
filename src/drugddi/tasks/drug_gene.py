"""Helpers for building an R-GCN representation model on drug/gene heterographs."""

from __future__ import annotations

from torch import nn

from drugddi.models.hetero_rgcn import HeteroNodeEmbedding, HeteroRGCNEncoder


class DrugGeneRGCN(nn.Module):
    """Featureless-node heterogeneous R-GCN baseline.

    This is suitable when drug/gene nodes have IDs but no aligned feature matrix.
    The caller supplies a DGL heterograph with semantically meaningful edge
    types. It returns one embedding tensor per node type; task-specific heads
    (classification or typed link prediction) are intentionally separate.
    """

    def __init__(
        self,
        graph,
        embedding_dim: int = 64,
        hidden_features: int = 64,
        out_features: int = 32,
        num_layers: int = 2,
        num_bases: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        node_counts = {node_type: graph.num_nodes(node_type) for node_type in graph.ntypes}
        self.node_embedding = HeteroNodeEmbedding(node_counts, embedding_dim)
        self.encoder = HeteroRGCNEncoder(
            relation_names=graph.etypes,
            in_features=embedding_dim,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
        )

    def forward(self, graph):
        return self.encoder(graph, self.node_embedding())
