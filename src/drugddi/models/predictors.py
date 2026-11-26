"""Trainable and parameter-free edge decoders shared by all DDI encoders."""

from __future__ import annotations

import torch
from torch import nn


class DotProductPredictor(nn.Module):
    """Dot-product edge decoder used by the original notebook."""

    def forward(self, node_embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2:
            raise ValueError("edge_pairs must have shape [2, num_edges]")
        source, target = edge_pairs.long()
        return (node_embeddings[source] * node_embeddings[target]).sum(dim=-1)


class MLPEdgePredictor(nn.Module):
    """Symmetric MLP decoder using elementwise product and absolute difference."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2:
            raise ValueError("edge_pairs must have shape [2, num_edges]")
        source, target = edge_pairs.long()
        source_embedding = node_embeddings[source]
        target_embedding = node_embeddings[target]
        symmetric_features = torch.cat(
            [source_embedding * target_embedding, torch.abs(source_embedding - target_embedding)],
            dim=-1,
        )
        return self.network(symmetric_features).squeeze(-1)
