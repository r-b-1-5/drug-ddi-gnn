"""Variational Graph Autoencoder (VGAE) for unsupervised link prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from dgl.nn import GraphConv
except ImportError as exc:  # pragma: no cover - requires DGL environment
    raise ImportError("Install the DGL extra to use VGAE.") from exc


class VGAEEncoder(nn.Module):
    """DGL GraphConv implementation of a variational graph encoder.

    The latent decoder is an inner product, following Kipf & Welling's VGAE.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 64,
        latent_features: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared = GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.mean = GraphConv(hidden_features, latent_features, allow_zero_in_degree=True)
        self.log_std = GraphConv(hidden_features, latent_features, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        hidden = F.relu(self.shared(graph, features))
        hidden = self.dropout(hidden)
        mean = self.mean(graph, hidden)
        log_std = self.log_std(graph, hidden).clamp(min=-10.0, max=10.0)
        if self.training:
            latent = mean + torch.randn_like(log_std) * torch.exp(log_std)
        else:
            latent = mean
        return latent, mean, log_std

    @staticmethod
    def decode(node_embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        source, target = edge_pairs.long()
        return (node_embeddings[source] * node_embeddings[target]).sum(dim=-1)

    @staticmethod
    def kl_divergence(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        # Mean-normalized KL(q(z|X,A) || N(0,I)); log_std is log standard deviation.
        return -0.5 * torch.mean(
            torch.sum(1.0 + 2.0 * log_std - mean.pow(2) - torch.exp(2.0 * log_std), dim=1)
        )
