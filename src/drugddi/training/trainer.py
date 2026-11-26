"""Full-batch transductive training for DDI link prediction."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from drugddi.evaluation.metrics import compute_link_metrics
from drugddi.models.predictors import DotProductPredictor, MLPEdgePredictor
from drugddi.models.vgae import VGAEEncoder


@dataclass
class TrainingConfig:
    epochs: int = 200
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    patience: int = 30
    predictor: str = "dot"
    kl_weight: float = 1.0
    threshold: float = 0.5
    seed: int = 42
    log_every: int = 10

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinkPredictionTrainer:
    """Train a DGL node encoder with a shared edge decoder.

    Validation/test edges are held out from the message-passing graph. The
    training positives remain in the training graph, as in the original
    transductive notebook's graph autoencoding setup.
    """

    def __init__(
        self,
        model: nn.Module,
        graph,
        features: torch.Tensor,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.graph = graph.to(self.device)
        self.features = features.to(self.device)
        if self.config.predictor not in {"dot", "mlp"}:
            raise ValueError("predictor must be 'dot' or 'mlp'")
        if isinstance(self.model, VGAEEncoder) and self.config.predictor != "dot":
            raise ValueError("VGAE uses its inner-product decoder; set predictor='dot'")
        output_dim = int(embedding_dim) if embedding_dim is not None else self._infer_output_dim()
        if output_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.predictor = (
            DotProductPredictor()
            if self.config.predictor == "dot"
            else MLPEdgePredictor(output_dim, dropout=0.1)
        ).to(self.device)
        parameters = list(self.model.parameters()) + list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        set_seed(self.config.seed)

    def _infer_output_dim(self) -> int:
        # Model encoders expose output width as the final projection/layer.
        if hasattr(self.model, "output") and hasattr(self.model.output, "out_features"):
            return int(self.model.output.out_features)
        if hasattr(self.model, "conv2") and hasattr(self.model.conv2, "_out_feats"):
            return int(self.model.conv2._out_feats)
        if hasattr(self.model, "conv2") and hasattr(self.model.conv2, "out_feats"):
            return int(self.model.conv2.out_feats)
        if hasattr(self.model, "mean") and hasattr(self.model.mean, "_out_feats"):
            return int(self.model.mean._out_feats)
        if hasattr(self.model, "mean") and hasattr(self.model.mean, "out_feats"):
            return int(self.model.mean.out_feats)
        # A dry-run is safer than guessing dimensions for future encoder classes.
        with torch.no_grad():
            encoded = self.model(self.graph, self.features)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            return int(encoded.shape[-1])

    @staticmethod
    def _edge_tensor(edge_array, device):
        return torch.as_tensor(np.asarray(edge_array).T.copy(), dtype=torch.long, device=device)

    def _encode(self):
        encoded = self.model(self.graph, self.features)
        if isinstance(self.model, VGAEEncoder):
            latent, mean, log_std = encoded
            return latent, mean, log_std
        return encoded, None, None

    def _score(self, embeddings, edges):
        if isinstance(self.model, VGAEEncoder):
            return self.model.decode(embeddings, edges)
        return self.predictor(embeddings, edges)

    def _loss(self, positive_logits, negative_logits, mean=None, log_std=None):
        logits = torch.cat([positive_logits, negative_logits], dim=0)
        labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0
        )
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if isinstance(self.model, VGAEEncoder) and mean is not None and log_std is not None:
            loss = loss + self.config.kl_weight * self.model.kl_divergence(mean, log_std)
        return loss

    def _evaluate(self, positive_edges, negative_edges) -> Dict[str, float]:
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            embeddings, _, _ = self._encode()
            positive_logits = self._score(embeddings, self._edge_tensor(positive_edges, self.device))
            negative_logits = self._score(embeddings, self._edge_tensor(negative_edges, self.device))
            metrics = compute_link_metrics(
                positive_logits, negative_logits, threshold=self.config.threshold
            )
        return metrics

    def fit(self, split) -> Dict[str, object]:
        """Train with validation ROC-AUC early stopping and restore best weights."""
        train_pos = self._edge_tensor(split.train_positive, self.device)
        train_neg = self._edge_tensor(split.train_negative, self.device)
        best_auc = float("-inf")
        best_epoch = 0
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_predictor_state = copy.deepcopy(self.predictor.state_dict())
        history = []
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.predictor.train()
            embeddings, mean, log_std = self._encode()
            positive_logits = self._score(embeddings, train_pos)
            negative_logits = self._score(embeddings, train_neg)
            loss = self._loss(positive_logits, negative_logits, mean, log_std)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            validation = self._evaluate(split.validation_positive, split.validation_negative)
            row = {"epoch": epoch, "loss": float(loss.detach().cpu()), **validation}
            history.append(row)

            if validation["roc_auc"] > best_auc:
                best_auc = validation["roc_auc"]
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_predictor_state = copy.deepcopy(self.predictor.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.config.log_every > 0 and (epoch == 1 or epoch % self.config.log_every == 0):
                print(
                    f"epoch={epoch:03d} loss={row['loss']:.4f} "
                    f"val_auc={row['roc_auc']:.4f} val_ap={row['average_precision']:.4f}"
                )
            if epochs_without_improvement >= self.config.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

        self.model.load_state_dict(best_model_state)
        self.predictor.load_state_dict(best_predictor_state)
        test_metrics = self._evaluate(split.test_positive, split.test_negative)
        result = {
            "best_epoch": best_epoch,
            "best_validation_roc_auc": best_auc,
            "test_metrics": test_metrics,
            "history": history,
            "training_config": asdict(self.config),
        }
        return result
