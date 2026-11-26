"""Evaluation metrics for binary link prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_link_metrics(positive_logits: torch.Tensor, negative_logits: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute ROC-AUC, AP, precision, recall and F1 from positive/negative logits."""
    positive = positive_logits.detach().float().cpu().numpy().reshape(-1)
    negative = negative_logits.detach().float().cpu().numpy().reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both positive and negative evaluation scores must be non-empty")
    logits = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(positive.size), np.zeros(negative.size)]).astype(np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    predictions = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
