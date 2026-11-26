"""Sparse, reproducible edge splitting and negative sampling for DDI link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeSplit:
    """Positive and negative edges for train/validation/test.

    Edge arrays have shape ``(num_edges, 2)`` and store undirected pairs in
    canonical ``source < target`` order.
    """

    train_positive: np.ndarray
    validation_positive: np.ndarray
    test_positive: np.ndarray
    train_negative: np.ndarray
    validation_negative: np.ndarray
    test_negative: np.ndarray


def canonicalize_edges(edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Canonicalize undirected edges, remove self-loops and duplicate pairs."""
    pairs = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Each edge must have 2 endpoints; got {edge!r}")
        source, target = int(edge[0]), int(edge[1])
        if source == target:
            continue
        pairs.add((min(source, target), max(source, target)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _sample_negative_edges(
    num_nodes: int,
    count: int,
    forbidden: Set[Tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unique undirected non-edges without constructing a dense matrix."""
    if count <= 0:
        return np.empty((0, 2), dtype=np.int64)
    possible = num_nodes * (num_nodes - 1) // 2
    available = possible - len(forbidden)
    if count > available:
        raise ValueError(
            f"Requested {count} negative pairs but only {available} are available"
        )

    chosen: Set[Tuple[int, int]] = set()
    # Rejection sampling remains efficient for sparse graphs; use larger
    # batches to avoid Python-level work on every proposed pair.
    while len(chosen) < count:
        remaining = count - len(chosen)
        batch_size = max(1024, remaining * 3)
        sources = rng.integers(0, num_nodes, size=batch_size)
        targets = rng.integers(0, num_nodes, size=batch_size)
        for source, target in zip(sources, targets):
            if source == target:
                continue
            pair = (int(min(source, target)), int(max(source, target)))
            if pair in forbidden or pair in chosen:
                continue
            chosen.add(pair)
            if len(chosen) == count:
                break
    return np.asarray(sorted(chosen), dtype=np.int64)


def make_link_prediction_split(
    num_nodes: int,
    positive_edges: Iterable[Sequence[int]],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> EdgeSplit:
    """Split known interactions and draw disjoint negatives using sparse sets.

    All known interactions are excluded from negative sampling. Positive
    interactions are deduplicated as undirected pairs before the split, avoiding
    the duplicate/reverse-edge leakage possible in the original dense-adjacency
    workflow.
    """
    if num_nodes < 2:
        raise ValueError("At least two nodes are required")
    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in [0, 1)")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive for binary link-prediction evaluation")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must both be positive")

    positives = canonicalize_edges(positive_edges)
    if positives.size == 0:
        raise ValueError("No valid positive interactions remain after filtering")
    if positives.min() < 0 or positives.max() >= num_nodes:
        raise ValueError("Positive edge endpoint is outside [0, num_nodes)")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positives))
    shuffled = positives[order]

    test_count = int(round(len(shuffled) * test_fraction))
    validation_count = int(round(len(shuffled) * validation_fraction))
    if test_count < 1 or validation_count < 1:
        raise ValueError(
            "Not enough positive interactions for non-empty validation and test splits; "
            "provide more edges or adjust the split fractions"
        )
    if len(shuffled) - test_count - validation_count < 1:
        raise ValueError("Split leaves no positive training edges")

    test_positive = shuffled[:test_count]
    validation_positive = shuffled[test_count : test_count + validation_count]
    train_positive = shuffled[test_count + validation_count :]

    all_positive = {tuple(edge) for edge in positives.tolist()}
    n_train = max(1, int(round(len(train_positive) * negative_ratio)))
    n_validation = max(1, int(round(len(validation_positive) * negative_ratio)))
    n_test = max(1, int(round(len(test_positive) * negative_ratio)))

    train_negative = _sample_negative_edges(num_nodes, n_train, all_positive, rng)
    used_negative = all_positive | {tuple(edge) for edge in train_negative.tolist()}
    validation_negative = _sample_negative_edges(
        num_nodes, n_validation, used_negative, rng
    )
    used_negative |= {tuple(edge) for edge in validation_negative.tolist()}
    test_negative = _sample_negative_edges(num_nodes, n_test, used_negative, rng)

    return EdgeSplit(
        train_positive=train_positive,
        validation_positive=validation_positive,
        test_positive=test_positive,
        train_negative=train_negative,
        validation_negative=validation_negative,
        test_negative=test_negative,
    )
