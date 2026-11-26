import numpy as np
import pytest

from drugddi.data.splitting import canonicalize_edges, make_link_prediction_split


def _as_set(edges):
    return {tuple(row) for row in np.asarray(edges).reshape(-1, 2).tolist()}


def test_canonicalize_edges_deduplicates_reversed_pairs_and_self_loops():
    edges = canonicalize_edges([(0, 1), (1, 0), (2, 2), (4, 3)])
    assert edges.tolist() == [[0, 1], [3, 4]]


def test_split_has_disjoint_positive_and_negative_sets():
    positives = [(i, (i + 1) % 40) for i in range(40)]
    # Add reversed duplicates to ensure they do not become a second example.
    positives += [(b, a) for a, b in positives]
    split = make_link_prediction_split(
        num_nodes=40,
        positive_edges=positives,
        validation_fraction=0.15,
        test_fraction=0.20,
        negative_ratio=1.0,
        seed=7,
    )

    train_pos = _as_set(split.train_positive)
    val_pos = _as_set(split.validation_positive)
    test_pos = _as_set(split.test_positive)
    train_neg = _as_set(split.train_negative)
    val_neg = _as_set(split.validation_negative)
    test_neg = _as_set(split.test_negative)
    all_pos = train_pos | val_pos | test_pos

    assert train_pos.isdisjoint(val_pos | test_pos)
    assert val_pos.isdisjoint(test_pos)
    assert not (all_pos & (train_neg | val_neg | test_neg))
    assert train_neg.isdisjoint(val_neg | test_neg)
    assert val_neg.isdisjoint(test_neg)
    assert len(train_neg) == len(split.train_positive)
    assert len(val_neg) == len(split.validation_positive)
    assert len(test_neg) == len(split.test_positive)
    assert all(a < b for a, b in train_pos | val_pos | test_pos | train_neg | val_neg | test_neg)


def test_split_rejects_empty_positive_edges():
    with pytest.raises(ValueError, match="No valid positive"):
        make_link_prediction_split(5, [])


def test_split_is_reproducible():
    edges = [(i, i + 1) for i in range(1, 30)]
    first = make_link_prediction_split(40, edges, seed=10)
    second = make_link_prediction_split(40, edges, seed=10)
    np.testing.assert_array_equal(first.train_positive, second.train_positive)
    np.testing.assert_array_equal(first.train_negative, second.train_negative)
