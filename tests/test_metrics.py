import torch

from drugddi.evaluation.metrics import compute_link_metrics


def test_link_metrics_are_in_expected_range():
    positive = torch.tensor([3.0, 2.0, 1.0])
    negative = torch.tensor([-2.0, -1.0, -3.0])
    metrics = compute_link_metrics(positive, negative)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
