from drugddi.models.factory import MODEL_NAMES


def test_model_factory_lists_six_requested_architectures():
    assert set(MODEL_NAMES) == {"gcn", "gat", "graphsage", "jknet", "vgae", "rgcn"}
