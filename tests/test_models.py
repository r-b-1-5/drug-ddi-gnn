import pytest
import torch


def test_homogeneous_dgl_encoders_forward():
    dgl = pytest.importorskip("dgl")
    from drugddi.models.factory import create_model

    graph = dgl.graph(
        (torch.tensor([0, 1, 2, 3, 1, 2]), torch.tensor([1, 2, 3, 0, 0, 1])),
        num_nodes=4,
    )
    graph = dgl.add_self_loop(graph)
    features = torch.randn(4, 10)
    model_options = {
        "gcn": {},
        "gat": {"num_heads": 2},
        "graphsage": {"aggregator_type": "mean"},
        "jknet": {"num_layers": 3, "jk_mode": "cat"},
        "vgae": {},
    }
    for name, options in model_options.items():
        model = create_model(name, 10, hidden_features=8, out_features=6, **options)
        output = model(graph, features)
        embeddings = output[0] if isinstance(output, tuple) else output
        assert embeddings.shape == (4, 6)


def test_homogeneous_rgcn_requires_and_uses_edge_types():
    dgl = pytest.importorskip("dgl")
    from drugddi.models.rgcn import RGCNEncoder

    graph = dgl.graph(
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])), num_nodes=4
    )
    graph = dgl.add_self_loop(graph)
    graph.edata["etype"] = torch.zeros(graph.num_edges(), dtype=torch.int64)
    model = RGCNEncoder(5, 8, 6, num_relations=1)
    output = model(graph, torch.randn(4, 5))
    assert output.shape == (4, 6)


def test_heterogeneous_rgcn_forward():
    dgl = pytest.importorskip("dgl")
    from drugddi.models.hetero_rgcn import HeteroRGCNEncoder, HeteroNodeEmbedding

    graph = dgl.heterograph(
        {
            ("drug", "targets", "gene"): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ("gene", "targeted_by", "drug"): (torch.tensor([1, 2]), torch.tensor([0, 1])),
        },
        num_nodes_dict={"drug": 2, "gene": 3},
    )
    embedding = HeteroNodeEmbedding({"drug": 2, "gene": 3}, embedding_dim=8)
    model = HeteroRGCNEncoder(graph.etypes, 8, hidden_features=8, out_features=6)
    output = model(graph, embedding())
    assert output["drug"].shape == (2, 6)
    assert output["gene"].shape == (3, 6)
