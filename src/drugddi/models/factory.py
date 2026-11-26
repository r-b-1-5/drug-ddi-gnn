"""Model factory for the six requested DGL model families."""

from __future__ import annotations

MODEL_NAMES = ("gcn", "gat", "graphsage", "jknet", "vgae", "rgcn")


def create_model(
    model_name: str,
    in_features: int,
    hidden_features: int = 64,
    out_features: int = 32,
    dropout: float = 0.2,
    **model_kwargs,
):
    """Construct one requested encoder without importing unused DGL modules."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if name == "gcn":
        from drugddi.models.gcn import GCNEncoder
        return GCNEncoder(in_features, hidden_features, out_features, dropout)
    if name == "gat":
        from drugddi.models.gat import GATEncoder
        return GATEncoder(
            in_features,
            hidden_features=model_kwargs.get("gat_hidden_features", max(8, hidden_features // 2)),
            out_features=out_features,
            num_heads=model_kwargs.get("num_heads", 4),
            dropout=dropout,
        )
    if name == "graphsage":
        from drugddi.models.graphsage import GraphSAGEEncoder
        return GraphSAGEEncoder(
            in_features,
            hidden_features,
            out_features,
            aggregator_type=model_kwargs.get("aggregator_type", "mean"),
            dropout=dropout,
        )
    if name == "jknet":
        from drugddi.models.jknet import JKNetEncoder
        return JKNetEncoder(
            in_features,
            hidden_features,
            out_features,
            num_layers=model_kwargs.get("num_layers", 4),
            mode=model_kwargs.get("jk_mode", "cat"),
            dropout=dropout,
        )
    if name == "vgae":
        from drugddi.models.vgae import VGAEEncoder
        return VGAEEncoder(in_features, hidden_features, out_features, dropout)
    if name == "rgcn":
        from drugddi.models.rgcn import RGCNEncoder
        num_relations = model_kwargs.get("num_relations")
        if num_relations is None:
            raise ValueError("R-GCN requires num_relations to be specified")
        return RGCNEncoder(
            in_features,
            hidden_features,
            out_features,
            num_relations=num_relations,
            num_bases=model_kwargs.get("num_bases"),
            dropout=dropout,
        )
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
