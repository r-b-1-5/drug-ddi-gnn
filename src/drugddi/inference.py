"""Inference helpers for drug pairs represented in a trained checkpoint graph."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from drugddi.models.factory import create_model
from drugddi.models.predictors import DotProductPredictor, MLPEdgePredictor
from drugddi.models.vgae import VGAEEncoder


def load_trained_model(checkpoint_path: str | Path, graph_path: str | Path, device=None):
    """Load a saved DDI encoder, edge decoder and its training graph."""
    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - requires DGL
        raise ImportError("DGL is required for checkpoint inference") from exc

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PyTorch checkpoints use pickle serialization: load only checkpoints you trust.
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    graph_list, _ = dgl.load_graphs(str(graph_path))
    if not graph_list:
        raise ValueError(f"No graph was found in {graph_path}")
    graph = graph_list[0].to(device)
    feature_dim = int(graph.ndata["feat"].shape[1])
    model_kwargs = payload.get("model_kwargs", {})
    hidden_dim = int(model_kwargs.get("hidden_features", 64))
    out_dim = int(model_kwargs.get("out_features", 32))
    model = create_model(
        payload["model_name"],
        feature_dim,
        hidden_features=hidden_dim,
        out_features=out_dim,
        dropout=float(model_kwargs.get("dropout", 0.2)),
        **{k: v for k, v in model_kwargs.items() if k not in {"hidden_features", "out_features", "dropout"}},
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    if payload["model_name"] == "vgae":
        predictor = None
    elif payload.get("predictor_name", "dot") == "mlp":
        predictor = MLPEdgePredictor(out_dim).to(device)
        predictor.load_state_dict(payload["predictor_state_dict"])
    else:
        predictor = DotProductPredictor().to(device)
        if payload.get("predictor_state_dict"):
            predictor.load_state_dict(payload["predictor_state_dict"])
    model.eval()
    if predictor is not None:
        predictor.eval()
    id_to_index = {str(drug_id): index for index, drug_id in enumerate(payload["drug_ids"])}
    return model, predictor, graph, id_to_index, device


def predict_drug_pairs(
    model,
    predictor,
    graph,
    id_to_index,
    pairs: Iterable[Tuple[str, str]],
    device,
) -> List[dict]:
    """Score pairs whose DrugBank IDs occur in the fitted graph."""
    encoded_pairs = []
    normalized_pairs = []
    for drug_a, drug_b in pairs:
        a, b = str(drug_a), str(drug_b)
        if a not in id_to_index or b not in id_to_index:
            raise KeyError(
                f"Unknown DrugBank ID in pair ({a}, {b}). This checkpoint is transductive; "
                "unseen drugs need a separately constructed molecular/interaction graph."
            )
        encoded_pairs.append((id_to_index[a], id_to_index[b]))
        normalized_pairs.append((a, b))
    if not encoded_pairs:
        raise ValueError("At least one candidate drug pair is required")
    edge_tensor = torch.as_tensor(encoded_pairs, dtype=torch.long, device=device).T
    with torch.no_grad():
        features = graph.ndata["feat"]
        encoded = model(graph, features)
        if isinstance(model, VGAEEncoder):
            embeddings = encoded[0]
            logits = model.decode(embeddings, edge_tensor)
        else:
            embeddings = encoded
            logits = predictor(embeddings, edge_tensor)
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
    return [
        {"drug_a": a, "drug_b": b, "interaction_probability": float(probability)}
        for (a, b), probability in zip(normalized_pairs, probabilities)
    ]
