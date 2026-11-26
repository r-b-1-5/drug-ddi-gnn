"""DGL graph construction for the original drug-level interaction network."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def build_dgl_graph(
    num_nodes: int,
    train_edges: np.ndarray,
    node_features: np.ndarray,
    device: Optional[torch.device] = None,
):
    """Build a bidirectional DGL graph from training positives only.

    Self-loops are added exactly once for stable message passing. Validation and
    test positives are intentionally not included in this graph.
    """
    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - requires DGL install
        raise ImportError(
            "DGL is required to construct a training graph. Follow README's DGL setup."
        ) from exc

    edge_array = np.asarray(train_edges, dtype=np.int64).reshape(-1, 2)
    if node_features.shape[0] != num_nodes:
        raise ValueError("node_features row count must equal num_nodes")
    if edge_array.size:
        source = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        target = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    else:
        source = np.empty(0, dtype=np.int64)
        target = np.empty(0, dtype=np.int64)

    graph = dgl.graph(
        (torch.as_tensor(source, dtype=torch.int64), torch.as_tensor(target, dtype=torch.int64)),
        num_nodes=num_nodes,
    )
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph.ndata["feat"] = torch.as_tensor(node_features, dtype=torch.float32)
    if device is not None:
        graph = graph.to(device)
    return graph
