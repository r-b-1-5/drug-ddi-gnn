"""End-to-end preparation of DDI data from the two original CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from drugddi.data.graph import build_dgl_graph
from drugddi.data.io import load_drug_table, load_interaction_table
from drugddi.data.molecular_features import featurize_smiles
from drugddi.data.splitting import EdgeSplit, make_link_prediction_split


@dataclass
class PreparedDDI:
    graph: object
    features: torch.Tensor
    drug_ids: List[str]
    id_to_index: Dict[str, int]
    split: EdgeSplit
    invalid_smiles_ids: List[str]
    filtered_interaction_count: int


def prepare_ddi_data(
    drug_csv: str | Path,
    interaction_csv: str | Path,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    negative_ratio: float = 1.0,
    seed: int = 42,
    invalid_smiles_policy: str = "zeros",
    device: Optional[torch.device] = None,
) -> PreparedDDI:
    """Load source data, build 167-bit MACCS features, split links and create graph."""
    drugs = load_drug_table(drug_csv)
    interactions = load_interaction_table(interaction_csv)

    drug_ids = drugs["Drug ID"].astype(str).tolist()
    id_to_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    features, invalid_ids = featurize_smiles(
        drug_ids, drugs["SMILES"].astype(str).tolist(), invalid_policy=invalid_smiles_policy
    )

    first = interactions["Drug1 ID"].astype(str).map(id_to_index)
    second = interactions["Drug2 ID"].astype(str).map(id_to_index)
    valid = first.notna() & second.notna()
    filtered_interaction_count = int((~valid).sum())
    edge_pairs = np.column_stack(
        [first[valid].to_numpy(dtype=np.int64), second[valid].to_numpy(dtype=np.int64)]
    )

    split = make_link_prediction_split(
        num_nodes=len(drug_ids),
        positive_edges=edge_pairs,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    graph = build_dgl_graph(len(drug_ids), split.train_positive, features, device=device)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return PreparedDDI(
        graph=graph,
        features=feature_tensor,
        drug_ids=drug_ids,
        id_to_index=id_to_index,
        split=split,
        invalid_smiles_ids=invalid_ids,
        filtered_interaction_count=filtered_interaction_count,
    )
