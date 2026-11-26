from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drugddi.data.io import load_drug_table, load_interaction_table
from drugddi.data.molecular_features import MACCS_FEATURE_DIM, smiles_to_maccs


def test_loaders_use_original_csv_schema(tmp_path: Path):
    drug_path = tmp_path / "drugs.csv"
    interaction_path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {" Drug ID ": [" DB00001 ", "DB00002"], " SMILES ": ["CCO", "CC"]}
    ).to_csv(drug_path, index=False)
    pd.DataFrame(
        {" Drug1 ID ": ["DB00001"], " Drug2 ID ": ["DB00002"]}
    ).to_csv(interaction_path, index=False)

    drugs = load_drug_table(drug_path)
    interactions = load_interaction_table(interaction_path)

    assert drugs["Drug ID"].tolist() == ["DB00001", "DB00002"]
    assert drugs["SMILES"].tolist() == ["CCO", "CC"]
    assert interactions.iloc[0].tolist() == ["DB00001", "DB00002"]


def test_load_drug_table_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "duplicates.csv"
    pd.DataFrame(
        {"Drug ID": ["DB00001", "DB00001"], "SMILES": ["CCO", "CC"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate Drug ID"):
        load_drug_table(path)


def test_maccs_fingerprint_has_original_width():
    pytest.importorskip("rdkit")
    fingerprint = smiles_to_maccs("CCO")
    assert fingerprint.shape == (MACCS_FEATURE_DIM,)
    assert fingerprint.dtype == np.float32
    assert set(np.unique(fingerprint)).issubset({0.0, 1.0})
