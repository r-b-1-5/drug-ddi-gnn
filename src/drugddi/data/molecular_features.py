"""RDKit feature generation matching the original notebook's MACCS encoding."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

MACCS_FEATURE_DIM = 167


def smiles_to_maccs(smiles: str) -> np.ndarray:
    """Convert one SMILES string to the original 167-bit MACCS fingerprint.

    RDKit is imported lazily so that data utilities which do not need chemical
    parsing remain importable in lightweight environments.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "RDKit is required for SMILES featurization. Install the 'chemistry' extra."
        ) from exc

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    fingerprint = MACCSkeys.GenMACCSKeys(molecule)
    features = np.asarray(list(fingerprint), dtype=np.float32)
    if features.shape != (MACCS_FEATURE_DIM,):
        raise RuntimeError(
            f"Expected {MACCS_FEATURE_DIM} MACCS bits, got shape {features.shape}"
        )
    return features


def featurize_smiles(
    drug_ids: Iterable[str],
    smiles_values: Iterable[str],
    invalid_policy: str = "zeros",
) -> Tuple[np.ndarray, List[str]]:
    """Create a feature matrix and return IDs whose SMILES could not be parsed.

    ``invalid_policy='zeros'`` deliberately matches the original notebook,
    which initialized a zero matrix and left invalid molecules as zero rows.
    ``invalid_policy='raise'`` provides stricter validation for new studies.
    """
    if invalid_policy not in {"zeros", "raise"}:
        raise ValueError("invalid_policy must be either 'zeros' or 'raise'")

    ids = list(drug_ids)
    smiles = list(smiles_values)
    if len(ids) != len(smiles):
        raise ValueError("drug_ids and smiles_values must have the same length")

    matrix = np.zeros((len(ids), MACCS_FEATURE_DIM), dtype=np.float32)
    invalid: List[str] = []
    for index, (drug_id, value) in enumerate(zip(ids, smiles)):
        try:
            matrix[index] = smiles_to_maccs(str(value))
        except (ValueError, TypeError):
            invalid.append(str(drug_id))
            if invalid_policy == "raise":
                raise ValueError(f"Could not featurize drug {drug_id!r}")
    return matrix, invalid
