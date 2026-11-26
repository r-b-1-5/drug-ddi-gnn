"""Input readers for the two CSV files used by the original DDI notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_drug_table(path: PathLike) -> pd.DataFrame:
    """Load DrugBank identifiers and SMILES strings.

    Expected columns are ``Drug ID`` and ``SMILES`` (the original dataset
    schema). Column whitespace is stripped, IDs are normalized to strings,
    and invalid/missing rows are reported by validation rather than silently
    changing the source table.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Drug ID", "SMILES"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Drug table is missing required columns {sorted(missing)}; "
            f"found {list(frame.columns)}"
        )

    frame = frame[["Drug ID", "SMILES"]].copy()
    frame["Drug ID"] = frame["Drug ID"].astype("string").str.strip()
    frame["SMILES"] = frame["SMILES"].astype("string").str.strip()
    frame = frame.dropna(subset=["Drug ID", "SMILES"])
    frame = frame[(frame["Drug ID"] != "") & (frame["SMILES"] != "")]
    if frame["Drug ID"].duplicated().any():
        duplicates = frame.loc[frame["Drug ID"].duplicated(), "Drug ID"].head(5).tolist()
        raise ValueError(f"Duplicate Drug ID values found, e.g. {duplicates}")
    return frame.reset_index(drop=True)


def load_interaction_table(path: PathLike) -> pd.DataFrame:
    """Load pairwise DDI labels using the original ``Drug1 ID``/``Drug2 ID`` schema."""
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Drug1 ID", "Drug2 ID"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Interaction table is missing required columns {sorted(missing)}; "
            f"found {list(frame.columns)}"
        )
    frame = frame[["Drug1 ID", "Drug2 ID"]].copy()
    for column in ("Drug1 ID", "Drug2 ID"):
        frame[column] = frame[column].astype("string").str.strip()
    frame = frame.dropna(subset=["Drug1 ID", "Drug2 ID"])
    frame = frame[(frame["Drug1 ID"] != "") & (frame["Drug2 ID"] != "")]
    return frame.reset_index(drop=True)
