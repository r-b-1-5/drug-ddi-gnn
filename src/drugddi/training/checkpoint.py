"""Checkpoint and graph persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(
    output_dir: str | Path,
    model,
    predictor,
    model_name: str,
    model_kwargs: Dict,
    drug_ids,
    training_result: Dict,
    predictor_name: str = "dot",
) -> Path:
    """Save model weights and an ID map; DGL graph is saved separately by caller."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "predictor_name": predictor_name,
        "drug_ids": list(map(str, drug_ids)),
        "training_result": training_result,
    }
    checkpoint_path = output / "model.pt"
    torch.save(payload, checkpoint_path)
    with (output / "drug_id_map.json").open("w", encoding="utf-8") as handle:
        json.dump({str(drug_id): index for index, drug_id in enumerate(drug_ids)}, handle, indent=2)
    with (output / "training_result.json").open("w", encoding="utf-8") as handle:
        json.dump(training_result, handle, indent=2)
    return checkpoint_path
