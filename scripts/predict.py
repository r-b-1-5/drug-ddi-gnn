#!/usr/bin/env python
"""Predict interaction probabilities for DrugBank ID pairs in a saved graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with Drug1 ID and Drug2 ID columns")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load DGL-backed inference code after parsing so --help is dependency-light.
    from drugddi.inference import load_trained_model, predict_drug_pairs

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    pairs_frame = pd.read_csv(args.pairs_csv)
    required = {"Drug1 ID", "Drug2 ID"}
    if not required.issubset(pairs_frame.columns):
        raise ValueError(f"Input CSV must contain {sorted(required)}")
    pairs = list(zip(pairs_frame["Drug1 ID"].astype(str), pairs_frame["Drug2 ID"].astype(str)))
    model, predictor, graph, id_to_index, device = load_trained_model(
        args.checkpoint, args.graph, device
    )
    predictions = predict_drug_pairs(model, predictor, graph, id_to_index, pairs, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(args.output, index=False)
    print(f"Saved {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
