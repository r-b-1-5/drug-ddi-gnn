#!/usr/bin/env python
"""Run a repeatable model comparison on the binary DDI CSV task."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


BENCHMARK_MODELS = ("gcn", "gat", "graphsage", "jknet", "vgae")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=BENCHMARK_MODELS, default=list(BENCHMARK_MODELS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--predictor", choices=("dot", "mlp"), default="dot")
    return parser.parse_args()


def main():
    args = parse_args()
    if "vgae" in args.models and args.predictor != "dot":
        raise SystemExit("VGAE uses its built-in inner-product decoder; use --predictor dot.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    train_script = Path(__file__).with_name("train.py")

    for model_name in args.models:
        model_dir = args.output_dir / model_name
        command = [
            sys.executable,
            str(train_script),
            "--data-dir",
            str(args.data_dir),
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--predictor",
            args.predictor,
            "--output-dir",
            str(model_dir),
        ]
        print(f"\n=== Benchmarking {model_name} ===", flush=True)
        start = time.perf_counter()
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - start

        result_path = model_dir / "training_result.json"
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        row = {
            "model": model_name,
            "best_epoch": result["best_epoch"],
            "best_validation_roc_auc": result["best_validation_roc_auc"],
            "runtime_seconds": round(elapsed, 3),
            **result["test_metrics"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    output_csv = args.output_dir / "benchmark_results.csv"
    columns = [
        "model",
        "best_epoch",
        "best_validation_roc_auc",
        "runtime_seconds",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved benchmark comparison to {output_csv}")


if __name__ == "__main__":
    main()
