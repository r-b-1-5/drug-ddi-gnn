#!/usr/bin/env python
"""Report the runtime and fail if required project dependencies are unavailable."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero exit code unless CUDA is available",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    errors = []
    try:
        import torch

        print(
            f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}; "
            f"CUDA available: {torch.cuda.is_available()}"
        )
        if args.require_cuda and not torch.cuda.is_available():
            errors.append("CUDA was required but is unavailable")
    except Exception as exc:
        errors.append(f"PyTorch import failed: {exc}")
        torch = None

    try:
        import dgl

        print(f"DGL: {dgl.__version__}")
    except Exception as exc:
        errors.append(f"DGL import failed: {exc}")

    try:
        import rdkit

        print(f"RDKit: {rdkit.__version__}")
    except Exception as exc:
        errors.append(f"RDKit import failed: {exc}")

    try:
        import pandas
        import sklearn
        import numpy

        print(
            f"NumPy: {numpy.__version__}; pandas: {pandas.__version__}; "
            f"scikit-learn: {sklearn.__version__}"
        )
    except Exception as exc:
        errors.append(f"Scientific Python stack import failed: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
