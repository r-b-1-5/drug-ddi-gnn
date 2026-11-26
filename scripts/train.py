#!/usr/bin/env python
"""Train one of the five encoders supported by the binary DDI CSV task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DDI_MODELS = ("gcn", "gat", "graphsage", "jknet", "vgae")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing both source CSV files")
    parser.add_argument("--drug-csv", default="drugIDandSMILES.csv")
    parser.add_argument("--interaction-csv", default="interaction_list.csv")
    parser.add_argument("--model", choices=DDI_MODELS, default="graphsage")
    parser.add_argument("--predictor", choices=("dot", "mlp"), default="dot")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/graphsage"))
    parser.add_argument("--jk-mode", choices=("cat", "max", "lstm"), default="cat")
    parser.add_argument("--jk-layers", type=int, default=4)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--sage-aggregator", choices=("mean", "gcn", "pool", "lstm"), default="mean")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import DGL-backed package components after argument parsing so --help works
    # even in environments where the optional graph runtime is not installed.
    from drugddi.data.prepare import prepare_ddi_data
    from drugddi.models.factory import create_model
    from drugddi.models.vgae import VGAEEncoder
    from drugddi.training.checkpoint import save_checkpoint
    from drugddi.training.trainer import LinkPredictionTrainer, TrainingConfig, set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")
    set_seed(args.seed)

    prepared = prepare_ddi_data(
        drug_csv=args.data_dir / args.drug_csv,
        interaction_csv=args.data_dir / args.interaction_csv,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        device=device,
    )
    print(
        "Prepared graph: "
        f"{prepared.graph.num_nodes():,} nodes, {prepared.graph.num_edges():,} message-passing edges"
    )
    print(
        "Positive pairs: "
        f"train={len(prepared.split.train_positive):,}, "
        f"validation={len(prepared.split.validation_positive):,}, "
        f"test={len(prepared.split.test_positive):,}"
    )
    print(
        f"Invalid SMILES with zero features: {len(prepared.invalid_smiles_ids)}; "
        f"interactions filtered for unknown DrugBank IDs: {prepared.filtered_interaction_count:,}"
    )

    model_kwargs = {
        "hidden_features": args.hidden_features,
        "out_features": args.out_features,
        "dropout": args.dropout,
        "num_heads": args.gat_heads,
        "num_layers": args.jk_layers,
        "jk_mode": args.jk_mode,
        "aggregator_type": args.sage_aggregator,
    }
    model = create_model(
        args.model,
        in_features=prepared.features.shape[1],
        hidden_features=args.hidden_features,
        out_features=args.out_features,
        dropout=args.dropout,
        num_heads=args.gat_heads,
        num_layers=args.jk_layers,
        jk_mode=args.jk_mode,
        aggregator_type=args.sage_aggregator,
    )
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        predictor=args.predictor,
        seed=args.seed,
    )
    trainer = LinkPredictionTrainer(
        model=model,
        graph=prepared.graph,
        features=prepared.features,
        config=config,
        device=device,
        embedding_dim=args.out_features,
    )
    result = trainer.fit(prepared.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.output_dir / "train_graph.bin"
    try:
        import dgl
    except ImportError as exc:  # pragma: no cover - requires DGL
        raise ImportError("DGL is required to save the training graph") from exc
    dgl.save_graphs(str(graph_path), [prepared.graph.to("cpu")])
    checkpoint_path = save_checkpoint(
        args.output_dir,
        trainer.model,
        trainer.predictor,
        args.model,
        model_kwargs,
        prepared.drug_ids,
        result,
        predictor_name="inner_product" if isinstance(trainer.model, VGAEEncoder) else args.predictor,
    )
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"device_used": str(device)}, handle, indent=2, default=str)

    print(f"Best validation ROC-AUC: {result['best_validation_roc_auc']:.4f} (epoch {result['best_epoch']})")
    print("Test metrics:")
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training graph: {graph_path}")


if __name__ == "__main__":
    main()
