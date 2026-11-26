# Drug–Drug Interaction Prediction with Graph Neural Networks

A modular, DGL-first Python framework for learning drug representations and
predicting candidate drug–drug interaction (DDI) links. The project modernizes
the code organization of the original notebook-based work while keeping its
central data representation and task semantics: DrugBank identifiers, SMILES,
167-bit RDKit MACCS fingerprints, a drug interaction graph, and link prediction.

> **Research software only.** Predicted scores are model outputs, not clinically
> validated evidence or medical advice. Do not use them to make treatment or
> prescribing decisions.

## Project overview

The original repository placed most model and data-processing logic in Jupyter
notebooks. This refactor moves reusable code into an importable `drugddi`
package and exposes command-line scripts for training and inference.

### Pipeline

```text
DrugBank CSVs
  ├── drugIDandSMILES.csv: Drug ID, SMILES
  └── interaction_list.csv: Drug1 ID, Drug2 ID
             │
             ▼
     RDKit MACCS fingerprints (167 bits)
             │
             ▼
  Drug nodes + training-only interaction graph
             │
             ▼
  GNN node encoder (GCN / GAT / GraphSAGE / JKNet / VGAE)
             │
             ▼
   Dot-product or symmetric MLP edge decoder
             │
             ▼
     Link-prediction metrics and predictions
```

The main DDI CSV task is an **undirected, binary, transductive link-prediction**
problem. The data preparation code canonicalizes each pair so `(A, B)` and
`(B, A)` are treated as the same interaction. It places only training positive
edges in the message-passing graph, excludes all known positives from negative
sampling, and does not allocate a dense drug-by-drug adjacency matrix.

## Supported architectures

| Architecture | Implementation | Main use |
|---|---|---|
| GCN | `src/drugddi/models/gcn.py` | Binary DDI baseline using DGL `GraphConv` |
| GAT | `src/drugddi/models/gat.py` | Multi-head attention-based graph encoding |
| GraphSAGE | `src/drugddi/models/graphsage.py` | Neighborhood aggregation (`mean`, `gcn`, `pool`, `lstm`) |
| Jumping Knowledge GCN | `src/drugddi/models/jknet.py` | Multi-layer representation aggregation (`cat`, `max`, `lstm`) |
| VGAE | `src/drugddi/models/vgae.py` | Variational graph encoding and inner-product edge decoding |
| R-GCN | `src/drugddi/models/rgcn.py` | Relation-labelled homogeneous graphs |
| Heterogeneous R-GCN | `src/drugddi/models/hetero_rgcn.py` | Typed drug/gene/entity heterographs |

The first five architectures are available in the binary DDI training CLI. The
original DDI CSV contains only interaction pairs, not multiple biological edge
relation labels. Therefore R-GCN is exposed for relation-labelled / heterogeneous
graph workflows; it is deliberately not forced into the binary DDI benchmark
by inventing relation types. See [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md).

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/                 # Reproducible example configurations
├── data/                    # Put original CSV files here (not redistributed)
├── docs/                    # Kaggle, model and upstream reference guides
├── notebooks/               # Notebook notes and Kaggle quickstart
├── scripts/
│   ├── check_environment.py
│   ├── train.py
│   ├── predict.py
│   └── benchmark.py
├── src/drugddi/
│   ├── data/                 # CSV loading, MACCS features, splitting, graph building
│   ├── models/               # GCN, GAT, GraphSAGE, JKNet, VGAE, R-GCN
│   ├── tasks/                # DDI and drug/gene task entry points
│   ├── training/             # Shared trainer and checkpointing
│   ├── evaluation/           # Link-prediction metrics
│   └── inference.py
└── tests/                    # Data, split, metric and model tests
```

## Data requirements

This code bundle does not redistribute the original datasets. Obtain the CSVs
from the [original repository](https://github.com/r-b-1-5/Predicting-Drug-Interactions-using-Graph-Neural-Networks-DGL-)
and place them in `data/` (or supply another directory with `--data-dir`).

| File | Required columns | Meaning |
|---|---|---|
| `drugIDandSMILES.csv` | `Drug ID`, `SMILES` | DrugBank identifier and SMILES string |
| `interaction_list.csv` | `Drug1 ID`, `Drug2 ID` | Known interaction pairs |

Identifiers are normalized to strings. Invalid SMILES are assigned an all-zero
MACCS row by default to match the original notebook's behavior; use the Python
API with `invalid_smiles_policy="raise"` to reject invalid molecules. Interactions
whose IDs are absent from the drug table are filtered and reported.

## Environment and installation

### Compatibility profile

The included compatibility profile pins **DGL 2.4.0 + PyTorch 2.4.1**. As
checked on 26 November 2026, DGL's GitHub release page lists 2.4.0 as the latest
tagged release, and its release notes specifically call out PyTorch 2.4 / CUDA
12.4 support. Although newer PyTorch versions exist, the project uses a matched
DGL 2.4 / Torch 2.4 binary-wheel profile rather than assuming an unverified
newest/newest pairing. Recommended Python for reproducibility: **3.11**.
The `pyproject.toml` accepts Python 3.10–3.12, but the actual Python/platform
wheel combination must be checked against the official DGL install matrix.

The choice is deliberate: choosing the numerically newest PyTorch and DGL
versions independently can produce incompatible binary wheels. Consult the
[DGL install guide](https://www.dgl.ai/pages/start.html) and the
[PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/)
when using a different Python or CUDA runtime.

### GPU installation (CUDA 12.4 example)

Create a clean Python 3.11 environment, then install PyTorch and DGL wheels for
matching versions. The commands below are an example for CUDA 12.4; choose the
appropriate PyTorch index and DGL wheel index for your machine if CUDA differs.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip

# PyTorch 2.4.1 CUDA 12.4 build
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# DGL wheel built for the PyTorch 2.4 line
python -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Remaining project dependencies and editable package install
python -m pip install -r requirements.txt
python -m pip install -e .
```

### CPU installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the environment check:

```bash
python scripts/check_environment.py
```

If DGL fails to import, check Python, PyTorch and CUDA compatibility before
training. Do not rely on `pip` exit code alone to establish binary compatibility.

## Run from Python

After installing the package and placing the CSVs in `data/`, use the reusable
preparation/model/training APIs directly:

```python
import torch

from drugddi.data.prepare import prepare_ddi_data
from drugddi.models.factory import create_model
from drugddi.training.trainer import LinkPredictionTrainer, TrainingConfig, set_seed

set_seed(42)  # Set before model creation for reproducible initialization.
prepared = prepare_ddi_data(
    "data/drugIDandSMILES.csv",
    "data/interaction_list.csv",
    validation_fraction=0.10,
    test_fraction=0.20,
    negative_ratio=1.0,
    seed=42,
)

model = create_model(
    "graphsage",                 # gcn, gat, graphsage, jknet, vgae
    in_features=prepared.features.shape[1],
    hidden_features=64,
    out_features=32,
    dropout=0.2,
    aggregator_type="mean",
)

trainer = LinkPredictionTrainer(
    model=model,
    graph=prepared.graph,
    features=prepared.features,
    config=TrainingConfig(epochs=200, learning_rate=1e-2, patience=30),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    embedding_dim=32,
)
result = trainer.fit(prepared.split)
print(result["test_metrics"])
```

For GAT, pass `num_heads=4`; for JKNet, pass `num_layers=4` and `jk_mode="cat"`;
for GraphSAGE, pass `aggregator_type="mean"`. VGAE uses an inner-product decoder
and variational KL term; keep `TrainingConfig(predictor="dot")` for VGAE.

### Train from the command line

```bash
python scripts/train.py \
  --data-dir data \
  --model graphsage \
  --epochs 200 \
  --hidden-features 64 \
  --out-features 32 \
  --device auto \
  --output-dir artifacts/graphsage
```

Available `--model` values are `gcn`, `gat`, `graphsage`, `jknet` and `vgae`.

Examples:

```bash
python scripts/train.py --data-dir data --model gcn --output-dir artifacts/gcn
python scripts/train.py --data-dir data --model gat --gat-heads 4 --output-dir artifacts/gat
python scripts/train.py --data-dir data --model jknet --jk-layers 4 --jk-mode cat --output-dir artifacts/jknet
python scripts/train.py --data-dir data --model vgae --output-dir artifacts/vgae
```

Each run saves `model.pt`, `train_graph.bin`, `drug_id_map.json`,
`training_result.json` and `run_config.json` in its output directory. No pretrained
checkpoint is included; these commands train from the supplied CSV files.

### Compare the DDI-ready models

The benchmark command runs all five architectures supported by the binary DDI
CSV task with the same seed and split settings, records runtime and test metrics,
and writes `benchmark_results.csv`:

```bash
python scripts/benchmark.py --data-dir data --epochs 200 --device auto \
  --output-dir artifacts/benchmark
```

R-GCN is intentionally omitted from this binary comparison because it requires
meaningful edge-relation labels; use its relational graph API for typed data.

### Predict candidate pairs

Create a CSV with `Drug1 ID` and `Drug2 ID` columns, containing identifiers that
are present in the training graph:

```bash
python scripts/predict.py \
  --checkpoint artifacts/graphsage/model.pt \
  --graph artifacts/graphsage/train_graph.bin \
  --pairs-csv candidate_pairs.csv \
  --output predictions.csv
```

Output columns are `drug_a`, `drug_b`, `interaction_probability`. The checkpoint
is **transductive**: unseen drug IDs are rejected rather than assigned arbitrary
features or node indices. Only load checkpoint files you trust because PyTorch
checkpoint files use pickle serialization.

## Run in Kaggle

For step-by-step Kaggle instructions (uploading files, inspecting the runtime,
installing a compatible DGL/PyTorch pair, training and downloading predictions),
see [`docs/KAGGLE.md`](docs/KAGGLE.md). In brief:

1. Upload this project as a Kaggle Dataset or clone your GitHub copy into `/kaggle/working`.
2. Attach a Dataset containing the two source CSVs.
3. Enable Internet for package installation and select a GPU if needed.
4. Create the isolated `/kaggle/working/ddi-venv` environment and install the matching DGL/PyTorch wheels.
5. Run `scripts/train.py` through that environment and save outputs under `/kaggle/working`.

## Evaluation protocol and caveats

The evaluator reports ROC-AUC, average precision, precision, recall and F1.
Negative pairs are sampled without replacement and excluded from all known
positive pairs. Because the source CSV lists known interactions but does not
provide experimentally confirmed non-interactions, sampled non-edges are
**unobserved pairs treated as benchmark negatives**, not verified safe pairs.
Positive edges are deduplicated as undirected pairs before the split. Validation
and test positive edges are not included in the training message-passing graph.
Splits and model initialization are seeded (call `set_seed` before model creation
when using the Python API).

The current implementation uses full-batch message passing. For larger graphs,
GPU memory may become a limiting factor; neighbor sampling and mini-batch link
prediction are future extensions, not silently implied features of this release.

The main binary DDI pipeline uses the original 167-bit MACCS features. The
molecular graph autoencoder notebooks from the source repository are retained as
upstream research references and are not represented as fully migrated model
pipelines in this package. The existing drug/gene workflow can use the new
heterogeneous R-GCN encoder, but its source-specific data downloads and task
heads must still be configured from the original dataset schema.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Data, splitting and metrics tests run without DGL. DGL-dependent model forward
checks are skipped automatically when DGL is not installed. For a complete
verification, run the suite in the pinned DGL/PyTorch environment.

## Upstream references and attribution

This project is a refactor/reimplementation informed by the original DDI
repository, PanEcho's package organization, and the GNN references listed in
[`docs/UPSTREAMS.md`](docs/UPSTREAMS.md). Consult upstream project licenses and
dataset terms before redistribution or publication.
