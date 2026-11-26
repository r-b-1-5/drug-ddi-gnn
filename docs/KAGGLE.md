# Running in Kaggle

The notebook needs Internet enabled to install packages (or an attached Kaggle
Dataset containing wheels and source files). Select a GPU accelerator if you
want CUDA training. The training code also supports CPU execution.

## 1. Add the files

Upload/extract this project as a Kaggle Dataset, or clone your own published
GitHub copy into `/kaggle/working`. Also attach a Kaggle Dataset containing the
two original CSV files. The input directory must contain:

```text
/kaggle/input/<your-data-dataset>/drugIDandSMILES.csv
/kaggle/input/<your-data-dataset>/interaction_list.csv
```

The project code should be under `/kaggle/working/drugddi-gnn` in the examples
below. Change the path to the directory you actually use.

## 2. Inspect the Kaggle runtime first

Run this before installing a graph framework. Do not assume the preinstalled
PyTorch/CUDA version is compatible with every DGL wheel.

```python
import sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
```

## 3. Install the documented compatibility profile in an isolated environment

This project pins DGL 2.4.0 with PyTorch 2.4.1. DGL's CUDA 12.4 wheel index
provides matching wheels for Python 3.10, 3.11 and 3.12. Create a separate
virtual environment under `/kaggle/working` so the Kaggle notebook's preinstalled
packages are not replaced. The CUDA commands below are for a CUDA 12.4-compatible
Kaggle runtime; use the official DGL/PyTorch selectors if your driver differs.

```python
%cd /kaggle/working/drugddi-gnn
!python -m venv /kaggle/working/ddi-venv

VENV_PY = "/kaggle/working/ddi-venv/bin/python"
!$VENV_PY -m pip install --upgrade pip
!$VENV_PY -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
!$VENV_PY -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
!$VENV_PY -m pip install -r requirements.txt
!$VENV_PY -m pip install -e .
```

For CPU-only use, replace the PyTorch command with
`--index-url https://download.pytorch.org/whl/cpu` and the DGL wheel index with
`https://data.dgl.ai/wheels/torch-2.4/repo.html`.

If the Kaggle Python version is 3.13 or the selected platform has no matching
DGL wheel, stop rather than force-installing incompatible binaries. Use a
supported Python 3.10–3.12 runtime/environment and matching CUDA wheel. This
virtual environment is invoked explicitly in the remaining cells, so the base
Kaggle kernel does not need to be restarted after package installation.

## 4. Verify imports and device

```python
VENV_PY = "/kaggle/working/ddi-venv/bin/python"
!$VENV_PY scripts/check_environment.py
!$VENV_PY -c "import torch, dgl; print('DGL:', dgl.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 5. Train a model

```python
DATA_DIR = "/kaggle/input/<your-data-dataset>"
PROJECT_DIR = "/kaggle/working/drugddi-gnn"

%cd $PROJECT_DIR
!$VENV_PY scripts/train.py \
    --data-dir "$DATA_DIR" \
    --model graphsage \
    --epochs 200 \
    --device auto \
    --output-dir /kaggle/working/artifacts/graphsage
```

Change `--model` to `gcn`, `gat`, `graphsage`, `jknet` or `vgae`. The `rgcn`
encoder is provided for relation-labelled/heterogeneous graph workflows and is
not a valid option for the original untyped binary DDI CSV benchmark.

## 6. Predict candidate pairs

Prepare a CSV with columns `Drug1 ID` and `Drug2 ID`, where both identifiers
exist in the graph used for training. Then run:

```python
!$VENV_PY scripts/predict.py \
    --checkpoint /kaggle/working/artifacts/graphsage/model.pt \
    --graph /kaggle/working/artifacts/graphsage/train_graph.bin \
    --pairs-csv /kaggle/input/<your-pairs-dataset>/candidate_pairs.csv \
    --output /kaggle/working/predictions.csv
```

Download `predictions.csv` from the Kaggle Notebook Output panel. The saved
checkpoint is transductive: unseen DrugBank IDs require a different inference
workflow and are rejected explicitly.

## 7. Kaggle caveats

- Enable Internet to install dependencies or prepare a Dataset with required wheels.
- Use `/kaggle/working` for checkpoints and predictions; `/kaggle/input` is read-only.
- If using a GPU, verify `torch.cuda.is_available()` after installation.
- Full-batch training stores the graph and node activations in memory. Reduce
  hidden dimensions or use CPU if the available GPU RAM is insufficient.
- No pretrained checkpoint is supplied; the commands train from the CSV files.
