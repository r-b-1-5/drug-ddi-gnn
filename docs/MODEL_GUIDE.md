# Model guide

| Model | Module | Intended use in this repository |
|---|---|---|
| GCN | `drugddi.models.gcn.GCNEncoder` | Binary DDI link-prediction baseline |
| GAT | `drugddi.models.gat.GATEncoder` | Attention-based DDI link prediction |
| GraphSAGE | `drugddi.models.graphsage.GraphSAGEEncoder` | DDI link prediction with mean/gcn/pool/lstm aggregation |
| JKNet | `drugddi.models.jknet.JKNetEncoder` | Multi-depth GCN representations with `cat`, `max` or `lstm` aggregation |
| VGAE | `drugddi.models.vgae.VGAEEncoder` | Variational graph autoencoding and inner-product link decoding |
| R-GCN | `drugddi.models.rgcn.RGCNEncoder` | Homogeneous graphs carrying integer `graph.edata['etype']` relation IDs |
| Heterogeneous R-GCN | `drugddi.models.hetero_rgcn.HeteroRGCNEncoder` | Heterographs with separate node types and semantically typed relations |

## Shared DDI benchmark

The CLI supports GCN, GAT, GraphSAGE, JKNet and VGAE against the original binary
interaction CSV. Each encoder produces node embeddings; a dot-product or MLP
edge decoder scores candidate drug pairs. VGAE uses its own inner-product decoder
and KL regularization.

R-GCN is not included in the binary DDI CLI because the source CSV has no typed
edge labels. Do not convert positive/negative labels into relation types: those
are supervision labels, not known biological relation categories. Use `RGCNEncoder`
for a homogeneous relation-labelled graph, or `DrugGeneRGCN` /
`HeteroRGCNEncoder` for a heterogeneous drug/gene graph whose edge types are
supported by the source data.

## Heterogeneous drug/gene example

```python
import torch
import dgl
from drugddi.tasks.drug_gene import DrugGeneRGCN

# Construct/load a DGL heterograph with real relation semantics.
# Example canonical edge types might be ('drug', 'targets', 'gene') and
# ('gene', 'associated_with', 'disease'); use the relations in your source data.
heterograph = dgl.heterograph({
    ('drug', 'targets', 'gene'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('gene', 'targeted_by', 'drug'): (torch.tensor([1, 2]), torch.tensor([0, 1])),
}, num_nodes_dict={'drug': 2, 'gene': 3})

model = DrugGeneRGCN(heterograph, embedding_dim=64, out_features=32)
node_embeddings = model(heterograph)
print({node_type: values.shape for node_type, values in node_embeddings.items()})
```

The sample only demonstrates model invocation with synthetic topology. It is not
an experiment or a performance claim. For a real task, use the actual node IDs,
feature matrices (if available), relation names and a task-specific output head.

## Important evaluation detail

The DDI split is built on canonical undirected pairs. Validation/test positive
edges are excluded from the message-passing graph, and negative sampling excludes
all known positive pairs. This reduces reverse-edge leakage and avoids allocating
a dense `num_drugs × num_drugs` adjacency matrix. The task remains transductive:
new drug IDs are not supported by a saved checkpoint without rebuilding and
retraining/augmenting the graph.
