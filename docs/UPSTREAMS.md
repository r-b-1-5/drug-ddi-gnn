# Source and implementation references

This refactor is a modular reimplementation/modernization. The listed repositories
are architectural references; their code is not copied wholesale. Consult each
upstream repository for its own license and any dataset restrictions.

## Original research project

- [Predicting Drug Interactions using Graph Neural Networks (DGL)](https://github.com/r-b-1-5/Predicting-Drug-Interactions-using-Graph-Neural-Networks-DGL-)
- Main DDI notebook: `predict_new_drug_interactions_using_smiles.ipynb`
- Molecular autoencoder notebooks: `Molecule_to_Graph_and,_Auto_Encoder.ipynb`,
  `Convert_Molecule_to_Graph_+_Auto_Encoder_with_Self_Attention.ipynb`,
  `Graph_Variational_Auto_Encoder_+_Self_Attention_+_Masking.ipynb`
- Drug/gene notebook: `Making_predictions_in_drug_gene_network.ipynb`
- Data preparation notebooks: `smiles-and-drugid-from-sdf.ipynb` and the DrugBank XML
  workflow in the upstream repository.

## Model references supplied for this project

- [DGL examples index](https://github.com/dmlc/dgl/blob/master/examples/README.md)
- [DGL GraphConv](https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/graphconv.py)
- [DGL GATConv documentation](https://www.dgl.ai/dgl_docs/en/2.4.x/generated/dgl.nn.pytorch.conv.GATConv.html)
- [Kipf & Welling GAE/VGAE reference](https://github.com/tkipf/gae)
- [JKNet-DGL](https://github.com/xnuohz/JKNet-dgl)
- [GraphSAGE reference](https://github.com/williamleif/GraphSAGE)
- [DGL heterogeneous R-GCN minibatch example](https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py)

## PanEcho as the organization reference

- [CarDS-Yale/PanEcho](https://github.com/CarDS-Yale/PanEcho)

PanEcho's separation of implementation, project assets and usage documentation
informed the package-oriented layout here.
