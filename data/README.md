# Dataset files

Place the two original CSV files in this directory before training:

- `drugIDandSMILES.csv` — `Drug ID`, `SMILES`
- `interaction_list.csv` — `Drug1 ID`, `Drug2 ID`

These source datasets are not redistributed in this code bundle. Obtain them
from the original project repository and review the data provider's terms:
<https://github.com/r-b-1-5/Predicting-Drug-Interactions-using-Graph-Neural-Networks-DGL->.

The loader validates the column names and normalizes identifiers to strings.
