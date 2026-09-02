# IEEE-CIS data setup

The raw IEEE-CIS Fraud Detection dataset is intentionally **not included** in this repository.

Place the official CSV files here before training:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The labeled training target is `isFraud` in `train_transaction.csv`. The official competition test transaction file is unlabeled; the project never creates a fake target for it.

`data/processed/` is reserved for optional local derived data and is gitignored. Do not commit the raw IEEE-CIS files or generated data products.
