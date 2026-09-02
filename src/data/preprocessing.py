"""Preprocessing pipeline construction."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Build a robust numeric/categorical preprocessing transformer."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )
