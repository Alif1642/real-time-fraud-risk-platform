"""Categorical feature helpers."""
from __future__ import annotations

import pandas as pd


def safe_string(series: pd.Series) -> pd.Series:
    """Convert categories/objects to a stable string representation while preserving missingness."""
    return series.astype("string")
