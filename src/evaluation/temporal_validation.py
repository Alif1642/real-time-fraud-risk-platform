"""Chronological train/validation/test splitting."""
from __future__ import annotations

import pandas as pd


def _ordered(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df:
        raise ValueError(f"Missing temporal split column: {time_col}")
    if df[time_col].is_monotonic_increasing:
        return df
    return df.sort_values(time_col, kind="stable")


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    time_col: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split oldest→newest; avoid a redundant full sort/copy when data is already ordered."""
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    ordered = _ordered(df, time_col)
    n = len(ordered)
    i = int(n * train_fraction)
    j = i + int(n * validation_fraction)
    # Shallow copies isolate column operations such as pop() while sharing underlying blocks.
    return (
        ordered.iloc[:i].copy(deep=False),
        ordered.iloc[i:j].copy(deep=False),
        ordered.iloc[j:].copy(deep=False),
    )


def split_validation_for_calibration(
    validation: pd.DataFrame,
    calibration_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split validation into calibration then threshold-selection windows."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    ordered = _ordered(validation, "TransactionDT")
    i = max(1, int(len(ordered) * calibration_fraction))
    return ordered.iloc[:i].copy(deep=False), ordered.iloc[i:].copy(deep=False)
