"""Schema and data-quality validation for training and inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_TRAIN_COLUMNS = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
REQUIRED_INFERENCE_COLUMNS = {"TransactionAmt"}


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    duplicate_transaction_ids: int
    missing_target: int
    missing_rate: float
    fraud_count: int
    fraud_rate: float
    negative_amounts: int
    negative_transaction_dt: int
    infinite_numeric_values: int
    missing_by_column: dict[str, int]
    dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_training_frame(df: pd.DataFrame) -> ValidationReport:
    """Validate training data without silently hiding quality problems."""
    missing = REQUIRED_TRAIN_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required training columns: {sorted(missing)}")
    if len(df) < 10:
        raise ValueError("Training frame is too small.")

    target = pd.to_numeric(df["isFraud"], errors="coerce")
    valid_target = set(target.dropna().unique())
    if not valid_target.issubset({0, 1}):
        raise ValueError(f"isFraud must contain only 0/1 labels; found {sorted(valid_target)}")

    amounts = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce")
    numeric = df.select_dtypes(include="number")
    inf_count = 0
    if not numeric.empty:
        inf_count = int(np.isinf(numeric.to_numpy(dtype="float64", copy=False)).sum())

    return ValidationReport(
        rows=len(df),
        columns=len(df.columns),
        duplicate_transaction_ids=int(df["TransactionID"].duplicated().sum()),
        missing_target=int(target.isna().sum()),
        missing_rate=float(df.isna().mean().mean()),
        fraud_count=int(target.fillna(0).sum()),
        fraud_rate=float(target.mean()),
        negative_amounts=int((amounts < 0).sum()),
        negative_transaction_dt=int((transaction_dt < 0).sum()),
        infinite_numeric_values=inf_count,
        missing_by_column={c: int(v) for c, v in df.isna().sum().items()},
        dtypes={c: str(t) for c, t in df.dtypes.items()},
    )


def assert_training_quality(report: ValidationReport) -> None:
    """Raise on conditions that make the training run invalid rather than merely noteworthy."""
    errors: list[str] = []
    if report.missing_target:
        errors.append(f"target has {report.missing_target} missing values")
    if report.negative_amounts:
        errors.append(f"TransactionAmt has {report.negative_amounts} negative values")
    if report.negative_transaction_dt:
        errors.append(f"TransactionDT has {report.negative_transaction_dt} negative values")
    if report.infinite_numeric_values:
        errors.append(f"numeric data has {report.infinite_numeric_values} infinite values")
    if report.fraud_count == 0 or report.fraud_count == report.rows:
        errors.append("target contains only one class")
    if errors:
        raise ValueError("Training data quality validation failed: " + "; ".join(errors))


def validate_inference_frame(df: pd.DataFrame) -> None:
    """Validate minimal inference schema and values."""
    missing = REQUIRED_INFERENCE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")
    amount = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    if amount.isna().any():
        raise ValueError("TransactionAmt must be numeric and non-missing.")
    if (amount < 0).any():
        raise ValueError("TransactionAmt cannot be negative.")
    if np.isinf(amount.to_numpy(dtype=float)).any():
        raise ValueError("TransactionAmt cannot be infinite.")
