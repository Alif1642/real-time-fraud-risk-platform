"""Leakage-safe fit/transform feature engineering for fraud risk."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.temporal_features import add_temporal_features


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learn statistics from the training window only and reuse them on future rows."""

    frequency_columns = ["ProductCD", "card1", "P_emaildomain", "DeviceInfo"]

    def __init__(self) -> None:
        self.frequency_maps_: dict[str, dict[Any, float]] = {}
        self.card_amount_mean_: dict[Any, float] = {}
        self.card_count_: dict[Any, int] = {}
        self.card_last_time_: dict[Any, float] = {}
        self.card_last_device_: dict[Any, Any] = {}
        self.card_last_email_: dict[Any, Any] = {}
        self.amount_median_: float = 0.0
        self.amount_q_: dict[float, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        amount = pd.to_numeric(X.get("TransactionAmt"), errors="coerce")
        self.amount_median_ = float(amount.median()) if amount.notna().any() else 0.0
        self.amount_q_ = {q: float(amount.quantile(q)) for q in (0.25, 0.5, 0.75, 0.95)}

        for col in self.frequency_columns:
            if col in X:
                counts = X[col].astype("string").value_counts(dropna=False, normalize=True)
                self.frequency_maps_[col] = counts.to_dict()

        if "card1" in X:
            tmp = pd.DataFrame({"card1": X["card1"], "amount": amount})
            grouped = tmp.groupby("card1", dropna=False, observed=False)["amount"].agg(["mean", "count"])
            self.card_amount_mean_ = grouped["mean"].to_dict()
            self.card_count_ = grouped["count"].to_dict()

            if "TransactionDT" in X:
                times = pd.to_numeric(X["TransactionDT"], errors="coerce")
                last_time = pd.DataFrame({"card1": X["card1"], "time": times}).groupby(
                    "card1", dropna=False, observed=False
                )["time"].max()
                self.card_last_time_ = last_time.to_dict()

            ordering = (
                X["TransactionDT"].sort_values(kind="stable").index
                if "TransactionDT" in X
                else X.index
            )
            ordered = X.loc[ordering]
            if "DeviceInfo" in ordered:
                self.card_last_device_ = (
                    ordered.dropna(subset=["card1"])
                    .groupby("card1", observed=False)["DeviceInfo"]
                    .last()
                    .to_dict()
                )
            if "P_emaildomain" in ordered:
                self.card_last_email_ = (
                    ordered.dropna(subset=["card1"])
                    .groupby("card1", observed=False)["P_emaildomain"]
                    .last()
                    .to_dict()
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # One intentional copy: features are added without mutating the caller's raw frame.
        out = X.copy()
        out = out.drop(columns=["isFraud", "TransactionID"], errors="ignore")
        out = add_temporal_features(out, copy=False)

        if "TransactionAmt" in out:
            amount = pd.to_numeric(out["TransactionAmt"], errors="coerce")
        else:
            amount = pd.Series(self.amount_median_, index=out.index, dtype="float64")
        amount_filled = amount.fillna(self.amount_median_)
        out["TransactionAmt_log1p"] = np.log1p(amount_filled.clip(lower=0))
        out["TransactionAmt_deviation"] = amount_filled - self.amount_median_
        q25, q50, q75, q95 = (
            self.amount_q_.get(q, self.amount_median_) for q in (0.25, 0.5, 0.75, 0.95)
        )
        out["TransactionAmt_percentile_band"] = np.select(
            [amount_filled <= q25, amount_filled <= q50, amount_filled <= q75, amount_filled <= q95],
            [0, 1, 2, 3],
            default=4,
        ).astype("int8")

        for col, mapping in self.frequency_maps_.items():
            if col in out:
                out[f"{col}_frequency"] = out[col].astype("string").map(mapping).fillna(0.0)

        if "card1" in out:
            out["transaction_count_by_card"] = out["card1"].map(self.card_count_).fillna(0).astype("float32")
            card_avg = out["card1"].map(self.card_amount_mean_).fillna(self.amount_median_)
            out["average_amount_by_card"] = card_avg.astype("float32")
            out["amount_deviation_from_card_average"] = (amount_filled - card_avg).astype("float32")

            if "TransactionDT" in out:
                current_time = pd.to_numeric(out["TransactionDT"], errors="coerce")
                previous = out["card1"].map(self.card_last_time_)
                out["time_since_previous_transaction"] = (
                    current_time - previous
                ).clip(lower=0).fillna(-1).astype("float32")

            if "DeviceInfo" in out:
                previous_device = out["card1"].map(self.card_last_device_)
                out["device_change_indicator"] = (
                    previous_device.notna()
                    & out["DeviceInfo"].notna()
                    & (previous_device.astype("string") != out["DeviceInfo"].astype("string"))
                ).astype("int8")
            else:
                out["device_change_indicator"] = 0

            if "P_emaildomain" in out:
                previous_email = out["card1"].map(self.card_last_email_)
                out["email_change_indicator"] = (
                    previous_email.notna()
                    & out["P_emaildomain"].notna()
                    & (previous_email.astype("string") != out["P_emaildomain"].astype("string"))
                ).astype("int8")
            else:
                out["email_change_indicator"] = 0
        else:
            out["transaction_count_by_card"] = 0.0
            out["average_amount_by_card"] = self.amount_median_
            out["amount_deviation_from_card_average"] = 0.0
            out["time_since_previous_transaction"] = -1.0
            out["device_change_indicator"] = 0
            out["email_change_indicator"] = 0

        out["transaction_frequency"] = out.get("card1_frequency", 0.0)
        return out
