"""Temporal feature helpers."""
from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Derive clock/calendar proxies from IEEE-CIS relative TransactionDT seconds."""
    out = df.copy() if copy else df
    if "TransactionDT" in out:
        dt = pd.to_numeric(out["TransactionDT"], errors="coerce").fillna(0)
    else:
        dt = pd.Series(0, index=out.index, dtype="float64")
    out["hour"] = ((dt // 3600) % 24).astype("int16")
    out["day"] = (dt // 86400).astype("int32")
    out["week"] = (dt // (86400 * 7)).astype("int16")
    out["day_of_week"] = ((dt // 86400) % 7).astype("int8")
    return out
