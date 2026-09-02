"""Metric-card helpers."""
from __future__ import annotations

import streamlit as st


def metric_row(values: dict[str, object]) -> None:
    """Render compact metric cards."""
    cols = st.columns(len(values))
    for col, (label, value) in zip(cols, values.items()):
        col.metric(label, value)
