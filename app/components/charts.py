"""Plotly chart helpers."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def probability_histogram(df: pd.DataFrame):
    return px.histogram(df, x="fraud_probability", nbins=20, title="Fraud Probability Distribution")


def risk_distribution(df: pd.DataFrame):
    counts = df["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="count")
    return px.bar(counts, x="risk_level", y="count", title="Risk-Level Distribution")
