"""Streamlit monitoring page using core PSI/KS utilities without synthetic data."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import settings
from src.monitoring.drift import drift_status, numeric_drift


def render() -> None:
    st.header("Monitoring")
    st.write("Upload two real transaction windows to compare numeric drift. No sample or synthetic dataset is bundled with this repository.")
    reference_file = st.file_uploader("Reference CSV", type=["csv"], key="reference_csv")
    current_file = st.file_uploader("Current CSV", type=["csv"], key="current_csv")
    if reference_file is None or current_file is None:
        st.info("Provide both CSV windows to calculate PSI and KS statistics.")
        return

    reference = pd.read_csv(reference_file)
    current = pd.read_csv(current_file)
    numeric = [c for c in reference.select_dtypes(include="number").columns if c in current.columns]
    if not numeric:
        st.warning("No shared numeric columns were found.")
        return
    selected = st.multiselect("Numeric features", numeric, default=numeric[: min(5, len(numeric))])
    if not selected:
        return
    drift = numeric_drift(reference, current, selected)
    if not drift.empty:
        drift["status"] = drift["psi"].map(lambda x: drift_status(x, settings.psi_warning, settings.psi_critical))
    st.dataframe(drift, use_container_width=True, hide_index=True)
    st.caption("PSI thresholds are configurable project thresholds, not universal standards.")
