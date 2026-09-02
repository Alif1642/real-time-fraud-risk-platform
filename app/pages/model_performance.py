"""Streamlit page for verified model comparison, holdout metrics, and explainability figures."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import settings


def render() -> None:
    st.header("Model Performance")
    metrics_dir = settings.report_dir / "metrics"
    figures_dir = settings.report_dir / "figures"
    comparison = metrics_dir / "model_comparison.csv"
    selected = metrics_dir / "selected_test_metrics.json"
    metadata = metrics_dir / "verified_training_metadata.json"

    if metadata.exists():
        meta = json.loads(metadata.read_text())
        cols = st.columns(4)
        cols[0].metric("Selected model", meta.get("model_name", "unknown"))
        cols[1].metric("Calibration", meta.get("calibration_method", "unknown"))
        cols[2].metric("Block threshold", f"{float(meta.get('threshold', 0.0)):.3f}")
        cols[3].metric("Result type", "Local holdout")
        st.caption(
            "Model selection is validation PR-AUC only. Held-out metrics are not used to choose the winner."
        )

    if not comparison.exists():
        st.info("Metrics are generated after running the training pipeline.")
        return

    st.subheader("Validation model comparison")
    st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)

    if selected.exists():
        metrics = json.loads(selected.read_text())
        st.subheader("Selected model — LOCAL IEEE-CIS HOLDOUT RESULTS")
        st.json(metrics)
        st.caption("These are local temporal holdout results, not Kaggle leaderboard scores.")

    holdout_figures = [
        figures_dir / "lightgbm_test_pr_curve.png",
        figures_dir / "lightgbm_test_confusion_matrix.png",
        figures_dir / "lightgbm_test_calibration.png",
    ]
    available = [path for path in holdout_figures if path.exists()]
    if available:
        st.subheader("Held-out evaluation")
        for path in available:
            st.image(str(path), caption=path.name)

    shap_figures = [
        figures_dir / "shap_summary.png",
        figures_dir / "shap_feature_importance.png",
        figures_dir / "shap_individual_transaction.png",
    ]
    available_shap = [path for path in shap_figures if path.exists()]
    if available_shap:
        st.subheader("SHAP explainability")
        for path in available_shap:
            st.image(str(path), caption=path.name)
