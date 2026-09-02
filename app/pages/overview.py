from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.metrics import metric_row
from src.config import settings
from src.models.predict import load_artifacts


def _load_verified_metrics() -> dict:
    path = settings.report_dir / "metrics" / "selected_test_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render() -> None:
    st.header("Overview")
    metrics = _load_verified_metrics()
    try:
        bundle = load_artifacts()
        st.success(f"Loaded model: {bundle.metadata.get('model_name', 'unknown')} | threshold {bundle.threshold:.3f}")
    except Exception as exc:
        st.info("No local model artifacts are currently available. Place the IEEE-CIS CSVs in data/raw/ and run `python -m src.models.train`.")
        st.caption(str(exc))

    if metrics:
        metric_row({
            "PR-AUC": f"{metrics.get('pr_auc', float('nan')):.4f}",
            "ROC-AUC": f"{metrics.get('roc_auc', float('nan')):.4f}",
            "Precision": f"{metrics.get('precision', float('nan')):.4f}",
            "Recall": f"{metrics.get('recall', float('nan')):.4f}",
            "F1": f"{metrics.get('f1', float('nan')):.4f}",
            "Brier": f"{metrics.get('brier_score', float('nan')):.4f}",
        })
        st.caption("LOCAL IEEE-CIS HOLDOUT RESULTS — not Kaggle leaderboard scores.")
    else:
        st.warning("No local holdout metrics found. Run the training pipeline first.")

    for name, label in [
        ("lightgbm_test_pr_curve.png", "Precision–Recall curve"),
        ("lightgbm_test_calibration.png", "Calibration curve"),
        ("shap_feature_importance.png", "SHAP feature importance"),
    ]:
        path = settings.report_dir / "figures" / name
        if path.exists():
            st.image(str(path), caption=label, use_container_width=True)
