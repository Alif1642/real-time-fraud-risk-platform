"""SHAP explanations mapped to analyst-friendly reason codes.

The helper functions deliberately densify only a *small explanation sample*.  The
training/inference pipeline remains sparse and memory efficient.  SHAP has changed
its LightGBM binary-classification return type across releases, so values are
normalised here instead of relying on one version-specific shape.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REASON_MAP = {
    "TransactionAmt": "unusual_transaction_amount",
    "TransactionAmt_log1p": "unusual_transaction_amount",
    "TransactionAmt_deviation": "unusual_transaction_amount",
    "card1": "unusual_card_activity",
    "card1_frequency": "unusual_card_activity",
    "transaction_count_by_card": "unusual_card_activity",
    "amount_deviation_from_card_average": "unusual_card_activity",
    "DeviceInfo": "high_risk_device",
    "DeviceType": "high_risk_device",
    "DeviceInfo_frequency": "high_risk_device",
    "hour": "unusual_transaction_timing",
    "day_of_week": "unusual_transaction_timing",
    "P_emaildomain": "unusual_email_pattern",
    "P_emaildomain_frequency": "unusual_email_pattern",
}

_TREE_MODEL_NAMES = {"LGBMClassifier", "XGBClassifier"}


def _reason_for_feature(name: str) -> str:
    for key, reason in REASON_MAP.items():
        if key in name:
            return reason
    return "unusual_feature_pattern"


def _dense_explanation_matrix(matrix: Any) -> np.ndarray:
    """Return a dense 2-D matrix for a bounded SHAP explanation sample only."""
    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    dense = np.asarray(dense)
    if dense.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix for SHAP, got shape {dense.shape}")
    return dense


def _normalise_binary_shap_values(
    values: Any,
    *,
    n_rows: int,
    n_features: int,
) -> np.ndarray:
    """Normalise SHAP outputs to ``(rows, features)`` for fraud class 1.

    Supported forms include:
    - list/tuple of per-class arrays (fraud class = last/class 1)
    - scipy sparse matrices
    - ndarray ``(rows, features)``
    - ndarray ``(rows, features, classes)``
    - defensive handling of ``(classes, rows, features)`` and
      ``(rows, classes, features)`` forms seen across explainer APIs.
    """
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("SHAP returned an empty class list")
        values = values[1] if len(values) > 1 else values[0]

    if hasattr(values, "toarray"):
        values = values.toarray()

    arr = np.asarray(values)

    if arr.ndim == 2:
        if arr.shape != (n_rows, n_features):
            raise ValueError(
                f"Unexpected 2-D SHAP shape {arr.shape}; expected {(n_rows, n_features)}"
            )
        return arr.astype(float, copy=False)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported SHAP output shape {arr.shape}")

    # Most recent Explanation-style form: rows × features × classes.
    if arr.shape[0] == n_rows and arr.shape[1] == n_features:
        class_index = 1 if arr.shape[2] > 1 else 0
        return arr[:, :, class_index].astype(float, copy=False)

    # Some wrappers expose classes × rows × features.
    if arr.shape[1] == n_rows and arr.shape[2] == n_features:
        class_index = 1 if arr.shape[0] > 1 else 0
        return arr[class_index, :, :].astype(float, copy=False)

    # Defensive support for rows × classes × features.
    if arr.shape[0] == n_rows and arr.shape[2] == n_features:
        class_index = 1 if arr.shape[1] > 1 else 0
        return arr[:, class_index, :].astype(float, copy=False)

    raise ValueError(
        "Could not align SHAP output with feature matrix: "
        f"SHAP={arr.shape}, expected rows={n_rows}, features={n_features}"
    )


def _tree_shap_inputs(pipeline: Any, raw_frame: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray]:
    """Transform raw rows and return model, dense bounded matrix, and feature names."""
    model = pipeline.named_steps["model"]
    if model.__class__.__name__ not in _TREE_MODEL_NAMES:
        raise TypeError("TreeSHAP requires the selected LightGBM/XGBoost model")

    feat = pipeline.named_steps["features"].transform(raw_frame)
    prep = pipeline.named_steps["preprocessor"]
    matrix = _dense_explanation_matrix(prep.transform(feat))
    names = np.asarray(prep.get_feature_names_out(), dtype=str)
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Transformed feature/name mismatch: matrix={matrix.shape[1]}, names={len(names)}"
        )
    return model, matrix, names


def _tree_shap_values(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Calculate and normalise fraud-class TreeSHAP values."""
    import shap

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(matrix)
    return _normalise_binary_shap_values(
        raw_values,
        n_rows=matrix.shape[0],
        n_features=matrix.shape[1],
    )


def explain_rows(pipeline: Any, raw_frame: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
    """Return top analyst-friendly reason codes for each scored row.

    SHAP is attempted for supported tree models.  A deterministic non-SHAP fallback
    keeps the API available if the optional explanation dependency is absent; the
    fallback is explicitly reason-code logic and is never represented as SHAP.
    """
    try:
        model, matrix, names = _tree_shap_inputs(pipeline, raw_frame)
        values = _tree_shap_values(model, matrix)
        result: list[list[str]] = []
        for row in values:
            order = np.argsort(np.abs(row))[::-1]
            reasons: list[str] = []
            for idx in order:
                reason = _reason_for_feature(str(names[idx]))
                if reason not in reasons:
                    reasons.append(reason)
                if len(reasons) >= top_k:
                    break
            result.append(reasons)
        return result
    except Exception as exc:
        logger.debug("SHAP explanation unavailable; using reason-code fallback: %s", exc)

    outputs: list[list[str]] = []
    for _, row in raw_frame.iterrows():
        reasons: list[str] = []
        if float(row.get("TransactionAmt", 0) or 0) > 500:
            reasons.append("unusual_transaction_amount")
        if pd.notna(row.get("DeviceInfo")):
            reasons.append("device_behavior_signal")
        reasons.append("model_feature_combination")
        outputs.append(reasons[:top_k])
    return outputs


def save_shap_summary(
    pipeline: Any,
    raw_frame: pd.DataFrame,
    output_path: Path,
    max_rows: int = 500,
) -> bool:
    """Save a real TreeSHAP beeswarm summary for a bounded raw-data sample."""
    try:
        import matplotlib.pyplot as plt
        import shap

        sample = raw_frame.head(max_rows)
        model, matrix, names = _tree_shap_inputs(pipeline, sample)
        values = _tree_shap_values(model, matrix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(values, matrix, feature_names=names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Could not save SHAP summary: %s", exc)
        return False


def save_shap_feature_importance(
    pipeline: Any,
    raw_frame: pd.DataFrame,
    output_path: Path,
    max_rows: int = 500,
) -> bool:
    """Save mean absolute fraud-class SHAP feature importance."""
    try:
        import matplotlib.pyplot as plt

        sample = raw_frame.head(max_rows)
        model, matrix, names = _tree_shap_inputs(pipeline, sample)
        values = _tree_shap_values(model, matrix)
        importance = np.abs(values).mean(axis=0)
        order = np.argsort(importance)[-20:]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.barh(names[order], importance[order])
        plt.xlabel("Mean |SHAP value|")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Could not save SHAP feature importance: %s", exc)
        return False


def save_individual_explanation(
    pipeline: Any,
    raw_frame: pd.DataFrame,
    output_path: Path,
) -> bool:
    """Save signed fraud-class SHAP contributions for one real transaction."""
    try:
        import matplotlib.pyplot as plt

        sample = raw_frame.head(1)
        model, matrix, names = _tree_shap_inputs(pipeline, sample)
        values = _tree_shap_values(model, matrix)
        row = values[0]
        order = np.argsort(np.abs(row))[-12:]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.barh(names[order], row[order])
        plt.axvline(0.0, linewidth=0.8)
        plt.xlabel("SHAP contribution (positive → fraud class)")
        plt.title("Individual Transaction Explanation")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Could not save individual SHAP explanation: %s", exc)
        return False
