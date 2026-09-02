"""Fraud-sensitive metrics and report plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_precision_at_fixed_fpr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fixed_fpr: float = 0.01,
) -> dict[str, float]:
    """Report recall and precision at the highest observed recall within a fixed FPR cap."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "fixed_fpr": float(fixed_fpr),
            "fixed_fpr_threshold": float("nan"),
            "recall_at_fixed_fpr": float("nan"),
            "precision_at_fixed_fpr": float("nan"),
        }
    fpr, _tpr, thresholds = roc_curve(y, p)
    valid = np.where(fpr <= fixed_fpr)[0]
    idx = valid[-1] if len(valid) else 0
    threshold = thresholds[idx]
    pred = (p >= threshold).astype(int)
    return {
        "fixed_fpr": float(fixed_fpr),
        "fixed_fpr_threshold": float(threshold),
        "recall_at_fixed_fpr": float(recall_score(y, pred, zero_division=0)),
        "precision_at_fixed_fpr": float(precision_score(y, pred, zero_division=0)),
    }


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate imbalance-aware classification, calibration and confusion metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    negative_count = tn + fp
    out: dict[str, float | int] = {
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fraud_capture_rate": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / negative_count) if negative_count else 0.0,
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    out.update(recall_precision_at_fixed_fpr(y, p))
    return out


def save_evaluation_plots(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
    prefix: str,
) -> None:
    """Persist precision-recall, calibration and confusion-matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    precision, recall, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} Precision-Recall")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_pr_curve.png", dpi=140)
    plt.close()

    if len(np.unique(y)) > 1:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.figure(figsize=(6, 4))
        plt.plot(mean, frac, marker="o", label="model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraud rate")
        plt.title(f"{prefix} Calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_calibration.png", dpi=140)
        plt.close()

    pred = (p >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y, pred)
    plt.title(f"{prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=140)
    plt.close()


def save_metrics_json(metrics: dict[str, float | int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")


def save_model_comparison(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
