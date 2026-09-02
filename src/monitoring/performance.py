"""Delayed-label performance monitoring."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics


def monitor_labeled_performance(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Recalculate core performance metrics once ground-truth labels arrive."""
    metrics = classification_metrics(y_true, probabilities, threshold)
    metrics["fraud_rate"] = float(np.asarray(y_true, dtype=int).mean())
    return metrics
