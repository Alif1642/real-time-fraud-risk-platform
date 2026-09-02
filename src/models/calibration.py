"""Leakage-safe post-hoc probability calibration."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class ProbabilityCalibrator:
    """Map raw model probabilities to calibrated probabilities."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic", "identity"}:
            raise ValueError("method must be sigmoid, isotonic, or identity")
        self.method = method
        self.model = None

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray):
        p = np.asarray(raw_prob, dtype=float)
        y = np.asarray(y_true, dtype=int)
        if len(np.unique(y)) < 2 and self.method != "identity":
            raise ValueError("Calibration requires both target classes.")
        if self.method == "sigmoid":
            eps = 1e-6
            logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
            self.model = LogisticRegression(solver="lbfgs").fit(logit, y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        p = np.asarray(raw_prob, dtype=float)
        if self.method == "identity" or self.model is None:
            return np.clip(p, 0, 1)
        if self.method == "sigmoid":
            eps = 1e-6
            logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
            return self.model.predict_proba(logit)[:, 1]
        return np.asarray(self.model.predict(p), dtype=float)


def choose_calibrator(
    raw_prob: np.ndarray,
    y_true: np.ndarray,
    selection_fraction: float = 0.70,
) -> tuple[ProbabilityCalibrator, dict[str, float]]:
    """Select raw/sigmoid/isotonic using a later calibration-selection slice, then refit.

    The base model is already frozen. The first part of this calibration window fits candidate
    mappings; the later part selects by Brier score. The chosen mapping is then refit on the full
    calibration window before the separate threshold-selection window is scored.
    """
    p = np.asarray(raw_prob, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if len(p) != len(y) or len(p) < 20:
        identity = ProbabilityCalibrator("identity").fit(p, y)
        return identity, {"identity": float(brier_score_loss(y, identity.predict(p)))}

    split = int(len(p) * selection_fraction)
    split = min(max(split, 10), len(p) - 10)
    p_fit, y_fit = p[:split], y[:split]
    p_eval, y_eval = p[split:], y[split:]

    methods = ["identity", "sigmoid"]
    if len(p_fit) >= 200 and len(np.unique(y_fit)) > 1:
        methods.append("isotonic")

    scores: dict[str, float] = {}
    for method in methods:
        candidate = ProbabilityCalibrator(method)
        try:
            candidate.fit(p_fit, y_fit)
            scores[method] = float(brier_score_loss(y_eval, candidate.predict(p_eval)))
        except ValueError:
            continue

    best_method = min(scores, key=scores.get) if scores else "identity"
    chosen = ProbabilityCalibrator(best_method)
    try:
        chosen.fit(p, y)
    except ValueError:
        chosen = ProbabilityCalibrator("identity").fit(p, y)
        best_method = "identity"
    return chosen, scores
