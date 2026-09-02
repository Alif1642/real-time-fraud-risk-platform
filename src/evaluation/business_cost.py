"""Business-cost-aware decision threshold optimization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostBreakdown:
    threshold: float
    false_positives: int
    false_negatives: int
    reviews: int
    total_cost: float


def decision_labels(probabilities: np.ndarray, block_threshold: float, review_lower: float = 0.30) -> np.ndarray:
    """Return APPROVE/REVIEW/BLOCK decisions."""
    p = np.asarray(probabilities, dtype=float)
    return np.where(p >= block_threshold, "BLOCK", np.where(p >= review_lower, "REVIEW", "APPROVE"))


def calculate_business_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    block_threshold: float,
    fp_cost: float = 10.0,
    fn_cost: float = 500.0,
    review_cost: float = 5.0,
    review_lower: float = 0.30,
) -> CostBreakdown:
    """Calculate project business cost for approve/review/block outcomes."""
    y = np.asarray(y_true, dtype=int)
    decisions = decision_labels(probabilities, block_threshold, review_lower)
    blocked = decisions == "BLOCK"
    approved = decisions == "APPROVE"
    reviews = int((decisions == "REVIEW").sum())
    fp = int(((y == 0) & blocked).sum())
    # Fraud sent to REVIEW is treated as caught by review; FN here means fraud approved.
    fn = int(((y == 1) & approved).sum())
    total = fp * fp_cost + fn * fn_cost + reviews * review_cost
    return CostBreakdown(float(block_threshold), fp, fn, reviews, float(total))


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fp_cost: float = 10.0,
    fn_cost: float = 500.0,
    review_cost: float = 5.0,
    review_lower: float = 0.30,
    candidates: np.ndarray | None = None,
) -> CostBreakdown:
    """Find block threshold minimizing cost on validation-only predictions."""
    candidates = candidates if candidates is not None else np.linspace(max(review_lower + 0.01, 0.31), 0.99, 69)
    results = [
        calculate_business_cost(y_true, probabilities, float(t), fp_cost, fn_cost, review_cost, review_lower)
        for t in candidates
    ]
    return min(results, key=lambda r: (r.total_cost, -r.threshold))
