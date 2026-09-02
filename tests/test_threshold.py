import numpy as np

from src.evaluation.business_cost import calculate_business_cost, optimize_threshold


def test_business_cost_calculation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.95])
    result = calculate_business_cost(y, p, block_threshold=0.8, fp_cost=10, fn_cost=500, review_cost=5, review_lower=0.3)
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.total_cost == 510


def test_threshold_optimizer_returns_validation_candidate():
    y = np.array([0,0,0,1,1,1])
    p = np.array([0.05,0.2,0.6,0.4,0.8,0.95])
    candidates = np.array([0.5, 0.7, 0.9])
    result = optimize_threshold(y, p, candidates=candidates)
    assert result.threshold in candidates
