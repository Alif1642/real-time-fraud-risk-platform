"""Regression tests for SHAP binary-class output normalization."""
from __future__ import annotations

import numpy as np

from src.explainability.shap_explainer import _normalise_binary_shap_values


def test_binary_shap_normalizer_handles_list_and_3d_outputs() -> None:
    class0 = np.zeros((2, 3), dtype=float)
    class1 = np.arange(6, dtype=float).reshape(2, 3)

    from_list = _normalise_binary_shap_values([class0, class1], n_rows=2, n_features=3)
    assert np.array_equal(from_list, class1)

    rows_features_classes = np.stack([class0, class1], axis=2)
    from_3d = _normalise_binary_shap_values(
        rows_features_classes,
        n_rows=2,
        n_features=3,
    )
    assert np.array_equal(from_3d, class1)
