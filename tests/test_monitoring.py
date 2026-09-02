import numpy as np
import pandas as pd

from src.monitoring.drift import drift_status, numeric_drift, population_stability_index


def test_monitoring_functions():
    ref = np.linspace(0, 1, 500)
    cur = np.linspace(0.5, 1.5, 500)
    psi = population_stability_index(ref, cur)
    assert psi >= 0
    df = numeric_drift(pd.DataFrame({"x": ref}), pd.DataFrame({"x": cur}), ["x"])
    assert {"psi", "ks_statistic"}.issubset(df.columns)
    assert drift_status(0.30) == "CRITICAL"
