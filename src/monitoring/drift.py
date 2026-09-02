"""Lightweight drift monitoring with optional Evidently report generation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import settings

logger = logging.getLogger(__name__)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate PSI for numeric distributions using reference quantile bins."""
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    edges = np.unique(ref.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_counts = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def numeric_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return PSI, KS and missing-rate changes for shared numeric columns."""
    rows = []
    for col in columns:
        if col not in reference or col not in current:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce")
        cur = pd.to_numeric(current[col], errors="coerce")
        ks = (
            ks_2samp(ref.dropna(), cur.dropna()).statistic
            if ref.notna().sum() and cur.notna().sum()
            else 0.0
        )
        rows.append(
            {
                "feature": col,
                "psi": population_stability_index(ref.to_numpy(), cur.to_numpy()),
                "ks_statistic": float(ks),
                "reference_missing_rate": float(ref.isna().mean()),
                "current_missing_rate": float(cur.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    """Total-variation distance between categorical distributions."""
    r = reference.astype("string").fillna("<MISSING>").value_counts(normalize=True)
    c = current.astype("string").fillna("<MISSING>").value_counts(normalize=True)
    cats = r.index.union(c.index)
    return float(
        0.5 * (r.reindex(cats, fill_value=0) - c.reindex(cats, fill_value=0)).abs().sum()
    )


def drift_status(psi: float, warning: float = 0.20, critical: float = 0.25) -> str:
    """Map PSI to configurable project status. Thresholds are not universal standards."""
    if psi > critical:
        return "CRITICAL"
    if psi > warning:
        return "WARNING"
    return "OK"


def build_evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_html: Path
) -> bool:
    """Generate a Data Drift HTML report when optional Evidently is enabled/installed.

    Evidently 0.7+ changed its public API. This implementation uses the current API first and
    retains a legacy fallback for older environments. Core PSI/KS monitoring is independent.
    """
    if not settings.enable_evidently:
        logger.info("Evidently report skipped because ENABLE_EVIDENTLY=false")
        return False
    output_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        result = report.run(current, reference)
        result.save_html(str(output_html))
        return True
    except (ImportError, AttributeError) as current_exc:
        logger.debug("Current Evidently API unavailable: %s", current_exc)
        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            report.save_html(str(output_html))
            return True
        except Exception as legacy_exc:
            logger.warning("Evidently report unavailable: %s", legacy_exc)
            return False
    except Exception as exc:
        logger.warning("Evidently report generation failed: %s", exc)
        return False
