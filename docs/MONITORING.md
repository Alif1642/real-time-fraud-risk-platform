# Monitoring

## Distinctions
- **Data drift:** distribution of model inputs changes.
- **Prediction drift:** distribution of model scores/decisions changes.
- **Concept drift:** relationship between inputs and fraud label changes.
- **Performance degradation:** labeled metrics such as PR-AUC/recall worsen.

## Implemented checks
`src/monitoring/drift.py` implements PSI, KS statistics, missing-rate comparison, categorical total-variation distance, project alert statuses and optional Evidently HTML report generation.

## Project alert thresholds
Default `PSI_WARNING=0.20` and `PSI_CRITICAL=0.25` are configurable demonstration thresholds. They are **not universal industry standards**.

## Delayed labels
Once investigation/chargeback labels arrive, `src/monitoring/performance.py` recalculates precision, recall, PR-AUC, Brier score and observed fraud rate using the deployed threshold.
