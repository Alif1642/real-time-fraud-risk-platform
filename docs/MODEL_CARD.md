# Model Card

> **This is a portfolio/research project and not a production banking fraud system.**

## Intended use
Demonstrate end-to-end fraud-risk ML engineering: chronological validation, imbalanced learning, calibration, cost-aware thresholding, explainability, serving and monitoring.

## Non-intended use
Do not use this repository as-is to block real customer transactions, make credit decisions, identify criminals, or satisfy regulated model-risk obligations.

## Dataset
IEEE-CIS Fraud Detection from Kaggle. The raw dataset is local-only and is not distributed with this repository.

## Training process
Candidate Logistic Regression, LightGBM and XGBoost pipelines consume raw selected attributes. Feature mappings are fit on training data only. The winning candidate is selected by validation PR-AUC.

## Validation strategy
Chronological `TransactionDT` split: oldest 70% train, next 15% validation, newest 15% final test. The validation portion is split again chronologically into calibration and threshold-selection windows.

## Metrics
PR-AUC is primary. ROC-AUC, precision, recall, F1, Recall@Fixed-FPR, Precision@Fixed-FPR, Brier score and business cost are also generated. **Metrics are generated after running the training pipeline.**

## Bias considerations
The public dataset contains anonymized proxies and may not reflect current deployment populations. Correlated attributes can create uneven error rates. Real deployment requires legal review, subgroup evaluation where lawful/available, fairness analysis and human escalation policies.

## Monitoring
Monitor feature/missingness distribution change, probability/risk-level distribution change and delayed-label performance. PSI thresholds in this project are configurable alerts, not universal standards.

## Retraining strategy
Retrain only after enough recent labeled transactions accumulate, compare the challenger against the incumbent on forward-time validation, recalibrate probabilities, re-optimize thresholds on validation costs, then shadow/canary before promotion.
