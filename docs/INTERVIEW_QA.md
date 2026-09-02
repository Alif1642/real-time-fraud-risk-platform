# Interview Q&A

1. **Why is accuracy not primary?** Fraud is rare; a model predicting every transaction legitimate can have high accuracy while catching zero fraud.
2. **Why PR-AUC?** It emphasizes precision-recall trade-offs for the minority fraud class and is more informative than accuracy under severe imbalance.
3. **Why temporal validation?** Deployment predicts future behavior from past data. Chronological validation approximates that constraint and exposes drift.
4. **What is data leakage?** Any training-time access to information unavailable at prediction time, including future aggregates or target-derived features.
5. **How can SMOTE cause leakage?** If applied before splitting, synthetic points blend information across future validation/test observations into training.
6. **Why not random train-test split?** Randomization mixes time periods, can inflate performance through recurring entities/patterns and underestimates forward drift.
7. **Why LightGBM/XGBoost?** They are strong nonlinear tabular learners, handle complex interactions and typically provide excellent ranking performance.
8. **Why Logistic Regression baseline?** It is fast, interpretable, stable and establishes whether complexity materially adds value.
9. **How did you select the threshold?** On a validation-only time window by minimizing configurable FP/FN/review business cost; never on the test set.
10. **Why optimize business cost?** The operational harm of missing fraud and inconveniencing legitimate customers is asymmetric; F1 does not encode dollar impact.
11. **What is calibration?** Mapping raw scores so predicted probabilities better correspond to observed event frequencies.
12. **Why Brier score?** It measures squared probabilistic error, directly rewarding well-calibrated probabilities.
13. **What is SHAP?** A game-theoretic attribution method that estimates each feature's contribution to a prediction relative to a baseline.
14. **How does SHAP help fraud analysts?** It converts a score into ranked contributing signals, enabling faster manual review and debugging; this UI exposes reason codes rather than raw SHAP numbers.
15. **What is concept drift?** The conditional relationship P(y|x) changes, so the same inputs imply different fraud risk.
16. **What is data drift?** The feature distribution P(x) changes, whether or not model performance has yet degraded.
17. **How would you retrain?** Accumulate recent labels, build forward-time train/validation/test windows, retrain challenger, recalibrate, re-optimize threshold, compare cost/PR-AUC, then canary and monitor.
18. **How would you deploy to AWS?** Containerize API, store artifacts in S3/registry, run on ECS/Fargate or SageMaker endpoint, use RDS PostgreSQL, CloudWatch, IAM and Secrets Manager.
19. **How would you handle millions of transactions?** Stateless horizontally scaled scoring workers, online feature store/cache, async event logging, partitioned analytics storage and controlled model refresh.
20. **How would you reduce API latency?** Preload artifacts, avoid per-request DB round trips on the critical path, cache feature state, batch where possible, profile SHAP, and compute detailed explanations asynchronously/off critical path in a real system.
21. **Why PostgreSQL?** Strong transactional semantics, indexing and SQL analytics for prediction-event metadata with a mature Python ecosystem.
22. **Why MLflow?** Reproducible experiment metadata, parameters, metrics and artifacts; it makes candidate comparisons auditable.
23. **Why Docker?** It standardizes Python/system dependencies and provides reproducible local/CI/deployment execution.
24. **Why FastAPI?** Typed Pydantic validation, async-capable web stack and automatic OpenAPI/Swagger documentation.
25. **How would you monitor degradation?** Track input/missingness drift, score/decision drift, delayed-label PR-AUC/recall/calibration, latency and business-cost proxies, then alert and investigate before retraining.
