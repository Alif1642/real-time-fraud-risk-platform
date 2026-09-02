-- Daily fraud rate after delayed labels become available.
SELECT date_trunc('day', timestamp) AS day,
       COUNT(*) FILTER (WHERE ground_truth_is_fraud IS NOT NULL) AS labeled_transactions,
       AVG(ground_truth_is_fraud::double precision) FILTER (WHERE ground_truth_is_fraud IS NOT NULL) AS daily_fraud_rate
FROM prediction_events
GROUP BY 1 ORDER BY 1;

-- High-risk and blocked transactions.
SELECT transaction_id, timestamp, transaction_amount, fraud_probability, risk_level, decision, model_version
FROM prediction_events
WHERE risk_level = 'HIGH' OR decision = 'BLOCK'
ORDER BY timestamp DESC;

-- Average transaction amount and fraud probability by day.
SELECT date_trunc('day', timestamp) AS day,
       AVG(transaction_amount) AS average_transaction_amount,
       AVG(fraud_probability) AS average_fraud_probability
FROM prediction_events
GROUP BY 1 ORDER BY 1;

-- Model/decision summaries.
SELECT model_version, decision, COUNT(*) AS count, AVG(fraud_probability) AS avg_probability
FROM prediction_events
GROUP BY model_version, decision
ORDER BY model_version, decision;
