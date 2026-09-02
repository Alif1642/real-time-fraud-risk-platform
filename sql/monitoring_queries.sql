-- Daily high-risk / blocked / review volumes
SELECT date_trunc('day', timestamp) AS day,
       COUNT(*) FILTER (WHERE risk_level = 'HIGH') AS high_risk_transactions,
       COUNT(*) FILTER (WHERE decision = 'BLOCK') AS blocked_transactions,
       COUNT(*) FILTER (WHERE decision = 'REVIEW') AS review_transactions,
       AVG(fraud_probability) AS avg_fraud_probability,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY fraud_probability) AS median_fraud_probability
FROM prediction_events
GROUP BY 1 ORDER BY 1;

-- Probability bands for prediction-drift dashboards
SELECT width_bucket(fraud_probability, 0.0, 1.0, 10) AS probability_bucket,
       COUNT(*) AS count
FROM prediction_events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY 1 ORDER BY 1;

-- API latency by model version
SELECT model_version,
       COUNT(*) AS predictions,
       AVG(latency_ms) AS avg_latency_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
FROM prediction_events
GROUP BY model_version;
