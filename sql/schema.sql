CREATE TABLE IF NOT EXISTS prediction_events (
    id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transaction_amount DOUBLE PRECISION NULL,
    ground_truth_is_fraud SMALLINT NULL CHECK (
        ground_truth_is_fraud IN (0, 1)
    ),
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (
        fraud_probability BETWEEN 0 AND 1
    ),
    risk_level VARCHAR(16) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_events_timestamp ON prediction_events (timestamp);

CREATE INDEX IF NOT EXISTS idx_prediction_events_transaction_id ON prediction_events (transaction_id);

CREATE INDEX IF NOT EXISTS idx_prediction_events_decision ON prediction_events (decision);

CREATE INDEX IF NOT EXISTS idx_prediction_events_model_version ON prediction_events (model_version);

CREATE INDEX IF NOT EXISTS idx_prediction_events_label ON prediction_events (ground_truth_is_fraud);
