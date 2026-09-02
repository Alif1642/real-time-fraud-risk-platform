from fastapi.testclient import TestClient

from api.dependencies import clear_bundle_cache
from api.main import app


def test_api_health_prediction_batch_and_validation(trained_artifacts):
    clear_bundle_cache()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {"TransactionAmt": 125.5, "ProductCD": "W", "card1": 12345, "card4": "visa"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["prediction"] in {0, 1}
    assert body["decision"] in {"APPROVE", "REVIEW", "BLOCK"}

    batch = client.post("/predict/batch", json={"transactions": [payload, payload]})
    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2

    invalid = client.post("/predict", json={"TransactionAmt": -1})
    assert invalid.status_code == 422
