import pandas as pd

from src.models.predict import load_artifacts, score_dataframe


def test_prediction_output(trained_artifacts):
    bundle = load_artifacts(trained_artifacts)
    # Deliberately provide only a minimal direct input. score_dataframe must align it
    # to the saved raw feature schema, just as the API's Pydantic model does.
    row = pd.DataFrame([{"TransactionAmt": 125.5, "ProductCD": "W", "card1": 12345}])
    result = score_dataframe(bundle, row, include_explanations=False)[0]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in {0, 1}
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
