from src.data.loader import load_training_data
from src.features.feature_engineering import FraudFeatureEngineer


def test_feature_generation_on_real_ieee_rows(real_data_available):
    df = load_training_data(sample_size=2_000).drop(columns=["isFraud"])
    fe = FraudFeatureEngineer().fit(df.iloc[:1_500])
    out = fe.transform(df.iloc[1_500:])
    expected = {
        "hour",
        "day_of_week",
        "TransactionAmt_log1p",
        "transaction_count_by_card",
        "amount_deviation_from_card_average",
    }
    assert expected.issubset(out.columns)
    assert out["TransactionAmt_log1p"].notna().all()
