from src.data.loader import load_training_data
from src.data.validator import validate_training_frame


def test_real_ieee_sample_validates(real_data_available):
    df = load_training_data(sample_size=2_000)
    report = validate_training_frame(df)
    assert report.rows == len(df)
    assert report.missing_target == 0
    assert report.duplicate_transaction_ids == 0
    assert 0 < report.fraud_rate < 1
