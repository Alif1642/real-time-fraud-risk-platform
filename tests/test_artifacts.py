from src.models.predict import load_artifacts


def test_real_artifacts_load(trained_artifacts):
    bundle = load_artifacts(trained_artifacts)
    assert 0 < bundle.threshold < 1
    assert bundle.metadata.get("result_label") == "LOCAL IEEE-CIS HOLDOUT RESULTS"
