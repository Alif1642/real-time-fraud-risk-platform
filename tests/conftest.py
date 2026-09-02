from __future__ import annotations

from pathlib import Path

import pytest

from src.config import settings


@pytest.fixture(scope="session")
def real_data_available() -> Path:
    required = [
        settings.raw_dir / "train_transaction.csv",
        settings.raw_dir / "train_identity.csv",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Real IEEE-CIS training CSVs are not present in data/raw/.")
    return settings.raw_dir


@pytest.fixture(scope="session")
def trained_artifacts() -> Path:
    model_dir = settings.model_dir
    required = [
        model_dir / "model.joblib",
        model_dir / "calibrator.joblib",
        model_dir / "threshold.json",
        model_dir / "model_metadata.json",
        model_dir / "feature_schema.json",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Model artifacts are not present. Run `python -m src.models.train` first.")
    return model_dir
