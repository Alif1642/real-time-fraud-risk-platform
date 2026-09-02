"""Central configuration loaded from the repository .env file and environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _path_from_env(name: str, default: str) -> Path:
    raw = Path(os.getenv(name, default)).expanduser()
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    random_state: int = _int("RANDOM_STATE", 42)
    model_n_estimators: int = _int("MODEL_N_ESTIMATORS", 100)
    logistic_max_iter: int = _int("LOGISTIC_MAX_ITER", 20)
    train_fraction: float = _float("TRAIN_FRACTION", 0.70)
    validation_fraction: float = _float("VALIDATION_FRACTION", 0.15)
    test_fraction: float = _float("TEST_FRACTION", 0.15)
    fp_cost: float = _float("FP_COST", 10.0)
    fn_cost: float = _float("FN_COST", 500.0)
    review_cost: float = _float("REVIEW_COST", 5.0)
    review_lower_bound: float = _float("REVIEW_LOWER_BOUND", 0.30)
    max_batch_size: int = _int("MAX_BATCH_SIZE", 500)
    psi_warning: float = _float("PSI_WARNING", 0.20)
    psi_critical: float = _float("PSI_CRITICAL", 0.25)
    enable_database: bool = _bool("ENABLE_DATABASE", False)
    enable_mlflow: bool = _bool("ENABLE_MLFLOW", True)
    enable_evidently: bool = _bool("ENABLE_EVIDENTLY", False)
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://fraud:change_me_local@localhost:5432/fraud"
    )
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "fraud-risk-platform")
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    cors_origins: tuple[str, ...] = tuple(
        x.strip()
        for x in os.getenv(
            "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        ).split(",")
        if x.strip()
    )

    @property
    def raw_dir(self) -> Path:
        return _path_from_env("RAW_DATA_DIR", "data/raw")


    @property
    def model_dir(self) -> Path:
        return _path_from_env("MODEL_DIR", "models")

    @property
    def report_dir(self) -> Path:
        return _path_from_env("REPORT_DIR", "reports")

    def validate_split(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Train/validation/test fractions must sum to 1.0, got {total:.4f}")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("Train/validation/test fractions must all be greater than zero.")


settings = Settings()
