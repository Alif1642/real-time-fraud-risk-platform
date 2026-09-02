"""Health and model metadata routes."""
from fastapi import APIRouter

from api.dependencies import get_bundle
from api.schemas import HealthResponse, ModelInfoResponse
from src.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_bundle()
        loaded = True
    except Exception:
        loaded = False
    db_status = "enabled" if settings.enable_database else "disabled (optional)"
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded, database=db_status)


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    bundle = get_bundle()
    m = bundle.metadata
    return ModelInfoResponse(
        model_name=m.get("model_name", "unknown"),
        model_version=m.get("model_version", "unknown"),
        calibration_method=m.get("calibration_method", "unknown"),
        threshold=bundle.threshold,
        metrics=m.get("test_metrics", {}),
    )
