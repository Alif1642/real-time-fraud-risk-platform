"""Monitoring status route."""
from fastapi import APIRouter

from src.config import settings

router = APIRouter()


@router.get("/monitoring/status")
def monitoring_status() -> dict:
    return {
        "status": "configured",
        "psi_warning": settings.psi_warning,
        "psi_critical": settings.psi_critical,
        "note": "These are configurable project thresholds, not universal industry standards.",
        "drift_types": ["data_drift", "prediction_drift", "concept_drift", "performance_degradation"],
    }
