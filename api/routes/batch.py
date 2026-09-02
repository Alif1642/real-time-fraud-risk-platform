"""Batch scoring route."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_bundle
from api.schemas import BatchRequest, BatchResponse, PredictionResponse
from src.config import settings
from src.data.validator import validate_inference_frame
from src.models.predict import score_dataframe

router = APIRouter()


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    if not request.transactions:
        raise HTTPException(status_code=422, detail="transactions cannot be empty")
    if len(request.transactions) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"Maximum batch size is {settings.max_batch_size}")
    try:
        frame = pd.DataFrame([x.model_dump() for x in request.transactions])
        validate_inference_frame(frame)
        rows = score_dataframe(get_bundle(), frame, include_explanations=True)
        return BatchResponse(predictions=[PredictionResponse(**r) for r in rows])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
