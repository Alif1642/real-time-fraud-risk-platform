"""Single transaction scoring route."""
from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_bundle
from api.schemas import PredictionResponse, TransactionRequest
from src.data.validator import validate_inference_frame
from src.database.postgres import persist_prediction
from src.models.predict import score_dataframe

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    started = time.perf_counter()
    try:
        frame = pd.DataFrame([request.model_dump()])
        validate_inference_frame(frame)
        result = score_dataframe(get_bundle(), frame, include_explanations=True)[0]
        latency = (time.perf_counter() - started) * 1000
        persist_prediction(result, latency, transaction_amount=request.TransactionAmt)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
