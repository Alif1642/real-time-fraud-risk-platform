"""Pydantic request/response contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    TransactionID: int | None = None
    TransactionDT: float | None = 0
    TransactionAmt: float = Field(ge=0)
    ProductCD: str | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    C1: float | None = None
    C2: float | None = None
    D1: float | None = None
    D2: float | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    id_01: float | None = None
    id_02: float | None = None
    id_31: str | None = None


class PredictionResponse(BaseModel):
    transaction_id: int | None
    fraud_probability: float
    prediction: int
    risk_level: str
    decision: str
    threshold: float
    model_version: str
    reason_codes: list[str]
    prediction_timestamp: datetime


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    calibration_method: str
    threshold: float
    metrics: dict[str, Any]
