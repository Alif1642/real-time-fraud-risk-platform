"""Optional SQLAlchemy persistence for prediction events."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PredictionEvent(Base):
    __tablename__ = "prediction_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    transaction_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_truth_is_fraud: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    threshold: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)


def get_engine():
    """Create the configured SQLAlchemy engine only when database use is enabled."""
    if not settings.enable_database:
        raise RuntimeError("PostgreSQL is disabled. Set ENABLE_DATABASE=true to enable persistence.")
    return create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=1800)


def init_db() -> None:
    """Create tables if PostgreSQL is enabled and reachable."""
    Base.metadata.create_all(get_engine())


def persist_prediction(
    result: dict[str, Any], latency_ms: float, transaction_amount: float | None = None
) -> bool:
    """Persist non-sensitive prediction metadata; return False when disabled/unavailable.

    Scoring must remain available in local mode even when PostgreSQL is not installed or running.
    """
    if not settings.enable_database:
        return False
    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        with Session.begin() as session:
            session.add(
                PredictionEvent(
                    transaction_id=result.get("transaction_id"),
                    timestamp=datetime.now(timezone.utc),
                    transaction_amount=transaction_amount,
                    ground_truth_is_fraud=None,
                    fraud_probability=result["fraud_probability"],
                    risk_level=result["risk_level"],
                    decision=result["decision"],
                    threshold=result["threshold"],
                    model_version=result["model_version"],
                    latency_ms=float(latency_ms),
                )
            )
        return True
    except Exception as exc:
        logger.warning("Prediction persistence skipped: %s", exc)
        return False
