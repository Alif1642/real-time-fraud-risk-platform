"""Artifact loading and transaction scoring, including a beginner-friendly CLI."""
from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.evaluation.business_cost import decision_labels
from src.explainability.shap_explainer import explain_rows

logger = logging.getLogger(__name__)


@dataclass
class ArtifactBundle:
    model: Any
    calibrator: Any
    threshold: float
    metadata: dict[str, Any]
    feature_schema: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(model_dir: Path | None = None) -> ArtifactBundle:
    """Load trained artifacts from disk with a clear regeneration error on incompatibility."""
    model_dir = Path(model_dir or settings.model_dir)
    required = [
        "model.joblib",
        "calibrator.joblib",
        "threshold.json",
        "model_metadata.json",
        "feature_schema.json",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. Run `python -m src.models.train` after placing the IEEE-CIS files in data/raw/."
        )

    metadata = _load_json(model_dir / "model_metadata.json")
    trained_python = str(metadata.get("runtime", {}).get("python", "unknown"))
    current_python = platform.python_version()
    if trained_python != "unknown" and trained_python.split(".")[:2] != current_python.split(".")[:2]:
        logger.warning(
            "Artifacts were created with Python %s but current Python is %s. "
            "Regenerate with `python -m src.models.train` if loading fails.",
            trained_python,
            current_python,
        )

    try:
        model = joblib.load(model_dir / "model.joblib")
        calibrator = joblib.load(model_dir / "calibrator.joblib")
    except Exception as exc:
        raise RuntimeError(
            "Saved model artifacts are incompatible with this Python/library environment. "
            "Regenerate them with `python -m src.models.train`."
        ) from exc

    threshold_data = _load_json(model_dir / "threshold.json")
    return ArtifactBundle(
        model=model,
        calibrator=calibrator,
        threshold=float(threshold_data["block_threshold"]),
        metadata=metadata,
        feature_schema=_load_json(model_dir / "feature_schema.json"),
    )


def _prepare_frame(bundle: ArtifactBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Align direct/CLI inputs to the raw columns used when the pipeline was fitted."""
    aligned = frame.copy()
    required = bundle.feature_schema.get("required_inference_columns", ["TransactionAmt"])
    missing_required = [c for c in required if c not in aligned.columns]
    if missing_required:
        raise ValueError(f"Missing required prediction columns: {missing_required}")

    raw_columns = bundle.feature_schema.get("raw_training_columns", [])
    for col in raw_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0 if col == "TransactionDT" else np.nan
    if raw_columns:
        aligned = aligned[[c for c in raw_columns if c in aligned.columns]]
    return aligned


def _risk_level(probability: float, threshold: float, review_lower: float) -> str:
    if probability >= threshold:
        return "HIGH"
    if probability >= review_lower:
        return "MEDIUM"
    return "LOW"


def score_dataframe(
    bundle: ArtifactBundle, frame: pd.DataFrame, include_explanations: bool = True
) -> list[dict[str, Any]]:
    """Score raw transaction rows using calibrated probabilities and the saved threshold."""
    aligned = _prepare_frame(bundle, frame)
    raw = bundle.model.predict_proba(aligned)[:, 1]
    calibrated = np.asarray(bundle.calibrator.predict(raw), dtype=float)
    decisions = decision_labels(calibrated, bundle.threshold, settings.review_lower_bound)
    reasons = (
        explain_rows(bundle.model, aligned, top_k=3)
        if include_explanations
        else [[] for _ in range(len(aligned))]
    )
    now = datetime.now(timezone.utc).isoformat()
    outputs: list[dict[str, Any]] = []
    for i, probability in enumerate(calibrated):
        tx_id = aligned.iloc[i].get("TransactionID")
        p = float(np.clip(probability, 0, 1))
        outputs.append(
            {
                "transaction_id": int(tx_id) if pd.notna(tx_id) else None,
                "fraud_probability": p,
                "prediction": int(p >= bundle.threshold),
                "risk_level": _risk_level(p, bundle.threshold, settings.review_lower_bound),
                "decision": str(decisions[i]),
                "threshold": float(bundle.threshold),
                "model_version": bundle.metadata.get("model_version", "unknown"),
                "reason_codes": reasons[i],
                "prediction_timestamp": now,
            }
        )
    return outputs


def main() -> None:
    """Score one transaction from PowerShell: `python -m src.models.predict`."""
    parser = argparse.ArgumentParser(description="Score one transaction with saved artifacts.")
    parser.add_argument("--amount", type=float, default=125.50, help="Transaction amount")
    parser.add_argument("--product", default="W", help="ProductCD")
    parser.add_argument("--card1", type=float, default=12345, help="card1 value")
    parser.add_argument("--card4", default="visa", help="card4 value")
    parser.add_argument("--device", default="desktop", help="DeviceType")
    args = parser.parse_args()
    if args.amount < 0:
        parser.error("--amount must be >= 0")

    bundle = load_artifacts()
    frame = pd.DataFrame(
        [
            {
                "TransactionDT": 0,
                "TransactionAmt": args.amount,
                "ProductCD": args.product,
                "card1": args.card1,
                "card4": args.card4,
                "DeviceType": args.device,
            }
        ]
    )
    print(json.dumps(score_dataframe(bundle, frame, include_explanations=True)[0], indent=2))


if __name__ == "__main__":
    main()
