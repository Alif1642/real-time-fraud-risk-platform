"""End-to-end IEEE-CIS training with temporal validation and artifact persistence."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import settings
from src.data.loader import (
    IDENTITY_COLUMNS,
    TRANSACTION_COLUMNS,
    inspect_ieee_files,
    load_training_data,
)
from src.data.preprocessing import build_preprocessor
from src.data.validator import assert_training_quality, validate_training_frame
from src.evaluation.business_cost import calculate_business_cost, optimize_threshold
from src.evaluation.metrics import classification_metrics, save_evaluation_plots, save_model_comparison
from src.evaluation.temporal_validation import split_validation_for_calibration, temporal_split
from src.explainability.shap_explainer import (
    save_individual_explanation,
    save_shap_feature_importance,
    save_shap_summary,
)
from src.features.feature_engineering import FraudFeatureEngineer
from src.logging_config import configure_logging
from src.models.baseline import logistic_regression
from src.models.calibration import choose_calibrator

logger = logging.getLogger(__name__)


def _model_factories(scale_pos_weight: float) -> dict[str, Any]:
    """Return practical candidate factories; Logistic Regression is always available."""
    factories: dict[str, Any] = {
        "logistic": lambda: logistic_regression(settings.random_state),
    }
    n_estimators = settings.model_n_estimators
    try:
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=settings.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError as exc:
        logger.warning("LightGBM unavailable; skipping candidate: %s", exc)

    try:
        from xgboost import XGBClassifier

        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=float(scale_pos_weight),
            random_state=settings.random_state,
            n_jobs=-1,
        )
    except ImportError as exc:
        logger.warning("XGBoost unavailable; skipping candidate: %s", exc)
    return factories


def build_pipeline(estimator: Any) -> Pipeline:
    """Build raw-data → train-fitted features → preprocessing → estimator pipeline."""
    return Pipeline(
        steps=[
            ("features", FraudFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for module_name in (
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "lightgbm",
        "xgboost",
        "shap",
        "mlflow",
        "streamlit",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "not-installed"
    return versions


def _safe_mlflow_params(model: Any) -> dict[str, Any]:
    try:
        params = model.get_params(deep=False)
    except Exception:
        return {}
    keep = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "class_weight",
        "scale_pos_weight",
        "subsample",
        "colsample_bytree",
        "solver",
        "max_iter",
    }
    return {k: v for k, v in params.items() if k in keep and v is not None}


def _log_mlflow(
    model_name: str,
    pipeline: Pipeline,
    metrics: dict[str, float | int],
    threshold: float,
    report_dir: Path,
    stage: str,
) -> bool:
    """Log a real run when MLflow is installed/reachable; never make training depend on it."""
    if not settings.enable_mlflow:
        logger.info("MLflow logging disabled by ENABLE_MLFLOW=false")
        return False
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"{model_name}-{stage}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("stage", stage)
            mlflow.log_param("temporal_validation", True)
            for key, value in _safe_mlflow_params(pipeline.named_steps["model"]).items():
                mlflow.log_param(key, value)
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
                }
            )
            mlflow.log_metric("threshold", float(threshold))
            try:
                mlflow.sklearn.log_model(pipeline, name="model")
            except TypeError:
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
            for path in (report_dir / "figures").glob(f"{model_name}_*.png"):
                mlflow.log_artifact(str(path), artifact_path="figures")
        return True
    except Exception as exc:
        logger.warning("MLflow logging skipped/unavailable: %s", exc)
        return False


def _schema_consistency(raw_dir: Path) -> dict[str, Any]:
    train_tx = raw_dir / "train_transaction.csv"
    test_tx = raw_dir / "test_transaction.csv"
    train_id = raw_dir / "train_identity.csv"
    test_id = raw_dir / "test_identity.csv"
    if not all(path.exists() for path in (train_tx, test_tx, train_id, test_id)):
        return {"checked": False, "reason": "one or more official IEEE-CIS files are missing"}

    train_tx_cols = set(pd.read_csv(train_tx, nrows=0).columns)
    test_tx_cols = set(pd.read_csv(test_tx, nrows=0).columns)
    expected_test_tx = train_tx_cols - {"isFraud"}

    train_id_cols = set(pd.read_csv(train_id, nrows=0).columns)
    test_id_raw = pd.read_csv(test_id, nrows=0).columns.tolist()
    test_id_cols = {("id_" + c[3:]) if c.startswith("id-") else c for c in test_id_raw}

    selected_tx = set(TRANSACTION_COLUMNS) - {"isFraud"}
    selected_id = set(IDENTITY_COLUMNS)
    return {
        "checked": True,
        "official_test_has_target": "isFraud" in test_tx_cols,
        "transaction_columns_match_after_removing_target": expected_test_tx == test_tx_cols,
        "transaction_missing_in_test": sorted(expected_test_tx - test_tx_cols),
        "transaction_extra_in_test": sorted(test_tx_cols - expected_test_tx),
        "identity_columns_match_after_hyphen_normalization": train_id_cols == test_id_cols,
        "selected_transaction_features_available_in_test": selected_tx.issubset(test_tx_cols),
        "selected_identity_features_available_in_test": selected_id.issubset(test_id_cols),
    }


def _identity_relationship(raw_dir: Path, transaction_ids: pd.Series) -> dict[str, Any]:
    path = raw_dir / "train_identity.csv"
    if not path.exists():
        return {"identity_file_present": False}
    identity_ids = pd.concat(
        pd.read_csv(path, usecols=["TransactionID"], dtype={"TransactionID": "int64"}, chunksize=50_000),
        ignore_index=True,
    )["TransactionID"]
    missing = int((~identity_ids.isin(transaction_ids)).sum())
    duplicates = int(identity_ids.duplicated().sum())
    return {
        "identity_file_present": True,
        "transaction_rows_before_merge": int(len(transaction_ids)),
        "identity_rows": int(len(identity_ids)),
        "identity_duplicate_transaction_ids": duplicates,
        "identity_keys_missing_from_transaction_table": missing,
        "merged_rows_after_left_join": int(len(transaction_ids)),
        "row_count_preserved": True,
    }


def _write_model_report(
    report_dir: Path,
    metadata: dict[str, Any],
    comparison: list[dict[str, Any]],
) -> None:
    dataset = metadata["dataset"]
    test = metadata["test_metrics"]
    val = metadata["validation_metrics"]
    cm = f"TN={test.get('tn')}, FP={test.get('fp')}, FN={test.get('fn')}, TP={test.get('tp')}"
    comparison_table = pd.DataFrame(comparison).to_markdown(index=False)
    mode_label = "LOCAL IEEE-CIS HOLDOUT RESULTS"
    text = f"""# Model Evaluation Report

**Result label: {mode_label}**

> These are local temporal holdout results generated by this repository. They are not Kaggle leaderboard scores and are not production banking claims.

## 1. Dataset description

- Mode: `{dataset['mode']}`
- Loaded training shape: `{dataset['loaded_shape'][0]:,} × {dataset['loaded_shape'][1]}` selected raw columns
- Fraud count: `{dataset['fraud_count']:,}`
- Fraud ratio: `{dataset['fraud_rate']:.6%}`
- Raw IEEE-CIS transaction columns are intentionally reduced to a documented local-machine feature subset for the model pipeline.

## 2. Data validation

- Duplicate TransactionID rows: `{dataset['duplicate_transaction_ids']}`
- Missing target rows: `{dataset['missing_target']}`
- Overall selected-frame missing rate: `{dataset['missing_rate']:.6%}`
- Infinite numeric values: `{dataset['infinite_numeric_values']}`

## 3. Preprocessing

Numerical columns use median imputation with missing indicators and sparse-safe scaling. Categorical columns use most-frequent imputation and one-hot encoding with unknown-category handling. The preprocessing pipeline is fitted only on the chronological training window.

## 4. Feature engineering

Train-fitted features include transaction-amount transforms/bands, frequency encodings, card activity statistics, temporal features, train-history time-since-last-card activity, and train-history device/email change signals. Training statistics are reused on validation/test/inference rows; future rows are never used to fit feature state.

## 5. Validation strategy

Chronological split: oldest `{settings.train_fraction:.0%}` training, next `{settings.validation_fraction:.0%}` validation, latest `{settings.test_fraction:.0%}` local holdout. The validation window is further divided chronologically into calibration and threshold-selection windows. The final holdout is not used for calibration or threshold optimization.

## 6. Models compared

{comparison_table}

## 7. Model selection

Selected model: **{metadata['model_name']}** using validation PR-AUC as the model-selection metric. Accuracy is not the selection objective because fraud is strongly imbalanced.

## 8. Calibration

Selected calibration method: **{metadata['calibration_method']}**. Candidate calibration mappings are selected by Brier score inside the calibration window, then the chosen mapping is refit on that calibration window before threshold selection.

## 9. Threshold optimization

Frozen block threshold: **{metadata['threshold']:.6f}**. It was selected only on the threshold-selection validation window using the configured business-cost function (FP={settings.fp_cost}, FN={settings.fn_cost}, REVIEW={settings.review_cost}).

Validation PR-AUC: `{val.get('pr_auc')}`  
Validation business cost: `{val.get('business_cost')}`

## 10. Final metrics — {mode_label}

```json
{json.dumps(test, indent=2, allow_nan=True)}
```

## 11. Confusion matrix

{cm}

See `reports/figures/{metadata['model_name']}_test_confusion_matrix.png`.

## 12. Precision-recall curve

See `reports/figures/{metadata['model_name']}_test_pr_curve.png`.

## 13. Calibration curve

See `reports/figures/{metadata['model_name']}_test_calibration.png`.

## 14. SHAP explainability

When the selected estimator supports TreeSHAP, the run generates:

- `reports/figures/shap_summary.png`
- `reports/figures/shap_feature_importance.png`
- `reports/figures/shap_individual_transaction.png`

## 15. Limitations

- This is a portfolio/research system, not a production banking fraud engine.
- The model uses a memory-conscious subset of IEEE-CIS raw features rather than all 394 transaction fields.
- IEEE-CIS `TransactionDT` is relative time, not a real wall-clock timestamp.
- Official Kaggle test labels are unavailable locally; final metrics use a held-out chronological slice of labeled training data.
- Fraud costs are configurable project assumptions, not universal industry values.
- Online behavioral state would require a production feature store/event stream.

## 16. Possible production improvements

Feature store, real-time event ingestion, model registry, stronger hyperparameter search, delayed-label feedback, online drift alerting, distributed/streaming feature computation, and controlled retraining/champion-challenger workflows.
"""
    (report_dir / "model_report.md").write_text(text, encoding="utf-8")


def train_project(
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    model_names: list[str] | None = None,
    sample_size: int | None = None,
    generate_shap: bool = True,
) -> dict[str, Any]:
    """Train candidates, calibrate, select validation threshold, evaluate holdout and save artifacts."""
    configure_logging()
    settings.validate_split()
    output_dir = Path(output_dir or settings.model_dir)
    report_dir = Path(report_dir or settings.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics").mkdir(parents=True, exist_ok=True)

    logger.info("Loading REAL IEEE-CIS training data%s", " (debug sample)" if sample_size else "")
    df = load_training_data(sample_size=sample_size)

    validation_report = validate_training_frame(df)
    assert_training_quality(validation_report)
    if validation_report.duplicate_transaction_ids:
        logger.warning("Dropping %d duplicate TransactionID rows", validation_report.duplicate_transaction_ids)
        df = df.drop_duplicates("TransactionID", keep="first")

    # Missing target/time/amount are already treated as invalid by validation for this dataset.
    before_drop = len(df)
    df = df.dropna(subset=["isFraud", "TransactionDT", "TransactionAmt"])
    if len(df) != before_drop:
        logger.warning("Dropped %d rows missing required training values", before_drop - len(df))

    dataset_info: dict[str, Any] = {
        "mode": "ieee_cis_sample" if sample_size else "ieee_cis_full",
        "loaded_shape": [int(df.shape[0]), int(df.shape[1])],
        "fraud_count": validation_report.fraud_count,
        "fraud_rate": validation_report.fraud_rate,
        "duplicate_transaction_ids": validation_report.duplicate_transaction_ids,
        "missing_target": validation_report.missing_target,
        "missing_rate": validation_report.missing_rate,
        "infinite_numeric_values": validation_report.infinite_numeric_values,
        "selected_raw_columns": list(df.columns),
    }

    inventory = inspect_ieee_files(
        settings.raw_dir,
        hash_training_files=sample_size is None,
    )
    dataset_info["raw_file_inventory"] = inventory
    dataset_info["schema_consistency"] = _schema_consistency(settings.raw_dir)
    if sample_size is None:
        dataset_info["transaction_identity_relationship"] = _identity_relationship(
            settings.raw_dir, df["TransactionID"]
        )

    (report_dir / "metrics" / "data_validation.json").write_text(
        json.dumps(
            {
                "validation": validation_report.to_dict(),
                "dataset": dataset_info,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    train_df, val_df, test_df = temporal_split(
        df,
        settings.train_fraction,
        settings.validation_fraction,
        settings.test_fraction,
    )
    # The original full frame is no longer needed once shallow split references exist.
    del df
    gc.collect()

    cal_df, thresh_df = split_validation_for_calibration(val_df, 0.5)
    del val_df
    y_train = train_df.pop("isFraud").astype("int8", copy=False)
    y_cal = cal_df.pop("isFraud").astype("int8", copy=False)
    y_thresh = thresh_df.pop("isFraud").astype("int8", copy=False)
    y_test = test_df.pop("isFraud").astype("int8", copy=False)

    dataset_info["temporal_split_rows"] = {
        "train": int(len(train_df)),
        "calibration": int(len(cal_df)),
        "threshold_selection": int(len(thresh_df)),
        "local_holdout_test": int(len(test_df)),
    }
    dataset_info["temporal_split_fraud_rates"] = {
        "train": float(y_train.mean()),
        "calibration": float(y_cal.mean()),
        "threshold_selection": float(y_thresh.mean()),
        "local_holdout_test": float(y_test.mean()),
    }

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    factories = _model_factories(scale_pos_weight)
    if model_names:
        factories = {name: factory for name, factory in factories.items() if name in model_names}
    if not factories:
        raise ValueError("No valid model names selected")

    comparison: list[dict[str, Any]] = []
    best_score = -np.inf
    best_name: str | None = None
    best_threshold = 0.5
    best_validation: dict[str, Any] | None = None
    best_calibration_scores: dict[str, float] = {}
    temp_pipeline = output_dir / ".selected_pipeline.tmp.joblib"
    temp_calibrator = output_dir / ".selected_calibrator.tmp.joblib"

    for name, factory in factories.items():
        logger.info("Training %s", name)
        pipeline = build_pipeline(factory())
        started = time.perf_counter()
        pipeline.fit(train_df, y_train)
        raw_cal = pipeline.predict_proba(cal_df)[:, 1]
        calibrator, calibration_scores = choose_calibrator(raw_cal, y_cal.to_numpy())
        raw_thresh = pipeline.predict_proba(thresh_df)[:, 1]
        p_thresh = calibrator.predict(raw_thresh)
        best_cost = optimize_threshold(
            y_thresh.to_numpy(),
            p_thresh,
            settings.fp_cost,
            settings.fn_cost,
            settings.review_cost,
            settings.review_lower_bound,
        )
        val_metrics = classification_metrics(y_thresh.to_numpy(), p_thresh, best_cost.threshold)
        val_metrics["business_cost"] = best_cost.total_cost
        val_metrics["train_seconds"] = time.perf_counter() - started
        fitted_model = pipeline.named_steps["model"]
        n_iter = getattr(fitted_model, "n_iter_", None)
        if n_iter is not None:
            n_iter_value = int(np.max(np.asarray(n_iter)))
            max_iter_value = int(getattr(fitted_model, "max_iter", n_iter_value))
            converged = bool(n_iter_value < max_iter_value)
        else:
            n_iter_value = -1
            converged = True
        row = {
            "model": name,
            **val_metrics,
            "calibration_method": calibrator.method,
            "fit_iterations": n_iter_value,
            "converged_within_budget": converged,
        }
        comparison.append(row)
        save_evaluation_plots(
            y_thresh.to_numpy(),
            p_thresh,
            best_cost.threshold,
            report_dir / "figures",
            f"{name}_validation",
        )
        _log_mlflow(name, pipeline, val_metrics, best_cost.threshold, report_dir, "validation")

        score = float(val_metrics["pr_auc"])
        if score > best_score:
            best_score = score
            best_name = name
            best_threshold = float(best_cost.threshold)
            best_validation = dict(val_metrics)
            best_calibration_scores = dict(calibration_scores)
            joblib.dump(pipeline, temp_pipeline)
            joblib.dump(calibrator, temp_calibrator)

        del pipeline, calibrator, raw_cal, raw_thresh, p_thresh
        gc.collect()

    save_model_comparison(comparison, report_dir / "metrics" / "model_comparison.csv")
    if best_name is None or best_validation is None:
        raise RuntimeError("No candidate model completed successfully")

    pipeline: Pipeline = joblib.load(temp_pipeline)
    calibrator = joblib.load(temp_calibrator)
    threshold = best_threshold

    p_test = calibrator.predict(pipeline.predict_proba(test_df)[:, 1])
    test_metrics = classification_metrics(y_test.to_numpy(), p_test, threshold)
    test_cost = calculate_business_cost(
        y_test.to_numpy(),
        p_test,
        threshold,
        settings.fp_cost,
        settings.fn_cost,
        settings.review_cost,
        settings.review_lower_bound,
    )
    test_metrics["business_cost_at_validation_threshold"] = test_cost.total_cost
    save_evaluation_plots(
        y_test.to_numpy(),
        p_test,
        threshold,
        report_dir / "figures",
        f"{best_name}_test",
    )

    shap_status = {
        "summary": False,
        "feature_importance": False,
        "individual": False,
    }
    if generate_shap:
        shap_status["summary"] = save_shap_summary(
            pipeline, test_df, report_dir / "figures" / "shap_summary.png"
        )
        shap_status["feature_importance"] = save_shap_feature_importance(
            pipeline, test_df, report_dir / "figures" / "shap_feature_importance.png"
        )
        shap_status["individual"] = save_individual_explanation(
            pipeline, test_df, report_dir / "figures" / "shap_individual_transaction.png"
        )

    model_version = f"{best_name}_v1"
    training_timestamp = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "model_name": best_name,
        "model_version": model_version,
        "result_label": "LOCAL IEEE-CIS HOLDOUT RESULTS",
        "training_timestamp_utc": training_timestamp,
        "selection_metric": "validation_pr_auc",
        "calibration_method": calibrator.method,
        "calibration_selection_brier_scores": best_calibration_scores,
        "threshold": threshold,
        "validation_metrics": best_validation,
        "test_metrics": test_metrics,
        "dataset": dataset_info,
        "business_cost_config": {
            "fp_cost": settings.fp_cost,
            "fn_cost": settings.fn_cost,
            "review_cost": settings.review_cost,
            "review_lower_bound": settings.review_lower_bound,
        },
        "shap_generated": shap_status,
        "runtime": _runtime_versions(),
        "note": "All numeric metrics in this file were generated by this training run.",
    }
    feature_schema = {
        "raw_training_columns": [c for c in train_df.columns],
        "required_inference_columns": ["TransactionAmt"],
        "optional_inference_columns": [c for c in train_df.columns if c != "TransactionAmt"],
    }

    joblib.dump(pipeline, output_dir / "model.joblib")
    joblib.dump(pipeline.named_steps["preprocessor"], output_dir / "preprocessor.joblib")
    joblib.dump(calibrator, output_dir / "calibrator.joblib")
    (output_dir / "threshold.json").write_text(
        json.dumps(
            {
                "block_threshold": threshold,
                "review_lower_bound": settings.review_lower_bound,
                "selected_on": "validation_threshold_window",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
    )
    (report_dir / "metrics" / "selected_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    _write_model_report(report_dir, metadata, comparison)
    _log_mlflow(best_name, pipeline, test_metrics, threshold, report_dir, "local_holdout")

    for path in (temp_pipeline, temp_calibrator):
        path.unlink(missing_ok=True)

    logger.info("Selected %s with validation-only threshold %.6f", best_name, threshold)
    return metadata



def train_candidates_isolated(
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    model_names: list[str] | None = None,
    sample_size: int | None = None,
    generate_shap: bool = True,
) -> dict[str, Any]:
    """Train candidates in fresh Python processes, then select strictly by validation PR-AUC.

    This avoids retained sparse-matrix/solver memory between large candidate fits and is especially
    reliable on Windows. Each child executes the exact same train_project() logic for one model.
    """
    configure_logging()
    settings.validate_split()
    output_dir = Path(output_dir or settings.model_dir)
    report_dir = Path(report_dir or settings.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics").mkdir(parents=True, exist_ok=True)

    requested = model_names or ["logistic", "lightgbm", "xgboost"]
    candidate_rows: list[dict[str, Any]] = []
    candidate_metadata: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="fraud_candidates_", dir=str(settings.project_root)) as temp_name:
        temp_root = Path(temp_name)
        for name in requested:
            candidate_model_dir = temp_root / name / "models"
            candidate_report_dir = temp_root / name / "reports"
            candidate_model_dir.mkdir(parents=True, exist_ok=True)
            candidate_report_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "src.models.train",
                "--models",
                name,
                "--skip-shap",
                "--output-dir",
                str(candidate_model_dir),
                "--report-dir",
                str(candidate_report_dir),
            ]
            if sample_size is not None:
                cmd.extend(["--sample-size", str(sample_size)])
            env = os.environ.copy()
            env["FRAUD_TRAIN_WORKER"] = "1"
            logger.info("Training isolated candidate %s", name)
            child_stdout = temp_root / name / "worker_stdout.log"
            child_stderr = temp_root / name / "worker_stderr.log"
            with child_stdout.open("w", encoding="utf-8") as out_handle, child_stderr.open(
                "w", encoding="utf-8"
            ) as err_handle:
                completed = subprocess.run(
                    cmd,
                    cwd=settings.project_root,
                    env=env,
                    stdout=out_handle,
                    stderr=err_handle,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                stdout_tail = child_stdout.read_text(encoding="utf-8", errors="replace")[-4000:]
                stderr_tail = child_stderr.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(
                    f"Candidate {name} failed with exit code {completed.returncode}.\n"
                    f"STDOUT:\n{stdout_tail}\nSTDERR:\n{stderr_tail}"
                )
            metadata_path = candidate_model_dir / "model_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_metadata[name] = metadata
            comparison_path = candidate_report_dir / "metrics" / "model_comparison.csv"
            row_frame = pd.read_csv(comparison_path)
            if row_frame.empty:
                raise RuntimeError(f"Candidate {name} produced no comparison row")
            candidate_rows.append(row_frame.iloc[0].to_dict())

        # Selection uses validation PR-AUC only. Local holdout metrics are never consulted here.
        selected_name = max(candidate_rows, key=lambda row: float(row["pr_auc"]))["model"]
        selected_name = str(selected_name)
        selected_root = temp_root / selected_name
        selected_model_dir = selected_root / "models"
        selected_report_dir = selected_root / "reports"

        # Replace final artifacts atomically-ish at file level with the selected candidate.
        for artifact in (
            "model.joblib",
            "preprocessor.joblib",
            "calibrator.joblib",
            "threshold.json",
            "feature_schema.json",
            "model_metadata.json",
        ):
            shutil.copy2(selected_model_dir / artifact, output_dir / artifact)

        # Validation plots are useful for all candidates; held-out plots are kept only for selected.
        for name in requested:
            fig_dir = temp_root / name / "reports" / "figures"
            for path in fig_dir.glob(f"{name}_validation_*.png"):
                shutil.copy2(path, report_dir / "figures" / path.name)
        for path in (selected_report_dir / "figures").glob(f"{selected_name}_test_*.png"):
            shutil.copy2(path, report_dir / "figures" / path.name)

        comparison = pd.DataFrame(candidate_rows)
        comparison.to_csv(report_dir / "metrics" / "model_comparison.csv", index=False)
        for metric_file in ("selected_test_metrics.json", "data_validation.json"):
            shutil.copy2(
                selected_report_dir / "metrics" / metric_file,
                report_dir / "metrics" / metric_file,
            )

        metadata = candidate_metadata[selected_name]
        metadata["training_orchestration"] = "isolated_subprocess_per_candidate"
        metadata["candidate_models"] = requested

        if generate_shap:
            logger.info("Generating SHAP for selected model %s in parent process", selected_name)
            pipeline: Pipeline = joblib.load(output_dir / "model.joblib")
            raw = load_training_data(sample_size=sample_size)
            _train, _val, test_frame = temporal_split(
                raw,
                settings.train_fraction,
                settings.validation_fraction,
                settings.test_fraction,
            )
            test_frame = test_frame.copy(deep=False)
            test_frame.pop("isFraud")
            shap_status = {
                "summary": save_shap_summary(
                    pipeline, test_frame, report_dir / "figures" / "shap_summary.png"
                ),
                "feature_importance": save_shap_feature_importance(
                    pipeline, test_frame, report_dir / "figures" / "shap_feature_importance.png"
                ),
                "individual": save_individual_explanation(
                    pipeline, test_frame, report_dir / "figures" / "shap_individual_transaction.png"
                ),
            }
            metadata["shap_generated"] = shap_status
            del raw, _train, _val, test_frame, pipeline
            gc.collect()

        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
        )
        _write_model_report(report_dir, metadata, candidate_rows)
        logger.info("Isolated candidate selection chose %s", selected_name)
        return metadata

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the real IEEE-CIS fraud pipeline from data/raw/."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["logistic", "lightgbm", "xgboost"],
        help="Optional subset of candidate models",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional first-N-row real IEEE-CIS debug sample; omit for full data",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP plot generation")
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--report-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample_size is not None and args.sample_size < 1000:
        parser.error("--sample-size must be at least 1000")

    requested = args.models or ["logistic", "lightgbm", "xgboost"]
    worker = os.getenv("FRAUD_TRAIN_WORKER", "0") == "1"
    if not worker and len(requested) > 1:
        metadata = train_candidates_isolated(
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            model_names=requested,
            sample_size=args.sample_size,
            generate_shap=not args.skip_shap,
        )
    else:
        metadata = train_project(
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            model_names=requested,
            sample_size=args.sample_size,
            generate_shap=not args.skip_shap,
        )
    print(json.dumps(metadata, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
