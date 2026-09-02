# Verification Report

Verification date: 2026-08-26

This report records only checks actually executed in the available environment. The verifier is Linux with Python 3.13.5; the target Windows 11 / Python 3.14.x environment remains a separate local verification step.

## Real IEEE-CIS dataset used

The supplied official IEEE-CIS files were extracted outside the repository and used without modifying or packaging them:

- `train_transaction.csv`: 590,540 rows × 394 columns
- `train_identity.csv`: 144,233 rows × 41 columns
- `test_transaction.csv`: 506,691 rows × 393 columns
- `test_identity.csv`: 141,907 rows × 41 columns
- labeled fraud rows: 20,663 (3.499%)
- missing `isFraud`: 0
- duplicate labeled `TransactionID`: 0
- identity left join preserved all 590,540 transaction rows
- official test transaction data has no `isFraud`

The training loader used 25 selected/merged raw columns with chunked parsing and early dtype optimization instead of loading every raw field repeatedly into memory.

## Fresh full-data training executed

The final cleaned code executed the full command against the real external IEEE-CIS data:

```text
python -m src.models.train
```

The run trained Logistic Regression, LightGBM, and XGBoost in isolated candidate processes, selected strictly by validation PR-AUC, calibrated the selected model, optimized the decision threshold on the validation threshold window, evaluated the frozen model on the chronological local holdout, saved artifacts, and generated SHAP outputs.

Verified validation model comparison:

| Model | Validation PR-AUC | Local holdout PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.1475608 | 0.14225 |
| LightGBM | **0.3296737** | 0.3767068 |
| XGBoost | 0.3138159 | **0.38462** |

Selected model: **LightGBM** because validation PR-AUC is the selection criterion. The better XGBoost local-holdout score was not used to revise the winner.

Selected LightGBM — **LOCAL IEEE-CIS HOLDOUT RESULTS**:

- PR-AUC: 0.3767067716
- ROC-AUC: 0.8496966186
- precision: 0.7531194296
- recall/fraud capture: 0.2740836847
- F1: 0.4019024970
- false-positive rate: 0.0032398419
- Brier score: 0.0270360078
- frozen threshold: 0.51
- confusion matrix: TN=85,221, FP=277, FN=2,238, TP=845
- Recall @ 1% FPR: 0.3295491404
- Precision @ 1% FPR: 0.5506775068

These are not Kaggle leaderboard scores.

## SHAP

The final selected real LightGBM run generated all required files successfully:

```text
reports/figures/shap_summary.png
reports/figures/shap_feature_importance.png
reports/figures/shap_individual_transaction.png
```

The compatibility code handles list/tuple, 2-D and 3-D binary-class SHAP outputs and sparse matrices, densifying only the bounded explanation sample.

## Import / compile verification

```text
python -m compileall -q src api app tests
→ success

python -c "from src.data.loader import load_training_data; print('IMPORT OK')"
→ IMPORT OK
```

## Prediction CLI

Executed against the freshly generated real LightGBM artifacts:

```text
python -m src.models.predict --amount 125.50 --product W --card1 12345 --card4 visa --device desktop
→ successful model-derived JSON prediction
```

Observed verification probability: `0.0095186576`; threshold: `0.51`; decision: `APPROVE`. This value came from the actual trained artifacts and is not hard-coded.

## Tests

Executed with the real IEEE-CIS files available externally and the real trained artifacts present:

```text
pytest -v
→ 9 passed, 5 warnings
```

The warnings are sklearn/SHAP compatibility warnings; no test failed. Tests do not package or generate a fake fraud dataset. Resource-dependent tests explicitly skip on a public checkout until the private IEEE-CIS CSVs and locally generated model artifacts are available.

## FastAPI

An actual Uvicorn process was started with the freshly generated real artifacts and PostgreSQL disabled. Verified:

```text
GET  /health        → HTTP 200
GET  /docs          → HTTP 200
POST /predict       → HTTP 200
POST /predict/batch → HTTP 200
negative amount     → HTTP 422
```

Verified health response:

```json
{"status":"ok","model_loaded":true,"database":"disabled (optional)"}
```

## Environment/tool status

```text
python --version
→ Python 3.13.5
```

`python -m pip check` reports an unrelated shared-environment moviepy/Pillow conflict. That package pair is not declared by this repository. A clean Windows/Python 3.14 virtual environment could not be created and installed inside this verifier, so target-runtime dependency installation remains not verified here.

The current verifier does not provide the Streamlit executable, MLflow executable, Docker executable, or a live PostgreSQL service. Those components are therefore environment-blocked rather than claimed as tested.

## Final status

| Component | Status | Evidence / limitation |
|---|---|---|
| Real IEEE-CIS loading | VERIFIED | official full files inspected and used |
| Memory-safe pipeline | VERIFIED | chunked selected-column full-data path completed |
| Feature engineering | VERIFIED | full training ran through reusable feature pipeline |
| Logistic Regression | VERIFIED | full candidate completed |
| LightGBM | VERIFIED | full candidate completed and selected by validation PR-AUC |
| XGBoost | VERIFIED | full candidate completed |
| Calibration | VERIFIED | sigmoid selected on calibration window |
| Threshold optimization | VERIFIED | threshold 0.51 selected on validation-only threshold window |
| SHAP | VERIFIED | all three required figures generated |
| Prediction CLI | VERIFIED | real-artifact command executed |
| FastAPI | VERIFIED | actual Uvicorn endpoints tested |
| Batch API | VERIFIED | actual batch request returned HTTP 200 |
| Monitoring core | VERIFIED | PSI/KS tests passed |
| Streamlit | BLOCKED BY ENVIRONMENT | Streamlit executable unavailable in verifier |
| MLflow UI/runtime | BLOCKED BY ENVIRONMENT | MLflow executable unavailable in verifier |
| PostgreSQL live service | BLOCKED BY ENVIRONMENT | no live PostgreSQL service available |
| Evidently runtime | BLOCKED BY ENVIRONMENT | optional dependency not installed in verifier |
| Docker | BLOCKED BY ENVIRONMENT | Docker executable unavailable |
| Tests | VERIFIED | 9/9 passed with real local resources |
| Python 3.14 / Windows 11 | NOT VERIFIED | verifier is Linux/Python 3.13.5 |
| GitHub cleanup | VERIFIED | final staging audit excludes raw data/secrets/model binaries/caches |
| Final ZIP | VERIFIED after final packaging pass | archive integrity and fresh-extract checks documented in final response |

## GitHub packaging policy

The final GitHub-ready ZIP deliberately excludes:

- raw IEEE-CIS CSVs and dataset ZIPs
- `.env`
- `.venv/`
- generated `models/*.joblib` and model metadata artifacts
- caches, logs, temporary candidate folders, and Python bytecode
- MLflow runtime directories

Verified metrics and SHAP figures remain under `reports/` as portfolio evidence. Generate local model binaries after checkout with:

```powershell
python -m src.models.train
```
