# Windows 11 / PowerShell Runbook

This repository uses the **real IEEE-CIS Fraud Detection dataset only**. No sample or synthetic fraud dataset is bundled.

## 1. Prepare the project

Extract the ZIP so the folder is:

```text
C:\path\to\real-time-fraud-risk-platform\
```

Place the official files in:

```text
data\raw\train_transaction.csv
data\raw\train_identity.csv
data\raw\test_transaction.csv
data\raw\test_identity.csv
```

Do not rename the files and do not add `isFraud` to the official test set.

## 2. First terminal — environment, training, prediction, tests

```powershell
cd "C:\path\to\real-time-fraud-risk-platform"
python --version
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
Copy-Item .env.example .env
python -c "from src.data.loader import load_training_data; print('IMPORT OK')"
python -m src.models.train
python -m src.models.predict
pytest -v
```

Expected milestones:

- import check prints `IMPORT OK`
- training loads the real IEEE-CIS CSVs, trains the three candidates, and selects using validation PR-AUC
- `models/` receives generated model/calibration/schema/threshold metadata files
- `reports/` receives fresh metrics, evaluation plots, and SHAP figures
- prediction prints model-derived JSON
- tests complete; resource-dependent tests should run because raw data and model artifacts now exist

Keep this terminal available for troubleshooting, but Uvicorn does not need to run here.

## 3. Second terminal — FastAPI

```powershell
cd "C:\path\to\real-time-fraud-risk-platform"
.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Keep this terminal running. Open:

```text
http://localhost:8000/
http://localhost:8000/health
http://localhost:8000/docs
```

## 4. Third terminal — Streamlit

```powershell
cd "C:\path\to\real-time-fraud-risk-platform"
.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Keep the FastAPI terminal running because the transaction-scoring page calls the API.

## 5. Fourth terminal — MLflow (optional)

```powershell
cd "C:\path\to\real-time-fraud-risk-platform"
.venv\Scripts\Activate.ps1
mlflow ui
```

Open:

```text
http://localhost:5000
```

## 6. PostgreSQL (optional)

```powershell
pip install -r requirements-postgres.txt
docker compose up -d postgres
```

Then edit `.env`:

```text
ENABLE_DATABASE=true
DATABASE_URL=postgresql+psycopg://fraud:change_me_local@localhost:5432/fraud
```

PostgreSQL is not required for training or basic local scoring.

## 7. Evidently (optional)

```powershell
pip install -r requirements-evidently.txt
```

If Evidently does not support your exact Python 3.14 build, keep:

```text
ENABLE_EVIDENTLY=false
```

Core PSI/KS monitoring remains available.

## 8. Docker stack (optional)

Train locally first so `models/` contains the generated artifacts, then:

```powershell
docker compose config
docker compose up --build
```

## 9. Common fixes

### `ModuleNotFoundError: No module named 'src'`

You are probably not in the repository root. Verify:

```powershell
Get-Location
python -c "import src; print(src.__file__)"
```

Run all module commands from the folder that directly contains `src`, `api`, `app`, and `README.md`.

### Raw data missing

If training says a CSV is missing, verify:

```powershell
Get-ChildItem data\raw
```

The four official IEEE-CIS filenames must match exactly.

### Artifact loading error

Regenerate artifacts inside the active virtual environment:

```powershell
python -m src.models.train
python -m src.models.predict
```

### PowerShell blocks venv activation

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Streamlit cannot reach FastAPI

Make sure the Uvicorn terminal is still running and `.env` contains:

```text
API_BASE_URL=http://localhost:8000
```

## 10. GitHub push

```powershell
git init
git add .
git status
git commit -m "Initial release: IEEE-CIS real-time fraud risk platform"
git branch -M main
git remote -v
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

If `origin` already exists, do not run `git remote add origin` again; inspect `git remote -v` and update it only if required.

Before committing, confirm `.env`, `.venv`, `data/raw/*.csv`, and generated model binaries are not staged.
