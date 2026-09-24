Docker Run Guide
The Docker setup runs the FastAPI service and Streamlit dashboard as separate containers.
Prerequisite
Install and start Docker Desktop. Real transaction scoring also requires these locally generated files in models/:
    model.joblib
calibrator.joblib
threshold.json
model_metadata.json
feature_schema.json

Without those artifacts, the containers can start but /health reports a degraded model state and transaction prediction is unavailable.
Build and start
From the repository root:
    docker compose up --build

Open:
Streamlit: http://localhost:8501
FastAPI: http://localhost:8000
API documentation: http://localhost:8000/docs
Health check: http://localhost:8000/health
The dashboard uses the internal Compose address http://api:8000; do not replace it with localhost inside compose.yaml.
Stop
    docker compose down

Rebuild after dependency changes
    docker compose build --no-cache
docker compose up

View logs
    docker compose logs -f api
docker compose logs -f dashboard

Important data rule
Raw IEEE-CIS CSV files and trained binary artifacts remain outside the Docker image. Compose mounts the local models/ and reports/ directories at runtime. Do not push raw transaction data, .env, credentials, or private customer data to GitHub.