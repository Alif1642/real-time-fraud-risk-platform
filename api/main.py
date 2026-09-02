"""FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import batch, health, monitoring, predict
from src.config import settings
from src.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Real-Time Transaction Fraud Risk & Monitoring Platform",
    version="0.1.0",
    description="Educational/portfolio fraud risk API. Not a production banking fraud system.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": app.title, "docs": "/docs", "status": "ok"}


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(monitoring.router)
