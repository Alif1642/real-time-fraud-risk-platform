# API

Run: `uvicorn api.main:app --reload`

Open Swagger UI at `http://localhost:8000/docs`.

## Endpoints
- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /model/info`
- `GET /monitoring/status`

## Example request
```json
{"TransactionAmt":125.50,"ProductCD":"W","card1":12345,"card2":555,"card3":150,"card4":"visa"}
```

The response contains calibrated fraud probability, LOW/MEDIUM/HIGH risk level, APPROVE/REVIEW/BLOCK decision, validation-selected threshold, model version, reason codes and UTC prediction timestamp.

## Security notes
Input is validated with Pydantic; CORS origins are environment-configured; no API keys are embedded; batch size is capped; raw transaction bodies are not written to application logs. Add authentication, TLS termination, rate limiting and a secrets manager for real deployments.
