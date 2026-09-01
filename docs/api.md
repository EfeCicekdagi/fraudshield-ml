# FraudShield ML - Production Inference API

This document describes the FastAPI inference service for the FraudShield ML project.

## Architecture & Security
- **Version-Independent Bundle**: The API loads the secure pure-NumPy/PyTorch bundle from Phase 7.
- **FastAPI Lifespan**: The `FraudPredictor` is initialized once during startup and shared via `app.state`. Per-request artifact loading is eliminated.
- **Concurrency**: CPU-bound inference logic runs via `run_in_threadpool`. Integrated Gradients explainability is protected by a per-process `asyncio.Semaphore` to prevent CPU/memory exhaustion.
- **Worker Scaling**: Each Uvicorn worker loads a separate model bundle into memory. Total explanation concurrency is workers multiplied by the per-worker limit. Do not increase workers without benchmarking CPU, memory, latency, and PyTorch thread oversubscription.
- **Strict Validation**: The Pydantic schema strictly forbids non-causal features like `isFraud` or internal account identifier mapping dependencies.
- **Upstream Fields**: Fields like `orig_account_type` and `dest_account_type` are trusted upstream fields. They must be verified/derived by the upstream account system and never trusted if supplied directly by a public client.
- **Security & API Keys**: API keys are for demo/service authentication, not complete banking security. They are read from environment variables (e.g. `FRAUDSHIELD_API_KEY`), never logged, and validated using `hmac.compare_digest`. CORS is disabled by default, and stack traces/raw payloads are never exposed.

## Endpoints

### `GET /health/live`
Returns 200 OK if the process is running.

### `GET /health/ready`
Returns 200 OK if the model bundle was verified and loaded successfully. 503 if corrupt or missing.

### `GET /api/v1/model-info`
Returns configuration, active thresholds, and known limitations.

### `POST /api/v1/predict`
Score a single transaction.
- **Query Params:** `explain=false` (Set to true to force Integrated Gradients).

### `POST /api/v1/explain`
Score a single transaction with explanations forcefully enabled.

### `POST /api/v1/predict/batch`
Score a batch of transactions (JSON list).
- **Limit:** 1000 transactions by default.

### `POST /api/v1/predict/file`
Upload a CSV file of transactions for scoring.
- **Limits:** Max 5MB, Max 10,000 rows.

## Usage Example

```bash
# Start the server
fraudshield serve --host 127.0.0.1 --port 8000

# Predict
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 1,
       "type": "PAYMENT",
       "amount": 100,
       "oldbalanceOrg": 1000,
       "orig_account_type": "C",
       "dest_account_type": "M"
     }'
```
