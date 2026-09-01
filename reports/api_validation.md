# API Validation & Benchmarks

This report summarizes the testing and performance evaluation of the Phase 8 Production Inference API.

## Security & Validation Testing

The `FastAPI` layer has been validated with `TestClient` across several dimensions (11 passed, 0 failed):
1. **Model Loading:** Verifies that the predictor is initialized precisely once during `lifespan` startup. Per-request artifact loading was eliminated.
2. **Schema Compliance (Pydantic):** 
   - Requests with missing fields, incorrect data types, or out-of-bound values (`amount < 0`) are successfully rejected with `422 Unprocessable Entity`.
   - Feature leakage prevention: Submission of forbidden fields (`isFraud`, `newbalanceOrig`, etc.) results in immediate rejection.
3. **Authentication:** Endpoints properly enforce `X-API-Key` (403 Forbidden) if the key is misconfigured or missing.
4. **Limits:** 
   - Batch limits (default 1000) are enforced, returning `413 Payload Too Large`.
   - CSV uploads enforce both file size (5MB) and row limits (10,000).

## API Benchmarks

The benchmark script (`scripts/benchmark_api.py`) executed local load testing against a running Uvicorn server (`fraudshield serve --workers 1`). 

**Results:**
- **Health Check Latency:** ~39.59 ms
- **Single Predict (No Explain):** p50=31.89 ms, p95=40.62 ms
- **Single Explain (With IG):** p50=47.01 ms, p95=54.04 ms
- **Batch Throughput (Size 100):** ~1125 ms per batch (Approx. 0.88 HTTP requests/sec | 88.8 transactions/sec)

**Concurrent Throughput (Single Predict without Explain):**
- **1 Worker:** Median Latency 33.18 ms | 27.49 req/sec
- **4 Workers:** Median Latency 89.55 ms | 37.50 req/sec
- **8 Workers:** Median Latency 151.02 ms | 41.73 req/sec
- **16 Workers:** Median Latency 261.77 ms | 34.44 req/sec

> [!NOTE]
> The direct predictor latency (Phase 7) was ~7.60 ms. The API overhead (Pydantic validation, serialization, HTTP parsing) adds approximately 25 ms per request locally. Throughput peaks around 40 req/sec on a single Uvicorn instance for concurrent traffic.

## Worker Scaling & Concurrency
Each Uvicorn worker loads a separate model bundle into memory. The Integrated Gradients explanation semaphore is instantiated *per process*, meaning total explanation concurrency equals the number of workers multiplied by the per-worker limit.
Do not recommend increasing workers without first benchmarking CPU utilization, memory use, latency overhead, and PyTorch thread oversubscription.

## Upstream Fields Trust
Fields such as `orig_account_type` and `dest_account_type` are designed to be trusted upstream fields. In a real deployment, these fields must be verified or securely derived by the upstream account system. They must never be treated as trustworthy when supplied directly by an untrusted public client.

## Security & API Keys
API keys are used as a basic demo or service-to-service authentication mechanism, not for complete banking security. 
- Keys are read strictly from environment variables (e.g. `FRAUDSHIELD_API_KEY`) or secret storage.
- Keys are never logged in the application logs.
- Key comparison uses `hmac.compare_digest` to mitigate timing attacks.
- CORS is disabled by default, and error handlers return safe JSON, ensuring stack traces, artifact paths, or raw payloads are never exposed to the client.
