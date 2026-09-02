# Observability

FraudShield exposes Prometheus metrics for the API and Inference logic.

## Key Metrics
- `http_requests_total`: Total API requests grouped by method, endpoint, and status.
- `http_request_duration_seconds`: API latency histogram.
- `fraudshield_predictions_total`: Total predictions grouped by risk level.
- `fraudshield_prediction_duration_seconds`: Inference latency histogram.

## Endpoint & Security
The metrics are exposed at the `/metrics` endpoint. 
**Important**: The `/metrics` endpoint does not enforce authentication by default to allow easy scraping by Prometheus. Ensure that your network or reverse proxy (like NGINX) restricts access to this endpoint to internal cluster traffic only.

## Architecture Limitations
Currently, the Prometheus metrics collection assumes **one API worker** per container (the default uvicorn setup in this project). If you configure `uvicorn` or `gunicorn` with multiple worker processes, Prometheus multiprocess mode must be configured (setting `PROMETHEUS_MULTIPROC_DIR`), otherwise metrics will be overwritten or missing as workers bind to the same endpoint without shared state. The current metrics contract does not enable multiprocess mode.

## Grafana Dashboards
Grafana is pre-provisioned with a datasource for Prometheus and a default `FraudShield Observability` dashboard showing request rates and prediction metrics.
