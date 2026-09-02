from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

metrics_router = APIRouter()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "fraudshield_predictions_total",
    "Total number of predictions made",
    ["risk_level"]
)

PREDICTION_LATENCY = Histogram(
    "fraudshield_prediction_duration_seconds",
    "Prediction latency in seconds",
    []
)

@metrics_router.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
