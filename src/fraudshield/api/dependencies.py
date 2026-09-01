import os
import hmac
from fastapi import Request, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from fraudshield.inference.predictor import FraudPredictor
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    expected_api_key = os.environ.get("FRAUDSHIELD_API_KEY")
    if expected_api_key:
        if api_key_header is not None and hmac.compare_digest(api_key_header, expected_api_key):
            return api_key_header
        else:
            raise HTTPException(
                status_code=403, detail="Could not validate API KEY"
            )
    return None

def init_predictor() -> FraudPredictor:
    """Initialize the FraudPredictor. Called once during lifespan."""
    # This will load the bundle and verify checksums
    predictor = FraudPredictor()
    return predictor

def get_predictor(request: Request) -> FraudPredictor:
    """Dependency to get the predictor from app state."""
    predictor = request.app.state.predictor
    if not predictor or not request.app.state.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return predictor
