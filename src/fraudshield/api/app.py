from fastapi import FastAPI
from contextlib import asynccontextmanager

from fraudshield.api.routers import health, prediction, model_info, cases
from fraudshield.api.middleware import add_request_id_middleware
from fraudshield.api.error_handlers import register_error_handlers
from fraudshield.api.dependencies import init_predictor
from fraudshield.api.db import init_db
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    # Initialize DB
    init_db()
    # Initialize the predictor once
    try:
        predictor = init_predictor()
        app.state.predictor = predictor
        app.state.is_ready = True
        logger.info("FraudPredictor successfully initialized and loaded into app state.")
    except Exception as e:
        logger.error(f"Failed to initialize FraudPredictor: {e}")
        app.state.predictor = None
        app.state.is_ready = False
        
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    app.state.predictor = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="FraudShield ML Inference API",
        version="1.0.0",
        description=(
            "Production-oriented real-time inference API for FraudShield ML.\n\n"
            "**Known Limitations**:\n"
            "- Trained entirely on PaySim, a synthetic dataset.\n"
            "- The probability is an empirically calibrated estimate, not a ground truth.\n"
            "- Feature attributions (explanations) explain the raw logit, not the final probability."
        ),
        lifespan=lifespan
    )
    
    # Middleware
    add_request_id_middleware(app)
    
    # Error Handlers
    register_error_handlers(app)
    
    # Routers
    app.include_router(health.router)
    app.include_router(model_info.router, prefix="/api/v1")
    app.include_router(prediction.router, prefix="/api/v1")
    app.include_router(cases.router, prefix="/api/v1")
    
    return app
