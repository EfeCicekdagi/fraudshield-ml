from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

def register_error_handlers(app: FastAPI):
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        
        errors = exc.errors()
        details = []
        for error in errors:
            loc = " -> ".join([str(x) for x in error.get("loc", [])])
            details.append(f"{loc}: {error.get('msg')}")
            
        logger.warning(f"Request validation failed for {request_id}: {details}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "request_id": request_id,
                    "details": details
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Map some common status codes to standard error codes
        code = "HTTP_ERROR"
        if exc.status_code == 503:
            code = "MODEL_NOT_READY"
        elif exc.status_code == 413:
            code = "FILE_TOO_LARGE"
        elif exc.status_code == 415:
            code = "UNSUPPORTED_MEDIA_TYPE"
        elif exc.status_code == 403:
            code = "UNAUTHORIZED"
            
        logger.warning(f"HTTP exception {exc.status_code} for {request_id}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "request_id": request_id,
                    "details": []
                }
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.error(f"Unhandled exception for {request_id}: {type(exc).__name__} - {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": "An internal error occurred during prediction.",
                    "request_id": request_id,
                    "details": []
                }
            }
        )
