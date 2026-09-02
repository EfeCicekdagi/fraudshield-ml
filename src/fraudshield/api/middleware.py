import time
import uuid
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
        except Exception as e:
            # We don't handle the error here, let error_handlers do it,
            # but we can log the failure if needed.
            raise e
        finally:
            process_time_ms = (time.time() - start_time) * 1000
            
            # Extract basic info for logging without logging sensitive payload
            status_code = response.status_code if 'response' in locals() else 500
            
            # Do not log raw transaction or API keys
            logger.info(
                f"method={request.method} path={request.url.path} "
                f"status={status_code} latency_ms={process_time_ms:.2f} "
                f"request_id={request_id}"
            )
            
            try:
                from fraudshield.api.metrics import REQUEST_COUNT, REQUEST_LATENCY
                import re
                
                # Normalize UUIDs in paths to prevent cardinality explosion
                route_path = request.url.path
                route_path = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '{id}', route_path)
                
                REQUEST_COUNT.labels(method=request.method, endpoint=route_path, http_status=status_code).inc()
                REQUEST_LATENCY.labels(method=request.method, endpoint=route_path).observe(process_time_ms / 1000.0)
            except Exception:
                pass

            
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response

def add_request_id_middleware(app: FastAPI):
    app.add_middleware(RequestIDMiddleware)
