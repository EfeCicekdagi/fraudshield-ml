from fastapi import APIRouter, Request, HTTPException

router = APIRouter(tags=["Health"])

@router.get("/health/live")
async def liveness():
    """
    Check if the application process is running.
    """
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness(request: Request):
    """
    Check if the Model Inference Environment is loaded and ready to serve requests.
    """
    if request.app.state.is_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized or bundle is corrupt.")
