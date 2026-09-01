import os
import io
import pandas as pd
from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.concurrency import run_in_threadpool
import asyncio

from fraudshield.inference.schemas import TransactionRequest, BatchTransactionRequest, InferenceResponse
from fraudshield.inference.predictor import FraudPredictor
from fraudshield.api.dependencies import get_predictor, get_api_key

router = APIRouter(tags=["Prediction"])

MAX_BATCH_SIZE = int(os.environ.get("FRAUDSHIELD_MAX_BATCH_SIZE", 1000))
MAX_FILE_ROWS = int(os.environ.get("FRAUDSHIELD_MAX_FILE_ROWS", 10000))
MAX_FILE_SIZE_MB = int(os.environ.get("FRAUDSHIELD_MAX_FILE_SIZE_MB", 5))

# Concurrency limits for explanation to avoid blowing up memory/CPU
MAX_CONCURRENT_EXPLANATIONS = int(os.environ.get("FRAUDSHIELD_MAX_CONCURRENT_EXPLANATIONS", 4))
explain_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXPLANATIONS)

@router.post("/predict", response_model=InferenceResponse, dependencies=[Depends(get_api_key)])
async def predict_single(
    request: TransactionRequest,
    explain: bool = Query(False, description="Enable explanations via Integrated Gradients"),
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score a single transaction. Explainability is disabled by default for performance.
    """
    if explain:
        async with explain_semaphore:
            return await run_in_threadpool(predictor.predict_single, request, explain=True)
    else:
        return await run_in_threadpool(predictor.predict_single, request, explain=False)

@router.post("/explain", response_model=InferenceResponse, dependencies=[Depends(get_api_key)])
async def explain_single(
    request: TransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score a single transaction and force explainability using Integrated Gradients.
    Returns reason codes and top feature contributors.
    """
    async with explain_semaphore:
        return await run_in_threadpool(predictor.predict_single, request, explain=True)

@router.post("/predict/batch", response_model=List[Union[InferenceResponse, dict]], dependencies=[Depends(get_api_key)])
async def predict_batch(
    request: BatchTransactionRequest,
    explain: bool = Query(False, description="Enable explanations for all transactions in batch"),
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score a batch of transactions via JSON list. 
    Limited by MAX_BATCH_SIZE to prevent unbounded requests.
    """
    if len(request.transactions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"Batch size {len(request.transactions)} exceeds maximum allowed ({MAX_BATCH_SIZE})"
        )
        
    if explain:
        async with explain_semaphore:
            return await run_in_threadpool(predictor.predict_batch, request, explain=True)
    else:
        return await run_in_threadpool(predictor.predict_batch, request, explain=False)

@router.post("/predict/file", response_model=List[Union[InferenceResponse, dict]], dependencies=[Depends(get_api_key)])
async def predict_file(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Enable explanations for all rows"),
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score transactions uploaded via CSV file.
    Only allows 'text/csv' up to a configured file size and row limit.
    """
    if file.content_type not in ["text/csv", "application/vnd.ms-excel"]:
        raise HTTPException(status_code=415, detail="Only CSV files are supported.")
        
    # Read file content safely
    content = await file.read()
    
    # Check size limit
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")
        
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail="Invalid CSV format.")
        
    if len(df) > MAX_FILE_ROWS:
        raise HTTPException(status_code=413, detail=f"File contains {len(df)} rows, exceeding the limit of {MAX_FILE_ROWS}.")
        
    # Convert DF to list of TransactionRequest
    # We rely on Pydantic to do validation!
    transactions = []
    for idx, row in df.iterrows():
        try:
            req = TransactionRequest(**row.to_dict())
            transactions.append(req)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Validation error on row {idx}: {e}")
            
    batch_req = BatchTransactionRequest(transactions=transactions)
    
    if explain:
        async with explain_semaphore:
            return await run_in_threadpool(predictor.predict_batch, batch_req, explain=True)
    else:
        return await run_in_threadpool(predictor.predict_batch, batch_req, explain=False)
