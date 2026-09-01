from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from sqlalchemy.orm import Session

from fraudshield.api.db import get_db
from fraudshield.api.dependencies import get_api_key
from fraudshield.cases.schemas import CaseCreate, CaseUpdate, CaseResponse, CaseEventResponse, PaginatedCases
from fraudshield.cases.service import CaseService
from fraudshield.cases.repository import CaseRepository

router = APIRouter(tags=["Cases"], dependencies=[Depends(get_api_key)])

@router.post("/cases", response_model=CaseResponse)
def create_case(data: CaseCreate, db: Session = Depends(get_db)):
    """Idempotently create a new case."""
    service = CaseService(db)
    return service.create_case(data)

@router.get("/cases", response_model=PaginatedCases)
def list_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_probability: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """List cases with pagination and filters."""
    repo = CaseRepository(db)
    items, total = repo.get_list(skip=skip, limit=limit, status=status, risk_level=risk_level, min_probability=min_probability)
    return PaginatedCases(items=items, total=total, page=skip // limit + 1, size=limit)

@router.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, db: Session = Depends(get_db)):
    repo = CaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case

@router.patch("/cases/{case_id}", response_model=CaseResponse)
def update_case(case_id: str, update_data: CaseUpdate, db: Session = Depends(get_db)):
    service = CaseService(db)
    return service.update_case(case_id, update_data)

@router.get("/cases/{case_id}/history", response_model=List[CaseEventResponse])
def get_case_history(case_id: str, db: Session = Depends(get_db)):
    repo = CaseRepository(db)
    case = repo.get_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return repo.get_events(case_id)

@router.get("/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    repo = CaseRepository(db)
    return repo.get_summary_metrics()
