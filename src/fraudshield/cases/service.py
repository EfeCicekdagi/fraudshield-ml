from fastapi import HTTPException
from sqlalchemy.orm import Session
from fraudshield.cases.repository import CaseRepository
from fraudshield.cases.schemas import CaseCreate, CaseUpdate
from fraudshield.cases.models import Case

ALLOWED_TRANSITIONS = {
    "NEW": ["UNDER_REVIEW", "CLOSED"],
    "UNDER_REVIEW": ["CONFIRMED_FRAUD", "FALSE_POSITIVE", "NEW"],
    "CONFIRMED_FRAUD": ["UNDER_REVIEW"],
    "FALSE_POSITIVE": ["UNDER_REVIEW"],
    "CLOSED": ["NEW"]
}

class CaseService:
    def __init__(self, session: Session):
        self.session = session
        self.repo = CaseRepository(session)

    def create_case(self, data: CaseCreate) -> Case:
        # Check idempotency via inference_id
        existing = self.repo.get_by_inference_id(data.inference_id)
        if existing:
            return existing
            
        case = self.repo.create(data)
        self.session.commit()
        return case

    def update_case(self, case_id: str, update_data: CaseUpdate) -> Case:
        case = self.repo.get_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
            
        current_status = case.status
        new_status = update_data.status or current_status
        new_priority = update_data.priority or case.priority
        new_note = update_data.analyst_note or case.analyst_note
        
        # Status transition validation
        if current_status != new_status:
            if new_status not in ALLOWED_TRANSITIONS.get(current_status, []):
                raise HTTPException(status_code=400, detail=f"Invalid status transition from {current_status} to {new_status}")

        success = self.repo.optimistic_update(
            case_id=case_id,
            expected_version=update_data.version,
            new_status=new_status,
            new_priority=new_priority,
            new_note=new_note,
            actor=update_data.actor,
            current_status=current_status
        )
        
        if not success:
            self.session.rollback()
            raise HTTPException(status_code=409, detail="Stale case version. The case was modified by another user.")
            
        self.session.commit()
        
        # Return updated
        return self.repo.get_by_id(case_id)
