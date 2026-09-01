from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
import json
from datetime import datetime

class CaseCreate(BaseModel):
    inference_id: str
    calibrated_probability: float
    risk_level: str
    reason_codes: List[str]
    model_version: str
    transaction_reference: Optional[str] = None
    priority: str = "LOW"

class CaseUpdate(BaseModel):
    version: int  # Required for optimistic locking
    status: Optional[str] = None
    priority: Optional[str] = None
    analyst_note: Optional[str] = None
    actor: str = "system"  # Client-provided audit metadata

class CaseResponse(BaseModel):
    case_id: str
    inference_id: str
    transaction_reference: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: str
    priority: str
    calibrated_probability: float
    risk_level: str
    reason_codes: List[str]
    model_version: str
    analyst_note: Optional[str]
    version: int

    @field_validator('reason_codes', mode='before')
    def parse_reason_codes(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return []
        return v

    class Config:
        from_attributes = True

class CaseEventResponse(BaseModel):
    event_id: str
    case_id: str
    previous_status: Optional[str]
    new_status: str
    actor: str
    note: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

class PaginatedCases(BaseModel):
    items: List[CaseResponse]
    total: int
    page: int
    size: int
