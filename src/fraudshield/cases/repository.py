import json
from datetime import datetime
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc, update

from fraudshield.cases.models import Case, CaseEvent
from fraudshield.cases.schemas import CaseCreate

class CaseRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, case_id: str) -> Optional[Case]:
        return self.session.query(Case).filter(Case.case_id == case_id).first()

    def get_by_inference_id(self, inference_id: str) -> Optional[Case]:
        return self.session.query(Case).filter(Case.inference_id == inference_id).first()

    def create(self, data: CaseCreate) -> Case:
        db_case = Case(
            inference_id=data.inference_id,
            calibrated_probability=data.calibrated_probability,
            risk_level=data.risk_level,
            reason_codes=json.dumps(data.reason_codes),
            model_version=data.model_version,
            transaction_reference=data.transaction_reference,
            priority=data.priority,
            status="NEW"
        )
        self.session.add(db_case)
        self.session.flush() # flush to get case_id generated
        
        # Log initial event
        event = CaseEvent(
            case_id=db_case.case_id,
            event_type="creation",
            previous_value=None,
            new_value="NEW",
            actor="system",
            case_version=1,
            audit_context={
                "event_source": "system",
                "changed_fields": ["status"],
                "previous_case_version": None,
                "new_case_version": 1
            }
        )
        self.session.add(event)
        
        return db_case

    def get_list(
        self, 
        skip: int = 0, 
        limit: int = 50, 
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        min_probability: Optional[float] = None
    ) -> Tuple[List[Case], int]:
        query = self.session.query(Case)
        
        if status:
            query = query.filter(Case.status == status)
        if risk_level:
            query = query.filter(Case.risk_level == risk_level)
        if min_probability is not None:
            query = query.filter(Case.calibrated_probability >= min_probability)
            
        total = query.count()
        # Deterministic sorting
        query = query.order_by(desc(Case.created_at), Case.case_id)
        
        items = query.offset(skip).limit(limit).all()
        return items, total

    def update_case(self, case: Case) -> None:
        """Saves a mutated Case object back to the DB."""
        # Using SQLAlchemy's standard ORM update. 
        # The optimistic locking logic must be handled properly.
        # But to enforce version explicitly via WHERE clause:
        
        # We will use the Session to commit changes, but version increment must be exact.
        pass

    def optimistic_update(
        self, 
        case_id: str, 
        expected_version: int, 
        new_status: str, 
        new_priority: str, 
        new_note: Optional[str],
        actor: str,
        current_status: str,
        current_priority: str,
        current_note: Optional[str]
    ) -> bool:
        """
        Performs an optimistic update. Returns True if successful, False if version mismatch.
        """
        # Execute an UPDATE with WHERE case_id=? AND version=?
        result = self.session.execute(
            update(Case)
            .where(Case.case_id == case_id, Case.version == expected_version)
            .values(
                status=new_status,
                priority=new_priority,
                analyst_note=new_note,
                version=expected_version + 1,
                updated_at=datetime.utcnow()
            )
        )
        
        if result.rowcount == 0:
            return False
            
        new_version = expected_version + 1
        
        # Log events for each mutated field
        if current_status != new_status:
            self.session.add(CaseEvent(
                case_id=case_id,
                event_type="status_change",
                previous_value=current_status,
                new_value=new_status,
                actor=actor,
                case_version=new_version,
                audit_context={
                    "event_source": "api_update",
                    "changed_fields": ["status"],
                    "previous_case_version": expected_version,
                    "new_case_version": new_version
                }
            ))
            
        if current_priority != new_priority:
            self.session.add(CaseEvent(
                case_id=case_id,
                event_type="priority_change",
                previous_value=current_priority,
                new_value=new_priority,
                actor=actor,
                case_version=new_version,
                audit_context={
                    "event_source": "api_update",
                    "changed_fields": ["priority"],
                    "previous_case_version": expected_version,
                    "new_case_version": new_version
                }
            ))
            
        if current_note != new_note:
            self.session.add(CaseEvent(
                case_id=case_id,
                event_type="note_change",
                previous_value="[REDACTED]",
                new_value="[REDACTED]",
                actor=actor,
                case_version=new_version,
                audit_context={
                    "event_source": "api_update",
                    "changed_fields": ["analyst_note"],
                    "previous_case_version": expected_version,
                    "new_case_version": new_version
                }
            ))
        
        return True

    def get_events(self, case_id: str) -> List[CaseEvent]:
        return self.session.query(CaseEvent).filter(CaseEvent.case_id == case_id).order_by(CaseEvent.timestamp).all()

    def get_summary_metrics(self) -> dict:
        total_open = self.session.query(Case).filter(Case.status.in_(["NEW", "UNDER_REVIEW"])).count()
        critical_high = self.session.query(Case).filter(
            Case.status.in_(["NEW", "UNDER_REVIEW"]), 
            Case.risk_level.in_(["CRITICAL", "HIGH"])
        ).count()
        confirmed_fraud = self.session.query(Case).filter(Case.status == "CONFIRMED_FRAUD").count()
        
        # Median probability for open cases
        # SQLite doesn't have a built-in PERCENTILE_CONT, so we do a quick fetch
        open_probs = [
            r[0] for r in self.session.query(Case.calibrated_probability)
            .filter(Case.status.in_(["NEW", "UNDER_REVIEW"])).all()
        ]
        
        if open_probs:
            open_probs.sort()
            mid = len(open_probs) // 2
            median_prob = open_probs[mid]
        else:
            median_prob = 0.0
            
        status_dist = dict(self.session.query(Case.status, func.count(Case.case_id)).group_by(Case.status).all())
        risk_dist = dict(self.session.query(Case.risk_level, func.count(Case.case_id)).filter(Case.status.in_(["NEW", "UNDER_REVIEW"])).group_by(Case.risk_level).all())
        
        return {
            "total_open_cases": total_open,
            "critical_high_cases": critical_high,
            "confirmed_fraud_count": confirmed_fraud,
            "median_probability": median_prob,
            "status_distribution": status_dist,
            "risk_distribution": risk_dist
        }
