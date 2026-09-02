import uuid
import datetime
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Enum, JSON
from fraudshield.logging_config import setup_logger

Base = declarative_base()

class Case(Base):
    __tablename__ = "cases"
    
    case_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    inference_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    transaction_reference: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
    status: Mapped[str] = mapped_column(String, default="NEW", nullable=False)
    priority: Mapped[str] = mapped_column(String, default="LOW", nullable=False)
    calibrated_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String, nullable=False)
    reason_codes: Mapped[str] = mapped_column(String, nullable=False)  # Stored as JSON string
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    analyst_note: Mapped[str] = mapped_column(String, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

class CaseEvent(Base):
    __tablename__ = "case_events"
    
    event_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id: Mapped[str] = mapped_column(String, ForeignKey("cases.case_id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False) # e.g. "status_change", "priority_change", "note_change", "creation"
    previous_value: Mapped[str] = mapped_column(String, nullable=True)
    new_value: Mapped[str] = mapped_column(String, nullable=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    case_version: Mapped[int] = mapped_column(Integer, nullable=False)
    audit_context: Mapped[dict] = mapped_column(JSON, nullable=True)
