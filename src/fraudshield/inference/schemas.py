import datetime
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Union

from typing import Literal

class TransactionRequest(BaseModel):
    model_config = {'extra': 'forbid'}

    step: int = Field(..., gt=0, description="1 step represents 1 hour of time. Must be positive.")
    type: Literal['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'] = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0.0, description="Amount of the transaction in local currency")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Initial balance before the transaction")
    
    # User approved using these instead of the raw account identifiers (nameOrig/nameDest)
    orig_account_type: Literal['C', 'M'] = Field(..., description="Type of the origin account (e.g., C for Customer, M for Merchant)")
    dest_account_type: Literal['C', 'M'] = Field(..., description="Type of the destination account")
    
    # Optional field that may not be available for all transactions but is safe
    oldbalanceDest: Optional[float] = Field(None, ge=0.0, description="Initial balance recipient before the transaction")

    @model_validator(mode='before')
    @classmethod
    def check_forbidden_fields(cls, data: dict) -> dict:
        forbidden = [
            'isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 
            'error_balance_orig', 'error_balance_dest', 'nameOrig', 'nameDest',
            'fraud_score', 'calibrated_probability', 'risk_level'
        ]
        for f in forbidden:
            if f in data:
                raise ValueError(f"Forbidden field '{f}' is not allowed in real-time inference.")
        return data

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]

class TopContributor(BaseModel):
    feature: str
    display_name: str
    value: Union[float, int, str, bool, None]
    attribution: float
    direction: str  # "increases_risk" or "decreases_risk"
    reason_code: Optional[str] = None

class ExplanationBlock(BaseModel):
    enabled: bool
    method: Optional[str] = None
    target: Optional[str] = None
    convergence_delta: Optional[float] = None

class InferenceResponse(BaseModel):
    fraud_score: float = Field(..., description="Raw logit score from the model")
    calibrated_probability: float = Field(..., description="Empirically calibrated estimate of fraud probability (0-1)")
    risk_level: str = Field(..., description="Policy-defined risk bucket (LOW, MEDIUM, HIGH, CRITICAL)")
    is_alert: bool = Field(..., description="Whether the probability exceeds the operational decision threshold")
    decision_threshold: float = Field(..., description="Operational decision threshold used for alert")
    reason_codes: List[str] = Field(default_factory=list, description="List of reason codes (signals) contributing to the decision")
    top_contributors: List[TopContributor] = Field(default_factory=list, description="Top feature contributions to the prediction")
    explanation: ExplanationBlock = Field(..., description="Details about the explanation generated (if enabled)")
    model_version: str = Field(..., description="Model version from metadata")
    inference_id: str = Field(..., description="Unique ID for this inference request")
    scored_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
