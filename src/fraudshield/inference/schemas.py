import datetime
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Union

class TransactionRequest(BaseModel):
    step: int = Field(..., description="1 step represents 1 hour of time.")
    type: str = Field(..., description="Transaction type, e.g., PAYMENT, TRANSFER, CASH_OUT")
    amount: float = Field(..., ge=0.0, description="Amount of the transaction in local currency")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Initial balance before the transaction")
    nameOrig: str = Field(..., description="Customer who started the transaction")
    nameDest: str = Field(..., description="Customer who is the recipient of the transaction")
    
    # Optional field that may not be available for all transactions but is safe
    oldbalanceDest: Optional[float] = Field(None, ge=0.0, description="Initial balance recipient before the transaction")

    @model_validator(mode='before')
    @classmethod
    def check_forbidden_fields(cls, data: dict) -> dict:
        forbidden = ['isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'error_balance_orig', 'error_balance_dest']
        for f in forbidden:
            if f in data:
                raise ValueError(f"Forbidden post-transaction or target feature '{f}' is not allowed in real-time inference.")
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

class InferenceResponse(BaseModel):
    fraud_score: float = Field(..., description="Raw logit score from the model")
    calibrated_probability: float = Field(..., description="Calibrated probability of fraud (0-1)")
    risk_level: str = Field(..., description="Policy-defined risk bucket (LOW, MEDIUM, HIGH, CRITICAL)")
    is_alert: bool = Field(..., description="Whether the probability exceeds the operational decision threshold")
    decision_threshold: float = Field(..., description="Operational decision threshold used for alert")
    reason_codes: List[str] = Field(default_factory=list, description="List of reason codes explaining the high risk")
    top_contributors: List[TopContributor] = Field(default_factory=list, description="Top feature contributions to the prediction")
    model_version: str = Field(..., description="Model version from metadata")
    inference_id: str = Field(..., description="Unique ID for this inference request")
    scored_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
