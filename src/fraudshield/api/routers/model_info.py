from fastapi import APIRouter, Depends
from fraudshield.inference.predictor import FraudPredictor
from fraudshield.api.dependencies import get_predictor, get_api_key

router = APIRouter(tags=["Metadata"])

@router.get("/model-info", dependencies=[Depends(get_api_key)])
async def model_info(predictor: FraudPredictor = Depends(get_predictor)):
    """
    Returns metadata about the active inference bundle and model configurations.
    """
    env = predictor.env
    
    return {
        "model_version": predictor.model_version,
        "model_type": "MLP_Weighted",
        "feature_contract": env.feature_schema,
        "calibration_method": "Isotonic",
        "calibrator_input_type": env.metadata.get("calibrator_input_type", "sigmoid_probability"),
        "decision_threshold": env.operational_threshold,
        "risk_policy": env.risk_levels,
        "explanation_method": "Integrated Gradients" if predictor.explainability_enabled else None,
        "bundle_format": "fraudshield-isotonic-v1",
        "known_limitations": [
            "Trained entirely on PaySim (synthetic data).",
            "Artifact does not use original balance difference features to ensure generalization.",
            "Explanation (IG) explains raw logit, not final probability."
        ]
    }
