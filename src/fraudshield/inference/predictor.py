import pandas as pd
import numpy as np
import torch
import time
import json
import uuid
import yaml
from typing import List, Dict, Any, Union
from pydantic import ValidationError

from fraudshield.inference.schemas import TransactionRequest, BatchTransactionRequest, InferenceResponse, TopContributor
from fraudshield.inference.model_loader import get_model_env
from fraudshield.explainability.reason_codes import ReasonCodeEngine
from fraudshield.models.risk_levels import assign_risk_level
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

# Fallback wrapper for captum if it's not installed but explainability is enabled
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
class FraudPredictor:
    def __init__(self, config_path: str = "configs/inference.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.env = get_model_env(self.config.get("artifacts_dir", "artifacts/final"))
        self.reason_engine = ReasonCodeEngine(self.config.get("reason_codes_config_path", "configs/reason_codes.yaml"))
        self.fail_fast = self.config.get("fail_fast", False)
        self.explainability_enabled = self.config.get("explainability", {}).get("enabled", True) and CAPTUM_AVAILABLE
        self.model_version = self.config.get("model_version", "1.0.0")
        
        # Calibrator Input Contract Verification (Step 3)
        calib_input_type = self.env.metadata.get("calibrator_input_type", "")
        if calib_input_type != "sigmoid_probability":
            raise ValueError(f"Calibrator input type mismatch. Expected 'sigmoid_probability', got '{calib_input_type}'.")
            
        self.env.model.eval() # PyTorch Inference Safety (Step 4)
        
        if self.explainability_enabled:
            baseline_path = self.env.bundle_path / "reference_baseline.json"
            with open(baseline_path, "r") as f:
                self.reference_baseline = json.load(f)
                
            self.ig = IntegratedGradients(self.env.model)
            self.feature_names_out = self._get_feature_names_out()
            self._precompute_baseline_tensor()

    def _get_feature_names_out(self):
        if hasattr(self.env.preprocessor, "get_feature_names_out"):
            return self.env.preprocessor.get_feature_names_out()
        return [f"feature_{i}" for i in range(self.env.model.network[0].in_features)]

    def _precompute_baseline_tensor(self):
        # Create a single row dataframe for the baseline
        df_base = pd.DataFrame([self.reference_baseline])
        
        # We need to make sure the df_base matches the feature_schema exactly
        df_ordered = df_base[self.env.feature_schema]
        
        # Transform
        X_base = self.env.preprocessor.transform(df_ordered)
        if hasattr(X_base, "toarray"):
            X_base = X_base.toarray()
            
        self.baseline_tensor = torch.FloatTensor(X_base).to(next(self.env.model.parameters()).device)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Point-in-time safe feature engineering. Minimal subset of builder.py
        """
        df = df.copy()
        
        df['hour_of_day'] = ((df['step'] - 1) % 24).astype('int8')
        df['day'] = ((df['step'] - 1) // 24 + 1).astype('int16')
        
        df['is_risky_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype('int8')
        df['amount_log1p'] = np.log1p(df['amount']).astype('float32')
        
        df['orig_oldbalance_is_zero'] = (df['oldbalanceOrg'] == 0).astype('int8')
        df['amount_to_oldbalance_orig_ratio'] = np.where(
            df['oldbalanceOrg'] == 0, 0.0, df['amount'] / df['oldbalanceOrg']
        ).astype('float32')
        
        if 'orig_account_type' not in df.columns and 'nameOrig' in df.columns:
            df['orig_account_type'] = df['nameOrig'].astype(str).str[0]
        df['orig_account_type'] = df.get('orig_account_type', 'C').astype('category')
        
        if 'dest_account_type' not in df.columns and 'nameDest' in df.columns:
            df['dest_account_type'] = df['nameDest'].astype(str).str[0]
        df['dest_account_type'] = df.get('dest_account_type', 'C').astype('category')
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df[self.env.feature_schema]

    def _group_attributions(self, attributions: np.ndarray) -> List[Dict[str, Any]]:
        grouped = {}
        for name_out, attr_val in zip(self.feature_names_out, attributions):
            clean_name = name_out.split('__')[-1]
            parent_feature = clean_name
            for raw_f in self.env.feature_schema:
                if clean_name.startswith(f"{raw_f}_"):
                    parent_feature = raw_f
                    break
            
            if parent_feature not in grouped:
                grouped[parent_feature] = 0.0
            grouped[parent_feature] += attr_val
            
        results = []
        for feat, val in grouped.items():
            results.append({
                "feature": feat,
                "display_name": feat.replace("_", " ").title(),
                "value": float(val), 
                "attribution": float(val),
                "direction": "positive" if val > 0 else "negative"
            })
            
        return sorted(results, key=lambda x: abs(x["attribution"]), reverse=True)

    def predict_single(self, request: TransactionRequest, explain: bool = False) -> InferenceResponse:
        t0 = time.time()
        inference_id = str(uuid.uuid4())
        
        df_raw = pd.DataFrame([request.model_dump()])
        
        # PyTorch Inference Safety: Reject NaN/Inf explicitly
        if not np.isfinite(df_raw.select_dtypes(include=[np.number])).all().all():
            raise ValueError("Input data contains NaN or Inf values which are not allowed.")
            
        df_feats = self._build_features(df_raw)
        
        # 1. Transform features
        X_t = self.env.preprocessor.transform(df_feats)
        if hasattr(X_t, "toarray"): X_t = X_t.toarray()
        X_tensor = torch.FloatTensor(X_t).to(next(self.env.model.parameters()).device)
        
        # 2. Get base fraud score (logit) and probability
        with torch.inference_mode(): # Step 4: Strict inference mode
            raw_output = self.env.model(X_tensor)
            fraud_score = float(raw_output.cpu().numpy()[0])
            # numerically stable sigmoid
            prob_base = float(torch.sigmoid(raw_output).cpu().numpy()[0])
            
        # 3. Apply stable calibrator directly
        prob = float(self.env.calibrator.predict([prob_base])[0])
        
        # Handle NaN/Inf safely
        if not np.isfinite(prob):
            prob = 0.0
            logger.error(f"Prediction resulted in non-finite probability for {inference_id}")
            
        is_alert = bool(prob >= self.env.operational_threshold)
        risk_level = assign_risk_level(prob, self.env.risk_levels)
        
        reason_codes = []
        top_contributors = []
        
        do_explain = explain and self.explainability_enabled
        
        explanation_block = {
            "enabled": do_explain,
            "method": "Integrated Gradients" if do_explain else None,
            "target": "Raw Logit" if do_explain else None,
            "convergence_delta": None
        }
        
        if do_explain:
            attributions, delta = self.ig.attribute(
                X_tensor, 
                baselines=self.baseline_tensor, 
                return_convergence_delta=True
            )
            
            # Explainability Contract Verification (Step 5)
            if delta is not None and abs(delta.item()) > 0.05:
                logger.warning(f"IG convergence delta {delta.item():.4f} is high for inference {inference_id}")
            if delta is not None:
                explanation_block["convergence_delta"] = float(delta.item())
            
            attr_np = attributions.cpu().detach().numpy()[0]
            grouped_attrs = self._group_attributions(attr_np)
            
            reason_codes = self.reason_engine.evaluate(grouped_attrs)
            
            # Map Reason Codes to Top Contributors and take top 5
            for attr in grouped_attrs[:5]:
                # find matching reason code if any
                rc_match = next((r['code'] for r in self.reason_engine.reason_codes if r['feature'] == attr['feature'] and r['code'] in reason_codes), None)
                top_contributors.append(TopContributor(
                    feature=attr['feature'],
                    display_name=attr['display_name'],
                    value=df_feats.iloc[0][attr['feature']],
                    attribution=attr['attribution'],
                    direction="increases_risk" if attr['direction'] == "positive" else "decreases_risk",
                    reason_code=rc_match
                ))
                
        return InferenceResponse(
            fraud_score=fraud_score,
            calibrated_probability=float(prob),
            risk_level=risk_level,
            is_alert=is_alert,
            decision_threshold=self.env.operational_threshold,
            reason_codes=reason_codes,
            top_contributors=top_contributors,
            explanation=explanation_block,
            model_version=self.model_version,
            inference_id=inference_id
        )
        
    def predict_batch(self, batch: BatchTransactionRequest, explain: bool = False) -> List[Union[InferenceResponse, dict]]:
        results = []
        for idx, req in enumerate(batch.transactions):
            try:
                res = self.predict_single(req, explain=explain)
                results.append(res)
            except Exception as e:
                if self.fail_fast:
                    raise e
                else:
                    logger.error(f"Error predicting on row {idx}: {e}")
                    results.append({"error": str(e), "row_index": idx})
        return results
