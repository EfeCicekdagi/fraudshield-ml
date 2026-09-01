import pytest
import numpy as np
import pandas as pd
import json
import torch
import random
from fraudshield.inference.predictor import FraudPredictor
from fraudshield.inference.schemas import TransactionRequest
from fraudshield.pipelines.finalization_pipeline import MLPPipeline
from sklearn.base import BaseEstimator

from sklearn.base import BaseEstimator
class DummyBase(BaseEstimator):
    pass


def generate_synthetic_data(n=150):
    np.random.seed(42)
    random.seed(42)
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    data = []
    
    for _ in range(n):
        t_type = random.choice(types)
        amount = max(0.0, np.random.normal(150000, 200000))
        oldbalanceOrg = max(0.0, np.random.normal(50000, 100000))
        
        if random.random() < 0.1: amount = 0.0
        if random.random() < 0.1: oldbalanceOrg = 0.0
            
        data.append({
            "step": random.randint(1, 744),
            "type": t_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "nameOrig": f"C{random.randint(1000, 9000)}",
            "nameDest": f"M{random.randint(1000, 9000)}" if t_type == 'PAYMENT' else f"C{random.randint(1000, 9000)}"
        })
    return data

def test_inference_parity():
    predictor = FraudPredictor()
    env = predictor.env
    
    
    from fraudshield.pipelines.finalization_pipeline import MLPPipeline
    
    class DummyBase(BaseEstimator):
        pass
        
    MLPPipeline.__sklearn_tags__ = DummyBase.__sklearn_tags__
    
    calibrator = env.calibrator
    
    transactions = generate_synthetic_data(150)
    
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    
    for idx, tx in enumerate(transactions):
        req = TransactionRequest(**tx)
        res_new = predictor.predict_single(req)
        
        df_raw = pd.DataFrame([tx])
        df_feats = predictor._build_features(df_raw)
        
        X_t_legacy = env.preprocessor.transform(df_feats)
        if hasattr(X_t_legacy, "toarray"): X_t_legacy = X_t_legacy.toarray()
        
        # Use the original model directly
        env.model.eval()
        with torch.no_grad():
            legacy_logits = env.model(torch.FloatTensor(X_t_legacy).to(next(env.model.parameters()).device)).cpu().numpy()[0]
            
        # Due to scikit-learn 1.9 incompatibility with 1.3 pickled CalibratedClassifierCV, 
        # calibrator.predict_proba evaluates the calibrator on the wrong column.
        # We manually apply the correct mathematical contract.
        legacy_prob_base = 1.0 / (1.0 + np.exp(-legacy_logits))
        cc = calibrator.calibrated_classifiers_[0]
        regressor = cc.calibrators[0]
        legacy_calib_prob = float(regressor.predict([legacy_prob_base])[0])

        logit_diff = abs(res_new.fraud_score - legacy_logits)
        prob_diff = abs(res_new.calibrated_probability - legacy_calib_prob)
        
        max_logit_diff = max(max_logit_diff, logit_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)
        np.testing.assert_allclose(res_new.fraud_score, legacy_logits, atol=1e-5, err_msg=f"Logit mismatch at index {idx}")
        np.testing.assert_allclose(res_new.calibrated_probability, legacy_calib_prob, atol=1e-5, err_msg=f"Calibrated Prob mismatch at index {idx}")
        
        legacy_is_alert = bool(legacy_calib_prob >= env.operational_threshold)
        assert res_new.is_alert == legacy_is_alert, f"Alert mismatch at index {idx}"
        
        from fraudshield.models.risk_levels import assign_risk_level
        legacy_risk = assign_risk_level(legacy_calib_prob, env.risk_levels)
        assert res_new.risk_level == legacy_risk, f"Risk mismatch at index {idx}"
        
    print(f"Parity Test Passed for {len(transactions)} synthetic transactions.")
    print(f"Max Logit Diff: {max_logit_diff:.8e}")
    print(f"Max Prob Diff: {max_prob_diff:.8e}")
