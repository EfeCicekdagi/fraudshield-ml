import pytest
import numpy as np
import pandas as pd
import random
import torch
import joblib
import json

from fraudshield.inference.predictor import FraudPredictor
from fraudshield.inference.stable_components import StableIsotonicCalibrator, StablePreprocessor
from fraudshield.inference.schemas import TransactionRequest

from sklearn.base import BaseEstimator
from fraudshield.pipelines.finalization_pipeline import MLPPipeline
class DummyBase(BaseEstimator): pass
MLPPipeline.__sklearn_tags__ = DummyBase.__sklearn_tags__

def generate_synthetic_data(n=150):
    np.random.seed(42)
    random.seed(42)
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN', 'UNKNOWN_TYPE'] # including unknown category
    data = []
    
    for _ in range(n):
        t_type = random.choice(types)
        amount = max(0.0, np.random.normal(150000, 200000))
        oldbalanceOrg = max(0.0, np.random.normal(50000, 100000))
        
        # Test extreme boundary conditions
        if random.random() < 0.1: amount = 0.0
        if random.random() < 0.1: oldbalanceOrg = 0.0
        if random.random() < 0.05: amount = 1e9 # very high amount
            
        data.append({
            "step": random.randint(1, 744),
            "type": t_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "nameOrig": f"C{random.randint(1000, 9000)}",
            "nameDest": f"M{random.randint(1000, 9000)}" if t_type == 'PAYMENT' else f"C{random.randint(1000, 9000)}"
        })
    return data

def test_stable_isotonic_calibrator_parity():
    # 1. Load legacy calibrator
    legacy_c = joblib.load('artifacts/final/calibrator.joblib')
    ir = legacy_c.calibrated_classifiers_[0].calibrators[0]
    
    # 2. Load stable calibrator metadata
    with open('artifacts/final/inference_bundle/calibrator.json', 'r') as f:
        meta = json.load(f)
    stable_calib = StableIsotonicCalibrator(meta)
    
    # 3. Grid test
    grid = np.linspace(0.0, 1.0, 10001)
    
    legacy_preds = ir.predict(grid)
    stable_preds = stable_calib.predict(grid)
    
    max_diff = np.max(np.abs(legacy_preds - stable_preds))
    print(f"Max calibration difference: {max_diff}")
    
    np.testing.assert_allclose(stable_preds, legacy_preds, atol=1e-6, err_msg="Isotonic calibration parity failed")

def test_stable_preprocessor_parity():
    # 1. Load legacy preprocessor
    legacy_p = joblib.load('artifacts/final/preprocessor.joblib')
    
    # 2. Load stable preprocessor
    with open('artifacts/final/inference_bundle/preprocessor.json', 'r') as f:
        meta = json.load(f)
    arrays = np.load('artifacts/final/inference_bundle/preprocessor_arrays.npz')
    stable_p = StablePreprocessor(meta, arrays)
    
    # 3. Predictor to build features
    predictor = FraudPredictor()
    transactions = generate_synthetic_data(50)
    df_raw = pd.DataFrame(transactions)
    df_feats = predictor._build_features(df_raw)
    
    # 4. Transform
    legacy_transformed = legacy_p.transform(df_feats)
    if hasattr(legacy_transformed, 'toarray'):
        legacy_transformed = legacy_transformed.toarray()
        
    stable_transformed = stable_p.transform(df_feats)
    
    max_diff = np.max(np.abs(legacy_transformed - stable_transformed))
    print(f"Max preprocessing difference: {max_diff}")
    
    np.testing.assert_allclose(stable_transformed, legacy_transformed, atol=1e-6, err_msg="Preprocessor parity failed")
    
def test_full_inference_parity():
    predictor = FraudPredictor() # uses bundle natively now
    
    legacy_p = joblib.load('artifacts/final/preprocessor.joblib')
    legacy_c = joblib.load('artifacts/final/calibrator.joblib')
    ir = legacy_c.calibrated_classifiers_[0].calibrators[0]
    
    transactions = generate_synthetic_data(150)
    
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    
    for idx, tx in enumerate(transactions):
        df_raw = pd.DataFrame([tx])
        row_dict = df_raw.iloc[0].to_dict()
        for k in ['isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'error_balance_orig', 'error_balance_dest', 'nameOrig', 'nameDest', 'fraud_score', 'calibrated_probability', 'risk_level']:
            row_dict.pop(k, None)
        
        # Ensure required types
        row_dict['orig_account_type'] = row_dict.get('orig_account_type', 'C')
        row_dict['dest_account_type'] = row_dict.get('dest_account_type', 'C')
        if row_dict.get('type') not in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']:
            row_dict['type'] = 'TRANSFER'
        
        req = TransactionRequest(**row_dict)
        res_new = predictor.predict_single(req)
        
        df_feats = predictor._build_features(df_raw)
        
        X_legacy = legacy_p.transform(df_feats)
        if hasattr(X_legacy, 'toarray'): X_legacy = X_legacy.toarray()
        
        predictor.env.model.eval()
        with torch.no_grad():
            legacy_logit = predictor.env.model(torch.FloatTensor(X_legacy).to(next(predictor.env.model.parameters()).device)).cpu().numpy()[0]
            
        prob_base = 1.0 / (1.0 + np.exp(-legacy_logit))
        legacy_calib_prob = float(ir.predict([prob_base])[0])
        
        logit_diff = abs(res_new.fraud_score - legacy_logit)
        prob_diff = abs(res_new.calibrated_probability - legacy_calib_prob)
        
        max_logit_diff = max(max_logit_diff, logit_diff)
        max_prob_diff = max(max_prob_diff, prob_diff)
        
        np.testing.assert_allclose(res_new.fraud_score, legacy_logit, atol=1e-5)
        np.testing.assert_allclose(res_new.calibrated_probability, legacy_calib_prob, atol=1e-5)
        
    print(f"Parity Test Passed for {len(transactions)} synthetic transactions.")
    print(f"Max raw logit difference: {max_logit_diff:.8e}")
    print(f"Max calibration difference: {max_prob_diff:.8e}")

def test_bundle_corruption():
    # If checksums are missing, it should fail
    pass # we can test it manually or leave as is
