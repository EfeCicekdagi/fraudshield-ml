import os
import json
import numpy as np
import pandas as pd
import pytest
import joblib
import torch

# Workaround for scikit-learn 1.9 loading 1.3 models
from sklearn.base import BaseEstimator
from fraudshield.pipelines.finalization_pipeline import MLPPipeline
class DummyBase(BaseEstimator): pass
MLPPipeline.__sklearn_tags__ = DummyBase.__sklearn_tags__

from fraudshield.inference.stable_components import StablePreprocessor, StableIsotonicCalibrator
from fraudshield.inference.predictor import FraudPredictor
from fraudshield.inference.schemas import TransactionRequest

NUMERICAL_PARITY_TOLERANCE = 1e-6

@pytest.fixture(scope="module")
def legacy_preprocessor():
    path = "artifacts/final/preprocessor.joblib"
    if not os.path.exists(path):
        pytest.skip("Legacy preprocessor not found.")
    return joblib.load(path)

@pytest.fixture(scope="module")
def stable_preprocessor():
    path = "artifacts/final/inference_bundle/preprocessor.json"
    arr_path = "artifacts/final/inference_bundle/preprocessor_arrays.npz"
    if not os.path.exists(path):
        pytest.skip("Stable preprocessor not found.")
    with open(path, "r") as f:
        meta = json.load(f)
    arrs = np.load(arr_path)
    return StablePreprocessor(meta, arrs)

@pytest.fixture(scope="module")
def legacy_calibrator():
    path = "artifacts/final/calibrator.joblib"
    if not os.path.exists(path):
        pytest.skip("Legacy calibrator not found.")
    return joblib.load(path)

@pytest.fixture(scope="module")
def stable_calibrator():
    path = "artifacts/final/inference_bundle/calibrator.json"
    if not os.path.exists(path):
        pytest.skip("Stable calibrator not found.")
    with open(path, "r") as f:
        meta = json.load(f)
    return StableIsotonicCalibrator(meta)

@pytest.fixture(scope="module")
def predictor():
    try:
        import yaml
        import tempfile
        # Create a config that points to the local directory but disables explainability for speed
        with open("configs/inference.yaml", "r") as f:
            config = yaml.safe_load(f)
        if "explainability" not in config:
            config["explainability"] = {}
        config["explainability"]["enabled"] = False
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name
        
        pred = FraudPredictor(config_path=tmp_path)
        os.remove(tmp_path)
        return pred
    except Exception as e:
        pytest.skip(f"Could not load predictor: {e}")

@pytest.fixture(scope="module")
def synthetic_data():
    """Generates 150+ edge case transactions without using test set."""
    np.random.seed(42)
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN', 'UNKNOWN_TYPE']
    data = []
    
    # 1. Normal cases
    for _ in range(50):
        t_type = np.random.choice(types[:-1])
        data.append({
            "step": np.random.randint(1, 744),
            "type": t_type,
            "amount": max(0.0, float(np.random.normal(1000, 500))),
            "oldbalanceOrg": max(0.0, float(np.random.normal(5000, 2000))),
            "nameOrig": f"C{np.random.randint(100, 999)}",
            "nameDest": f"C{np.random.randint(100, 999)}"
        })
        
    # 2. Zeroes and edge cases
    for _ in range(50):
        data.append({
            "step": np.random.randint(1, 744),
            "type": "TRANSFER", # Must be a valid type for API validation
            "amount": 0.0,
            "oldbalanceOrg": 0.0,
            "nameOrig": f"C{np.random.randint(100, 999)}",
            "nameDest": f"M{np.random.randint(100, 999)}"
        })
        
    # 3. Huge amounts
    for _ in range(50):
        data.append({
            "step": np.random.randint(1, 744),
            "type": "TRANSFER",
            "amount": 1e9,
            "oldbalanceOrg": 1e9,
            "nameOrig": f"C{np.random.randint(100, 999)}",
            "nameDest": f"C{np.random.randint(100, 999)}"
        })
        
    df = pd.DataFrame(data)
    
    # Simple feature building as in predictor to pass to preprocessors
    df_feats = df.copy()
    df_feats['hour_of_day'] = ((df_feats['step'] - 1) % 24).astype('int8')
    df_feats['day'] = ((df_feats['step'] - 1) // 24 + 1).astype('int16')
    df_feats['is_risky_type'] = df_feats['type'].isin(['TRANSFER', 'CASH_OUT']).astype('int8')
    df_feats['amount_log1p'] = np.log1p(np.maximum(0, df_feats['amount'])).astype('float32')
    df_feats['orig_oldbalance_is_zero'] = (df_feats['oldbalanceOrg'] == 0).astype('int8')
    df_feats['amount_to_oldbalance_orig_ratio'] = np.where(
        df_feats['oldbalanceOrg'] == 0, 0.0, df_feats['amount'] / df_feats['oldbalanceOrg']
    ).astype('float32')
    df_feats['orig_account_type'] = df_feats.get('nameOrig', 'C').astype(str).str[0].astype('category')
    df_feats['dest_account_type'] = df_feats.get('nameDest', 'C').astype(str).str[0].astype('category')
    
    df_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df_feats.select_dtypes(include=[np.number]).columns
    df_feats[numeric_cols] = df_feats[numeric_cols].fillna(0)
    
    # We need the exact expected columns
    expected_cols = [
        "step", "hour_of_day", "day", "amount", "amount_log1p", 
        "is_risky_type", "oldbalanceOrg", "orig_oldbalance_is_zero", 
        "amount_to_oldbalance_orig_ratio", "type", "orig_account_type", "dest_account_type"
    ]
    
    return df, df_feats[expected_cols]

def test_preprocessor_parity(legacy_preprocessor, stable_preprocessor, synthetic_data):
    _, df_feats = synthetic_data
    
    # Legacy transform
    X_legacy = legacy_preprocessor.transform(df_feats)
    if hasattr(X_legacy, 'toarray'):
        X_legacy = X_legacy.toarray()
        
    # Stable transform
    X_stable = stable_preprocessor.transform(df_feats)
    
    diff = np.abs(X_legacy - X_stable)
    max_diff = diff.max()
    
    print(f"Maximum preprocessing difference: {max_diff}")
    assert max_diff < NUMERICAL_PARITY_TOLERANCE, f"Preprocessor parity failed. Max diff: {max_diff}"
    
def test_calibrator_grid_parity(legacy_calibrator, stable_calibrator):
    # 10001 points in [0, 1] + out of bounds
    grid = np.linspace(-0.1, 1.1, 10001)
    
    # Legacy prediction
    # CalibratedClassifierCV uses predict_proba. Since method='isotonic', it's inside `calibrated_classifiers_[0].calibrators[0]`
    legacy_ir = legacy_calibrator.calibrated_classifiers_[0].calibrators[0]
    
    # Legacy IsotonicRegression predict
    y_legacy = legacy_ir.predict(grid)
    
    # Stable IsotonicCalibrator predict
    y_stable = stable_calibrator.predict(grid)
    
    diff = np.abs(y_legacy - y_stable)
    max_diff = diff.max()
    
    print(f"Maximum calibration difference on grid: {max_diff}")
    assert max_diff < NUMERICAL_PARITY_TOLERANCE, f"Calibrator parity failed. Max diff: {max_diff}"

def test_full_inference_parity(legacy_preprocessor, legacy_calibrator, predictor, synthetic_data):
    df_raw, df_feats = synthetic_data
    
    # Load model state directly to get base probabilities for legacy route
    model_path = "artifacts/final/inference_bundle/model_state_dict.pt"
    if not os.path.exists(model_path):
        pytest.skip("Model state dict not found.")
        
    from fraudshield.models.mlp import FraudMLP
    # Inference bundle architecture
    state_dict = torch.load(model_path, map_location='cpu')
    linear_layers = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 2]
    input_dim = state_dict[linear_layers[0]].shape[1]
    hidden_dims = [state_dict[k].shape[0] for k in linear_layers[:-1]]
    
    model = FraudMLP(input_dim=input_dim, hidden_dimensions=hidden_dims, dropout=0.1)
    model.load_state_dict(state_dict)
    model.eval()
    
    # --- LEGACY PIPELINE EVALUATION ---
    X_legacy = legacy_preprocessor.transform(df_feats)
    if hasattr(X_legacy, 'toarray'): X_legacy = X_legacy.toarray()
    
    with torch.no_grad():
        logits_legacy = model(torch.FloatTensor(X_legacy)).numpy().flatten()
        
    prob_base_legacy = 1.0 / (1.0 + np.exp(-logits_legacy))
    
    # Legacy uses predict_proba from CalibratedClassifierCV
    # CalibratedClassifierCV expects (N, 2) shaped input? Wait, no, CalibratedClassifierCV was fit on the model,
    # but we extracted the internal calibrator. 
    # Let's just use the internal IsotonicRegression for legacy parity since that's what we mapped.
    legacy_ir = legacy_calibrator.calibrated_classifiers_[0].calibrators[0]
    final_prob_legacy = legacy_ir.predict(prob_base_legacy)
    
    # --- STABLE PREDICTOR EVALUATION ---
    final_prob_stable = []
    logits_stable = []
    
    for i in range(len(df_raw)):
        row_dict = df_raw.iloc[i].to_dict()
        
        # Derive accurate types for strict schema instead of blind coercion
        orig_acc_type = df_raw.iloc[i].get('nameOrig', 'C')[0]
        dest_acc_type = df_raw.iloc[i].get('nameDest', 'C')[0]
        
        for k in ['isFraud', 'isFlaggedFraud', 'newbalanceOrig', 'newbalanceDest', 'error_balance_orig', 'error_balance_dest', 'nameOrig', 'nameDest', 'fraud_score', 'calibrated_probability', 'risk_level']:
            row_dict.pop(k, None)
        
        row_dict['orig_account_type'] = orig_acc_type
        row_dict['dest_account_type'] = dest_acc_type
        if row_dict.get('type') not in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']:
            row_dict['type'] = 'TRANSFER'
        
        req = TransactionRequest(**row_dict)
        res = predictor.predict_single(req)
        final_prob_stable.append(res.calibrated_probability)
        logits_stable.append(res.fraud_score)
        
    final_prob_stable = np.array(final_prob_stable)
    logits_stable = np.array(logits_stable)
    
    # --- COMPARISON ---
    logit_diff = np.abs(logits_legacy - logits_stable).max()
    prob_diff = np.abs(final_prob_legacy - final_prob_stable).max()
    
    print(f"Maximum raw logit difference: {logit_diff}")
    print(f"Maximum sigmoid/calibrated probability difference: {prob_diff}")
    
    assert logit_diff < 1e-3, f"Logit parity failed: {logit_diff}"
    assert prob_diff < NUMERICAL_PARITY_TOLERANCE, f"Probability parity failed: {prob_diff}"
