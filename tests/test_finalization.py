import os
import json
import pytest
from fraudshield.features.availability import check_feature_safety, FEATURE_AVAILABILITY
from fraudshield.models.risk_levels import get_risk_levels, assign_risk_level
import yaml

def test_post_transaction_feature_engelleme():
    # Attempting to use a post-transaction feature should fail
    bad_features = ['type', 'amount', 'newbalanceOrig']
    with pytest.raises(ValueError, match="is unsafe"):
        check_feature_safety(bad_features)

def test_dolayli_forbidden_dependency_kontrolu():
    # Any feature containing 'error' should be SYNTHETIC_ARTIFACT
    for f, status in FEATURE_AVAILABILITY.items():
        if 'error' in f:
            assert status == 'SYNTHETIC_ARTIFACT'
            
def test_risk_sinirlarinin_sirali_olmasi():
    thresholds = {
        'max_f1': 0.6,
        'recall_0.8': 0.4,
        'recall_0.9': 0.2
    }
    
    bounds = get_risk_levels(thresholds)
    
    assert bounds['LOW_MAX'] <= bounds['MEDIUM_MAX']
    assert bounds['MEDIUM_MAX'] <= bounds['HIGH_MAX']

def test_risk_level_assignment():
    bounds = {
        'LOW_MAX': 0.2,
        'MEDIUM_MAX': 0.4,
        'HIGH_MAX': 0.6
    }
    
    assert assign_risk_level(0.1, bounds) == 'LOW'
    assert assign_risk_level(0.3, bounds) == 'MEDIUM'
    assert assign_risk_level(0.5, bounds) == 'HIGH'
    assert assign_risk_level(0.7, bounds) == 'CRITICAL'

def test_synthetic_benchmark_is_not_api_candidate():
    config_path = "configs/final_model.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    features = config['feature_sets']['pre_transaction']
    
    # Assert that no error features are in the pre_transaction list
    for f in features:
        assert 'error' not in f
        assert 'newbalance' not in f
