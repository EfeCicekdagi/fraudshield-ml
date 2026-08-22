import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.build_features import build_all_features

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'step': [1, 25, 49],
        'type': ['TRANSFER', 'CASH_IN', 'PAYMENT'],
        'amount': [100.0, 0.0, 50.0],
        'nameOrig': ['C123', 'M456', 'C789'],
        'oldbalanceOrg': [100.0, 0.0, 100.0],
        'newbalanceOrig': [0.0, 0.0, 50.0],
        'nameDest': ['C999', 'C888', 'M777'],
        'oldbalanceDest': [0.0, 0.0, 0.0],
        'newbalanceDest': [100.0, 0.0, 0.0],
        'isFraud': [0, 0, 0],
        'isFlaggedFraud': [0, 0, 0]
    })

def test_build_all_features(sample_df):
    df_feat = build_all_features(sample_df)
    
    # Check time features
    assert df_feat.loc[0, 'hour_of_day'] == 0
    assert df_feat.loc[1, 'hour_of_day'] == 0 # step 25 -> hour 0 of day 2
    assert df_feat.loc[1, 'day'] == 2
    
    # Check transaction features
    assert df_feat.loc[0, 'is_risky_type'] == 1
    assert df_feat.loc[1, 'is_risky_type'] == 0
    
    # Check zero division handling
    # amount_to_oldbalance_orig_ratio for index 1 should be 0.0 (amount=0, oldbalance=0)
    assert df_feat.loc[1, 'amount_to_oldbalance_orig_ratio'] == 0.0
    
    # Check account type parsing
    assert df_feat.loc[0, 'orig_account_type'] == 'C'
    assert df_feat.loc[1, 'orig_account_type'] == 'M'
    
    # Check removed columns
    assert 'nameOrig' not in df_feat.columns
    assert 'nameDest' not in df_feat.columns
    
    # Check target preservation
    assert 'isFraud' in df_feat.columns
    assert 'isFlaggedFraud' in df_feat.columns

    # Ensure no NaNs or Infs
    assert df_feat.isnull().sum().sum() == 0
    assert not np.isinf(df_feat.select_dtypes(include=[np.number])).values.any()
