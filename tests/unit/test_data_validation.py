import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure src is in path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fraudshield.data.loader import load_raw_data
from fraudshield.data.validation import validate_raw_data

@pytest.fixture
def dummy_data_path(tmp_path):
    df = pd.DataFrame({
        'step': [1, 1],
        'type': ['PAYMENT', 'TRANSFER'],
        'amount': [100.0, 500.0],
        'nameOrig': ['A123', 'B456'],
        'oldbalanceOrg': [1000.0, 500.0],
        'newbalanceOrig': [900.0, 0.0],
        'nameDest': ['M123', 'C789'],
        'oldbalanceDest': [0.0, 100.0],
        'newbalanceDest': [0.0, 600.0],
        'isFraud': [0, 1],
        'isFlaggedFraud': [0, 0]
    })
    filepath = tmp_path / "dummy.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_load_raw_data_success(dummy_data_path):
    df = load_raw_data(dummy_data_path)
    assert not df.empty
    assert len(df) == 2
    assert df['type'].dtype.name == 'category'

def test_load_raw_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_file.csv")

def test_validate_raw_data(dummy_data_path):
    df = load_raw_data(dummy_data_path)
    required_cols = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 
        'newbalanceDest', 'isFraud', 'isFlaggedFraud'
    ]
    results = validate_raw_data(df, required_columns=required_cols)
    
    assert results['has_all_required_columns'] == True
    assert results['num_rows'] == 2
    assert results['num_cols'] == 11
    assert results['duplicate_rows'] == 0
    assert results['negative_amounts'] == 0
    assert results['isFraud_valid_values'] == True
    assert results['total_fraud'] == 1
    assert results['fraud_rate'] == 0.5
