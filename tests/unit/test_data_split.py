import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraudshield.data.splitting import temporal_train_val_test_split

@pytest.fixture
def temporal_df():
    # 10 records, with some duplicate steps
    return pd.DataFrame({
        'step': [1, 1, 2, 2, 2, 3, 4, 5, 5, 6],
        'value': range(10)
    })

def test_temporal_train_val_test_split(temporal_df):
    train, val, test = temporal_train_val_test_split(temporal_df, train_size=0.5, val_size=0.2)
    
    # Sizes: train approx 5, val approx 2, test approx 3
    # target train idx = 5. Step at index 5 is 3. Max index for step 3 is 5. Split is at 6.
    # target val idx = 7. Step at index 7 is 5. Max index for step 5 is 8. Split is at 9.
    assert len(train) == 6  # indices 0 to 5
    assert len(val) == 3    # indices 6 to 8
    assert len(test) == 1   # index 9
    
    # Ensure no step overlaps
    train_steps = set(train['step'])
    val_steps = set(val['step'])
    test_steps = set(test['step'])
    
    assert train_steps.isdisjoint(val_steps)
    assert val_steps.isdisjoint(test_steps)
    assert train_steps.isdisjoint(test_steps)
    
    # Ensure chronologically sorted
    assert train['step'].max() < val['step'].min()
    assert val['step'].max() < test['step'].min()
