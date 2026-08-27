import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fraudshield.models.evaluate import evaluate_model

def test_evaluate_model():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.8, 0.9, 0.4, 0.2])
    
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    # Check confusion matrix elements
    assert metrics['True Negative'] == 2
    assert metrics['False Positive'] == 1
    assert metrics['False Negative'] == 1
    assert metrics['True Positive'] == 1
    
    # Check alert rate (FP+TP) / Total = 2 / 5 = 0.4
    assert metrics['Fraud Alert Rate'] == 0.4
    assert metrics['Alerts per 1000'] == 400.0
    
    # Check basic bounds
    assert 0 <= metrics['PR-AUC'] <= 1
    assert 0 <= metrics['ROC-AUC'] <= 1
    assert 0 <= metrics['Precision'] <= 1
    assert 0 <= metrics['Recall'] <= 1
    assert 0 <= metrics['F1-score'] <= 1
