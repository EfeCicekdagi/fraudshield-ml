import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.threshold_analysis import analyze_thresholds, compute_cost

def test_compute_cost():
    assert compute_cost(fp=10, fn=5, fp_cost=1, fn_cost=100) == 510

def test_analyze_thresholds():
    # Synthetic predictions
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.35, 0.6, 0.8, 0.9])
    
    results = analyze_thresholds(y_true, y_prob, fp_cost=1, fn_cost=100)
    
    assert 'best_f1' in results
    assert 'min_cost' in results
    assert 'recall_80' in results
    
    # There should be exactly len(thresholds) elements in the arrays
    assert len(results['precisions']) == len(results['thresholds'])
    assert len(results['f1_scores']) == len(results['thresholds'])
    assert len(results['costs']) == len(results['thresholds'])
    
    # The minimum cost threshold should exist
    assert results['min_cost'] is not None
