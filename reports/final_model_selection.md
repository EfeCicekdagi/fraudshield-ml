# Final Model Selection

| Candidate | PR-AUC | F1 | Time (s) |
|-----------|--------|----|----------|
| SGD | 0.0461 | 0.0163 | 17.1 |
| LGBM_Unweighted | 0.3545 | 0.4338 | 11.9 |
| LGBM_Weighted | 0.5339 | 0.2493 | 13.2 |
| MLP_Weighted | 0.6637 | 0.6065 | 580.4 |

**Selected Champion:** `MLP_Weighted`
**Calibration Method:** `Isotonic`

## Calibration Metrics
- Final Brier score: 0.00208
- Brier score (before): *Missing*
- Log loss (before/after): *Missing*
- ECE (before/after): *Missing*
- Calibration split step range: *Missing*
- Decision split step range: *Missing*

## Risk Boundaries
- LOW: <= 0.1000
- MEDIUM: 0.1000 < score <= 0.3000
- HIGH: 0.3000 < score <= 0.5000
- CRITICAL: > 0.5000

*(Note: Transaction counts and empirical fraud rates per bucket were not saved as artifacts and test set rescoring is forbidden.)*