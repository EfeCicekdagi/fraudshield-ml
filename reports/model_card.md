# Model Card: FraudShield ML (v1.0.0)

## Model Details
- **Algorithm:** MLP_Weighted
- **Calibration:** Isotonic (empirically calibrated probability estimate, calibrated on a temporally separated calibration split. Calibration validity depends on future data resembling the calibration distribution)
- **Features Used:** point-in-time-safe (`pre_transaction`)

## Intended Use
- **Primary Use Case:** Real-time pre-transaction fraud scoring.
- **Out-of-Scope:** Post-transaction reconciliation.

## Known Limitations and Warnings
- **Synthetic Data:** Trained entirely on PaySim, a simulated dataset.
- **Artifact Exclusion:** Highly predictive balance errors were removed to ensure generalization. The `Full Engineered` model was downgraded to a benchmark artifact.
- **Real-World Validation Required:** The hypothetical costs and behavior do not exactly mirror a live bank.
- **Test Set Usage:** The test set was previously used for the Phase 3 baseline benchmark. Therefore, the results are not a completely independent and pristine holdout prediction.

## Performance on Test Set
- **F1 Score:** 0.5662
- **Recall:** 0.7574