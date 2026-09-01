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

## Environment and Portability
- **Training Environment:** Scikit-Learn 1.3 (pickled artifact generation)
- **Inference Environment:** NumPy + PyTorch + Pydantic (Scikit-Learn decoupled bundle)
- **Artifact Portability Approach:** Due to `InconsistentVersionWarning` issues with Scikit-Learn across environments (e.g., changes to internal properties like `_get_response_values` between v1.3 and v1.9), the inference deployment explicitly extracts the internal mathematical state of the `StandardScaler`, `OneHotEncoder`, and `IsotonicRegression` models into a generic `JSON` and `.npz` bundle.
- **Legacy Pickle Limitations:** Loading model artifacts (like `.joblib` files) from unknown sources is generally unsafe as unpickling can execute arbitrary code. The shift to a declarative JSON/NPZ bundle strategy provides strict sandboxing, enabling safe deployment to zero-trust endpoints without the overhead of the Scikit-Learn execution engine.