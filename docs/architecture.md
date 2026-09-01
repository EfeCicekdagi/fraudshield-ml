# FraudShield ML - Inference Architecture

## Overview
The Inference Architecture of FraudShield ML is designed around safety, speed, and strict environment independence. The pipeline is split into a heavily dependent Training Environment (Phase 6) and a minimalist, secure Inference Environment (Phase 7).

## Scikit-Learn Version Incompatibility
During the transition from Training (Scikit-Learn 1.3) to Inference (Scikit-Learn 1.9+), severe compatibility issues arise with pickled components, particularly:
1. `CalibratedClassifierCV`
2. `StandardScaler`
3. `OneHotEncoder`

Loading legacy `.joblib` files triggers an `InconsistentVersionWarning`. For components like `CalibratedClassifierCV`, changes to internal methods (`_get_response_values` returning a 2D array instead of 1D) lead to silent mathematical failures during probability calibration.

## Private Calibrator Access Removal
In earlier iterations, the inference module attempted to bypass wrapper bugs by manually accessing private attributes such as `.calibrators[0]` and `_get_response_values`. However, this bound the application tightly to the internal implementation details of the active Scikit-Learn version, risking critical failures when deploying to arbitrary cloud environments. All such private API usage has been **banned**.

## The Version-Independent Inference Bundle
To resolve both the incompatibility and the security risks associated with loading arbitrary pickled Python objects, the inference engine relies on a custom **Version-Independent Inference Bundle**. 

1. **Extraction:** A migration script reads the legacy `joblib` artifacts once, extracting only the raw mathematical parameters (e.g., `mean`, `scale`, `categories`, `x_thresholds`, `y_thresholds`).
2. **Stable Components:** The parameters are packaged into a purely declarative JSON and NPZ archive. At runtime, pure NumPy/Python classes (`StablePreprocessor`, `StableIsotonicCalibrator`) consume these arrays to execute transforms and piecewise interpolations.
3. **Checksum Enforcement:** To prevent tampering or accidental corruption, a strict SHA-256 manifest is verified before the predictor initializes.

## Parity Results
The stable implementations have been rigorously benchmarked against the legacy pipeline:
- **Calibrator Grid Parity (10,001 points):** < 1e-6 error tolerance.
- **Preprocessor Transform Parity:** < 1e-6 error tolerance across diverse synthetic datasets including edge cases (zero balances, huge amounts, unknown categorical types).

## Known Limitations
- Explainability (Integrated Gradients via Captum) requires running 50+ backward/forward passes per transaction, dropping batch throughput from ~140 req/sec to ~50 req/sec. It should be selectively enabled.
- The `StablePreprocessor` strictly replicates the extracted legacy behavior; if a feature drift occurs that necessitates retraining, a new bundle must be generated from the new Scikit-Learn artifacts.
