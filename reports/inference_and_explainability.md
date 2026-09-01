# Inference and Explainability Pipeline

This document summarizes the architecture, capabilities, and outputs of the Phase 7 Inference and Explainability pipeline for FraudShield ML.

## Architecture

The inference pipeline is designed to be lightweight, production-ready, and fully uncoupled from legacy Scikit-Learn Pickled artifacts (`.joblib`). By utilizing a **Version-Independent Inference Bundle**, we've eliminated the risk of breaking changes caused by Scikit-Learn version mismatches (`1.3` vs `1.9`) and avoided unsafe unpickling of arbitrary Python objects at runtime.

1.  **Data Validation:** Inbound requests are validated using Pydantic (`TransactionRequest`).
2.  **Feature Engineering:** Raw features are dynamically transformed (e.g., `hour_of_day`, `day_of_week`, `is_risky_type`).
3.  **Preprocessing:** Categorical and numerical features are standardized and one-hot encoded using a pure NumPy `StablePreprocessor`, which reads matrix weights directly from an `npz`/`json` bundle.
4.  **Model Inference:** The forward pass is computed via the PyTorch MLP model, returning raw logits.
5.  **Calibration:** The raw probability is calibrated using a pure NumPy `StableIsotonicCalibrator`, resolving dependency on Scikit-Learn's `CalibratedClassifierCV`.
6.  **Explainability (Optional):** If Captum is enabled, Integrated Gradients computes the attribution of each input feature compared to a reference baseline.
7.  **Reason Code Mapping:** Raw feature attributions are grouped into logical feature families.

## Inference Parity and Version Independence

A critical constraint in Phase 7 was guaranteeing strict parity while removing all private Scikit-Learn APIs from the inference runtime.

*   **Incompatibility Bypass:** Loading legacy models trained in Scikit-Learn `1.3` into newer `1.9` environments yields dangerous behavior and `InconsistentVersionWarning`. For security and reliability, we extracted the internals of both the `ColumnTransformer` (StandardScaler/OneHotEncoder) and `CalibratedClassifierCV` (IsotonicRegression) into a JSON/NPZ bundle via a one-time migration script.
*   **Pure NumPy Replacements:** `StablePreprocessor` and `StableIsotonicCalibrator` were developed to mimic Scikit-Learn's behavior down to `1e-6` precision using native Python and NumPy routines (like `np.interp`).
*   **Bundle Integrity Check:** The runtime enforces checksum verification upon loading the bundle. Corrupt or missing artifacts immediately halt the predictor.
*   **Parity Verification Results:** Extensive testing with synthetic adversarial cases (zero balance, huge amounts, unknown categories) yielded a maximum preprocessing and calibration parity difference of `< 1e-6`, validating that retraining was successfully avoided while maintaining exact operational boundaries.

## Performance Benchmark

The lightweight predictor was benchmarked with and without explainability enabled.

*   **Cold Load Time:** 13.53 ms (Drastic improvement due to removal of Scikit-Learn pickling)
*   **Warm Single Inference Latency:** ~7.60 ms (Without IG) / ~17.19 ms (With IG)
*   **Batch Throughput:** 
    * Without Explanations: 139.60 req/sec
    * With Explanations: 52.27 req/sec
*   **Memory Footprint:** ~360.69 MB

*Note: Integrated Gradients introduces measurable computational overhead. Real-time deployments may opt to disable IG by default (or run it asynchronously).*

## Explainability Details and Limitations

The pipeline uses **Integrated Gradients (IG)** for feature attribution.

*   **Baseline:** IG requires a baseline tensor, computed by aggregating the training data during the finalization phase (`reference_baseline.json`).
*   **Base Logit Attribution:** Attributions are calculated on the raw PyTorch logits *before* isotonic calibration. Isotonic regression is non-differentiable (step-wise piecewise constant), meaning gradients cannot flow backward through the calibrator. As a result, IG explains the model's raw internal belief (logit), not the final calibrated probability.
*   **Convergence Monitoring:** The predictor monitors the IG convergence delta to ensure attribution reliability. High deltas (>0.05) are logged as warnings, indicating that the integral approximation may be noisy for that specific transaction.
*   **Reason Codes:** Raw feature attributions are aggregated into distinct behavioral risks. See `reason_code_dictionary.md` for definitions.

## Usage Examples

The pipeline exposes a command-line interface (CLI) for both single and batch inference.

### Single Inference

```bash
python src/fraudshield/cli.py predict -i scratch/test_tx.json
```

**Output:**
```json
{
  "fraud_score": -1.178,
  "calibrated_probability": 0.6315,
  "risk_level": "CRITICAL",
  "is_alert": true,
  "decision_threshold": 0.0209,
  "reason_codes": [
    "RISKY_TRANSACTION_TYPE",
    "UNUSUAL_TRANSACTION_TIME",
    "LOW_SOURCE_BALANCE"
  ],
  "top_contributors": [
    {
      "feature": "oldbalanceOrg",
      "display_name": "Oldbalanceorg",
      "value": 300000.0,
      "attribution": 5.8835,
      "direction": "increases_risk",
      "reason_code": "LOW_SOURCE_BALANCE"
    }
  ],
  "model_version": "1.0.0",
  "inference_id": "bd5d43d8-8c98-496e-acce-00c51246152a",
  "scored_at": "2026-08-31T10:28:29.869703Z"
}
```

### Batch Inference

```bash
python src/fraudshield/cli.py predict-batch -i scratch/test_batch.csv -o scratch/test_out.csv
```
