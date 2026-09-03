import os
import json
import torch
import numpy as np
import hashlib
from pathlib import Path

def main():
    bundle_dir = Path("artifacts/ci_bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. manifest.json
    manifest = {
        "model_type": "Dummy_MLP",
        "version": "ci",
        "calibrator_input_type": "sigmoid_probability"
    }
    with open(bundle_dir / "manifest.json", "w") as f: json.dump(manifest, f)
    
    # 2. feature_schema.json
    feature_schema = {"features": ["feature_0", "feature_1", "feature_2"]}
    with open(bundle_dir / "feature_schema.json", "w") as f: json.dump(feature_schema, f)
    
    # 3. threshold.json
    threshold = {"operational_threshold": 0.5}
    with open(bundle_dir / "threshold.json", "w") as f: json.dump(threshold, f)
    
    # 4. risk_levels.json
    risk_levels = {"LOW_MAX": 0.3, "MEDIUM_MAX": 0.7, "HIGH_MAX": 1.0}
    with open(bundle_dir / "risk_levels.json", "w") as f: json.dump(risk_levels, f)
    
    # 5. preprocessor.json
    preprocessor = {
        "features": ["feature_0", "feature_1", "feature_2"],
        "means_shape": [3],
        "scales_shape": [3]
    }
    with open(bundle_dir / "preprocessor.json", "w") as f: json.dump(preprocessor, f)
    
    # 6. preprocessor_arrays.npz
    np.savez(bundle_dir / "preprocessor_arrays.npz", means=np.zeros(3), scales=np.ones(3))
    
    # 7. calibrator.json
    calibrator = {
        "X_min_": 0.0,
        "X_max_": 1.0,
        "X_thresholds_": [0.0, 1.0],
        "y_step_": [0.0, 1.0]
    }
    with open(bundle_dir / "calibrator.json", "w") as f: json.dump(calibrator, f)
    
    # 8. model_state_dict.pt
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(3, 2),
        nn.BatchNorm1d(2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(2, 1)
    )
    torch.save(model.state_dict(), bundle_dir / "model_state_dict.pt")
    
    # 9. checksums.json
    expected_files = [
        "manifest.json", "feature_schema.json", "threshold.json",
        "risk_levels.json", "preprocessor.json", "preprocessor_arrays.npz",
        "calibrator.json", "model_state_dict.pt"
    ]
    checksums = {}
    for filename in expected_files:
        filepath = bundle_dir / filename
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        checksums[filename] = sha256_hash.hexdigest()
        
    with open(bundle_dir / "checksums.json", "w") as f: json.dump(checksums, f)
    
    # Reference baseline
    baseline = {"feature_0": 0.0, "feature_1": 0.0, "feature_2": 0.0}
    with open(bundle_dir / "reference_baseline.json", "w") as f: json.dump(baseline, f)
    
    print(f"Synthetic CI bundle generated at {bundle_dir}")

if __name__ == "__main__":
    main()
