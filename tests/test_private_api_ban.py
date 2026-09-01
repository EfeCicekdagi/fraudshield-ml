import os
import glob
import pytest

BANNED_STRINGS = [
    ".calibrators[",
    "_get_response_values",
    "CalibratedClassifierCV",
    "IsotonicRegression",
    "joblib.load",
    ".predict_proba",
]

def test_no_private_sklearn_api_in_inference():
    inference_dir = os.path.join("src", "fraudshield", "inference")
    python_files = glob.glob(os.path.join(inference_dir, "**", "*.py"), recursive=True)
    
    for py_file in python_files:
        with open(py_file, "r") as f:
            content = f.read()
            for banned in BANNED_STRINGS:
                # We do not ban these strings in comments but for simplicity, we ban them globally in the file.
                # If they appear, the test fails.
                assert banned not in content, f"Banned string '{banned}' found in {py_file}!"

    print("All inference runtime files are clean from Scikit-Learn private APIs and joblib.")

def test_corrupt_bundle_rejection(tmp_path):
    """
    Tests that ModelEnvironment fails to initialize if the bundle is corrupt (checksum mismatch).
    """
    import json
    import hashlib
    from fraudshield.inference.model_loader import ModelEnvironment, get_model_env
    
    # Create a mock bundle
    bundle_dir = tmp_path / "inference_bundle"
    bundle_dir.mkdir()
    
    # Write a dummy manifest
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text('{"version": "1.0"}')
    
    # Calculate checksum for original manifest
    h = hashlib.sha256()
    h.update(b'{"version": "1.0"}')
    original_hash = h.hexdigest()
    
    # Let's create all required files
    required_files = [
        "feature_schema.json", "threshold.json", "risk_levels.json", 
        "preprocessor.json", "preprocessor_arrays.npz", "calibrator.json", "model_state_dict.pt"
    ]
    for rf in required_files:
        (bundle_dir / rf).write_text("dummy")
        
    # Update checksums
    checksums = {"manifest.json": original_hash}
    for rf in required_files:
        h = hashlib.sha256()
        h.update(b'dummy')
        checksums[rf] = h.hexdigest()
        
    checksums_path = bundle_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksums))
    
    # Corrupt the manifest
    manifest_path.write_text('{"version": "2.0"}') # Changed!
    
    ModelEnvironment._instance = None # Reset singleton
    with pytest.raises(RuntimeError, match="Corrupt bundle! Checksum mismatch"):
        get_model_env(str(tmp_path))
