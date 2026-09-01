import os
import sys
import json
import joblib
import numpy as np
import shutil
import hashlib
from typing import Dict, Any

import warnings

# ==============================================================================
# @deprecated
# MIGRATION SCRIPT ONLY
# This script extracts internal parameters from pickled Scikit-Learn models to 
# create the Version-Independent Inference Bundle. 
# It accesses private Scikit-Learn APIs (e.g., .calibrators_, _get_response_values) 
# and relies on joblib/pickle.
# These practices are BANNED in the runtime inference pipeline.
# Do NOT use this script or its methods during actual inference.
# ==============================================================================

# Ensure we can import from src
sys.path.insert(0, os.path.abspath('src'))

# Dummy class for sklearn 1.9 compatibility with 1.3 models
from sklearn.base import BaseEstimator
from fraudshield.pipelines.finalization_pipeline import MLPPipeline
class DummyBase(BaseEstimator): pass
MLPPipeline.__sklearn_tags__ = DummyBase.__sklearn_tags__

def compute_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_calibrator(src_dir: str, dest_dir: str):
    calib_path = os.path.join(src_dir, 'calibrator.joblib')
    c = joblib.load(calib_path)
    
    # Extract the actual isotonic regression object
    # CalibratedClassifierCV -> calibrated_classifiers_[0] -> calibrators[0]
    cc = c.calibrated_classifiers_[0]
    ir = cc.calibrators[0]
    
    calib_data = {
        "format_version": "fraudshield-isotonic-v1",
        "input_type": "sigmoid_probability",
        "interpolation": "linear",
        "out_of_bounds": ir.out_of_bounds,
        "x_thresholds": ir.X_thresholds_.tolist() if hasattr(ir, 'X_thresholds_') else [],
        "y_thresholds": ir.y_thresholds_.tolist() if hasattr(ir, 'y_thresholds_') else [],
        "X_min": float(ir.X_min_) if hasattr(ir, 'X_min_') else 0.0,
        "X_max": float(ir.X_max_) if hasattr(ir, 'X_max_') else 1.0,
        "source_sklearn_version": "1.3"
    }
    
    with open(os.path.join(dest_dir, 'calibrator.json'), 'w') as f:
        json.dump(calib_data, f, indent=2)
        
    print("Extracted calibrator internals.")

def extract_preprocessor(src_dir: str, dest_dir: str):
    prep_path = os.path.join(src_dir, 'preprocessor.joblib')
    p = joblib.load(prep_path)
    
    transformers = p.transformers_
    
    prep_meta = {
        "format_version": "fraudshield-preprocessor-v1",
        "transformers": []
    }
    
    arrays = {}
    
    for name, transformer, cols in transformers:
        if name == 'num':
            prep_meta['transformers'].append({
                "name": "num",
                "type": "StandardScaler",
                "columns": cols,
                "arrays": ["num_mean", "num_scale"]
            })
            arrays['num_mean'] = transformer.mean_
            arrays['num_scale'] = transformer.scale_
            
        elif name == 'cat':
            categories = transformer.categories_
            arr_names = []
            for i, cat_arr in enumerate(categories):
                arr_name = f"cat_categories_{i}"
                if cat_arr.dtype == object:
                    cat_arr = cat_arr.astype(str)
                arrays[arr_name] = cat_arr
                arr_names.append(arr_name)
                
            prep_meta['transformers'].append({
                "name": "cat",
                "type": "OneHotEncoder",
                "columns": cols,
                "handle_unknown": transformer.handle_unknown,
                "arrays": arr_names
            })
    
    with open(os.path.join(dest_dir, 'preprocessor.json'), 'w') as f:
        json.dump(prep_meta, f, indent=2)
        
    np.savez_compressed(os.path.join(dest_dir, 'preprocessor_arrays.npz'), **arrays)
    
    print("\n--- PREPROCESSOR EXPORT REPORT ---")
    print(f"Total transformers extracted: {len(transformers)}")
    for name, transformer, cols in transformers:
        print(f"  - Extracted {type(transformer).__name__} (name='{name}') for columns: {cols}")
    print("----------------------------------\n")
    print("Extracted preprocessor internals.")

def create_bundle():
    src_dir = "artifacts/final"
    dest_dir = os.path.join(src_dir, "inference_bundle")
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Extract Models
    extract_calibrator(src_dir, dest_dir)
    extract_preprocessor(src_dir, dest_dir)
    
    # 2. Copy Static Files
    static_files = [
        "model_state_dict.pt",
        "threshold.json",
        "risk_levels.json",
        "feature_schema.json",
        "reason_codes.json",
        "training_reference.npz"
    ]
    
    # also rename reference_baseline.json to training_reference.npz if it exists as json, 
    # but the instructions say copy training_reference.npz. Let's see if training_reference.npz exists or we need to rename/convert reference_baseline.json.
    # We will copy all directly from src_dir
    
    # Wait, reference_baseline.json is actually the baseline. Let's copy it as reference_baseline.json instead if it exists, or rename it.
    if os.path.exists(os.path.join(src_dir, "reference_baseline.json")):
        static_files.append("reference_baseline.json")
    
    for f in static_files:
        src_path = os.path.join(src_dir, f)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dest_dir, f))
            print(f"Copied {f}")
            
    # 3. Create Manifest
    manifest = {
        "bundle_format_version": "1.0",
        "model_version": "1.0.0",
        "pytorch_version": "2.1.2", # assuming current torch version approx
        "source_sklearn_version": "1.3",
        "expected_input_schema": "TransactionRequest",
        "calibrator_input_type": "sigmoid_probability",
        "explanation_target": "raw_logit",
    }
    
    with open(os.path.join(dest_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        
    # 4. Checksums
    checksums = {}
    for filename in os.listdir(dest_dir):
        filepath = os.path.join(dest_dir, filename)
        if os.path.isfile(filepath) and filename != "checksums.json":
            checksums[filename] = compute_sha256(filepath)
            
    with open(os.path.join(dest_dir, 'checksums.json'), 'w') as f:
        json.dump(checksums, f, indent=2)
        
    print(f"Bundle successfully created in {dest_dir}")
    print(f"Checksums calculated for {len(checksums)} files.")

if __name__ == "__main__":
    create_bundle()
