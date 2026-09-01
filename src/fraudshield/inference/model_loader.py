import os
import json
import torch
import hashlib
from pathlib import Path

from fraudshield.inference.stable_components import StableIsotonicCalibrator, StablePreprocessor

from fraudshield.models.mlp import FraudMLP
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

class ModelEnvironment:
    """
    Singleton-like loader for the FraudShield ML Inference Pipeline.
    Loads and caches all necessary artifacts upon first initialization.
    """
    _instance = None

    def __new__(cls, artifacts_dir: str = "artifacts/final"):
        if cls._instance is None:
            logger.info("Initializing Inference Environment...")
            cls._instance = super(ModelEnvironment, cls).__new__(cls)
            cls._instance._load_artifacts(artifacts_dir)
        return cls._instance

    def _load_artifacts(self, artifacts_dir: str):
        # We enforce using the inference_bundle for runtime safety
        base_path = Path(artifacts_dir) / "inference_bundle"
        
        if not base_path.exists():
            raise FileNotFoundError(f"Inference bundle directory not found at {base_path}")
            
        # Define expected files
        expected_files = {
            "manifest": "manifest.json",
            "feature_schema": "feature_schema.json",
            "threshold": "threshold.json",
            "risk_levels": "risk_levels.json",
            "preprocessor": "preprocessor.json",
            "preprocessor_arrays": "preprocessor_arrays.npz",
            "calibrator": "calibrator.json",
            "model_state": "model_state_dict.pt",
            "checksums": "checksums.json"
        }
        
        # Verify existence
        for key, filename in expected_files.items():
            filepath = base_path / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required artifact: {filepath}")
                
        # Validate Checksums
        with open(base_path / expected_files["checksums"]) as f:
            checksums = json.load(f)
            
        for filename, expected_hash in checksums.items():
            filepath = base_path / filename
            if not filepath.exists():
                continue # Handled by existence check
                
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
                    
            if sha256_hash.hexdigest() != expected_hash:
                raise RuntimeError(f"Corrupt bundle! Checksum mismatch for {filename}")
                
        # Load JSONs
        with open(base_path / expected_files["manifest"]) as f:
            self.metadata = json.load(f)
            
        with open(base_path / expected_files["feature_schema"]) as f:
            self.feature_schema = json.load(f)["features"]
            
        with open(base_path / expected_files["threshold"]) as f:
            self.threshold_data = json.load(f)
            self.operational_threshold = self.threshold_data["operational_threshold"]
            
        with open(base_path / expected_files["risk_levels"]) as f:
            self.risk_levels = json.load(f)
            
        # Load Stable Components
        with open(base_path / expected_files["calibrator"]) as f:
            calib_meta = json.load(f)
            self.calibrator = StableIsotonicCalibrator(calib_meta)
            
        with open(base_path / expected_files["preprocessor"]) as f:
            prep_meta = json.load(f)
        
        import numpy as np
        prep_arrays = np.load(base_path / expected_files["preprocessor_arrays"], allow_pickle=False)
        self.preprocessor = StablePreprocessor(prep_meta, prep_arrays)
        
        # Validation checks
        if not (0 <= self.operational_threshold <= 1):
            raise ValueError("Operational threshold is out of [0,1] bounds.")
            
        boundaries = [self.risk_levels["LOW_MAX"], self.risk_levels["MEDIUM_MAX"], self.risk_levels["HIGH_MAX"]]
        if boundaries != sorted(boundaries):
            raise ValueError("Risk boundaries are not monotonically ordered.")
            
        # Infer MLP Architecture from state_dict
        state_dict = torch.load(base_path / expected_files["model_state"], map_location='cpu', weights_only=True)
        
        # We know it's a sequence of Linear -> BatchNorm -> GELU/ReLU -> Dropout
        # We can extract the hidden dimensions by looking at the weights of the Linear layers.
        # network.0.weight -> shape: (hidden_dim_1, input_dim)
        # network.4.weight -> shape: (hidden_dim_2, hidden_dim_1) etc.
        linear_layers = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 2]
        
        if not linear_layers:
            raise RuntimeError("Invalid model_state_dict. No linear layers found.")
            
        # The input_dim is the second dimension of the first Linear layer
        input_dim = state_dict[linear_layers[0]].shape[1]
        
        # Hidden dimensions are the first dimensions of all Linear layers EXCEPT the very last one
        hidden_dims = [state_dict[k].shape[0] for k in linear_layers[:-1]]
        
        logger.info(f"Inferred MLP Architecture - Input: {input_dim}, Hiddens: {hidden_dims}")
        
        # Instantiate and load weights
        # We must use dropout > 0 so the nn.Dropout layers are added to the Sequential,
        # otherwise the state_dict layer indices (0, 1, 2, 3...) will not match!
        self.model = FraudMLP(
            input_dim=input_dim,
            hidden_dimensions=hidden_dims,
            dropout=0.1  # Value doesn't matter for inference due to .eval(), but must be > 0
        )
        self.model.load_state_dict(state_dict)
        self.model.eval() # Set to evaluation mode!
        
        logger.info("Inference Environment Successfully Initialized.")

def get_model_env(artifacts_dir: str = "artifacts/final") -> ModelEnvironment:
    return ModelEnvironment(artifacts_dir)
