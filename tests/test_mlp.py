import pytest
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typer.testing import CliRunner
from fraudshield.cli import app
from fraudshield.models.mlp import FraudMLP
from fraudshield.models.losses import FocalLoss, calculate_pos_weight
from fraudshield.models.torch_dataset import FraudDataset, create_dataloader

def test_fraud_mlp_architecture():
    # Test FraudMLP initialization and forward pass
    input_dim = 10
    hidden_dims = [64, 32]
    batch_size = 16
    
    model = FraudMLP(input_dim=input_dim, hidden_dimensions=hidden_dims)
    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    
    assert logits.shape == (batch_size,)
    # Output should be raw logits, not bounded by sigmoid (can be > 0 or < 0)
    assert not torch.all((logits >= 0.0) & (logits <= 1.0))

def test_focal_loss():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    logits = torch.tensor([0.5, -0.5, 2.0, -2.0])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0

def test_pos_weight():
    y = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 2 pos, 6 neg
    pw = calculate_pos_weight(y)
    assert pw.item() == 3.0

def test_torch_dataset():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    
    dataset = FraudDataset(X, y)
    assert len(dataset) == 10
    
    x_item, y_item = dataset[0]
    assert x_item.shape == (5,)
    assert y_item.dim() == 0
    assert x_item.dtype == torch.float32
    assert y_item.dtype == torch.float32
    
@pytest.fixture
def mock_mlp_environment(tmp_path):
    """Sets up a mock environment with dummy data and config for MLP pipeline integration testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    artifacts_dir = tmp_path / "artifacts"
    
    raw_dir.mkdir()
    processed_dir.mkdir()
    report_dir.mkdir()
    artifacts_dir.mkdir()
    
    data_config_path = tmp_path / "data.yaml"
    mlp_config_path = tmp_path / "mlp.yaml"
    
    # Create dummy data with chronological 'step'
    num_samples = 100
    df = pd.DataFrame({
        'step': np.arange(1, num_samples + 1),
        'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], num_samples),
        'amount': np.random.rand(num_samples) * 1000,
        'oldbalanceOrg': np.random.rand(num_samples) * 5000,
        'newbalanceOrig': np.random.rand(num_samples) * 5000,
        'oldbalanceDest': np.random.rand(num_samples) * 5000,
        'newbalanceDest': np.random.rand(num_samples) * 5000,
        'isFraud': np.random.choice([0, 1], p=[0.9, 0.1], size=num_samples)
    })
    
    # Ensure at least some positive class in all splits
    # inner_train (0-63), inner_val (64-79), outer_val (80-99)
    df.loc[10:15, 'isFraud'] = 1
    df.loc[70:75, 'isFraud'] = 1
    df.loc[90:95, 'isFraud'] = 1
    
    # 80/20 train/val split for outer split
    train_df = df.iloc[:80]
    val_df = df.iloc[80:100]
    
    train_df.to_parquet(processed_dir / "train.parquet")
    val_df.to_parquet(processed_dir / "val.parquet")
    
    data_config = {
        "raw_data_path": str(raw_dir / "raw.csv"),
        "processed_data_path": str(processed_dir),
        "report_path": str(report_dir),
        "target_column": "isFraud",
        "temporal_column": "step",
        "train_ratio": 0.7,
        "validation_ratio": 0.2,
        "test_ratio": 0.1,
        "random_seed": 42,
        "required_columns": ["step", "type", "amount", "oldbalanceOrg", "isFraud"]
    }
    
    with open(data_config_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(data_config, f)
        
    mlp_config = {
        "random_seed": 42,
        "device": "cpu",
        "batch_size": 16,
        "num_workers": 0,
        "max_epochs": 2,
        "patience": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "hidden_dimensions": [16],
        "dropout": 0.0,
        "activation": "relu",
        "batch_norm": False,
        "gradient_clip_norm": 1.0,
        "loss_type": "weighted_bce",
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "tuning_trials": 2,
        "tuning_max_rows": 100,
        "inner_validation_ratio": 0.2,
        "data_config_path": str(data_config_path),
        "model_artifact_dir": str(artifacts_dir),
        "report_dir": str(report_dir),
        "false_positive_cost": 1.0,
        "false_negative_cost": 10.0,
        "min_recall_targets": [0.8],
        "feature_sets": {
            "core": ["step", "type", "amount", "oldbalanceOrg"]
        }
    }
    
    with open(mlp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(mlp_config, f)
        
    return mlp_config_path, artifacts_dir, report_dir

def test_train_mlp_cli(mock_mlp_environment):
    config_path, run_dir, report_dir = mock_mlp_environment
    
    runner = CliRunner()
    result = runner.invoke(app, ["train-mlp", "--config", str(config_path)])
    
    assert result.exit_code == 0
    
    # Check if artifacts were created
    assert run_dir.exists()
    
    runs = [d for d in run_dir.iterdir() if d.is_dir()]
    assert len(runs) == 1
    
    specific_run_dir = runs[0]
    assert (specific_run_dir / "model_state_dict.pt").exists()
    assert (specific_run_dir / "preprocessor.joblib").exists()
    assert (specific_run_dir / "best_params.json").exists()
    assert (specific_run_dir / "config_snapshot.json").exists()
    assert (specific_run_dir / "threshold.json").exists()
    assert (specific_run_dir / "feature_schema.json").exists()
    assert (specific_run_dir / "training_history.json").exists()
    
    # Check reports
    assert (report_dir / "mlp_results.md").exists()
    assert (report_dir / "metrics" / "mlp_validation_metrics.json").exists()
    
    # Check figures
    figures_dir = report_dir / "figures" / "mlp"
    assert figures_dir.exists()
    assert list(figures_dir.glob("*.png"))
