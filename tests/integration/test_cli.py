import os
import yaml
import pytest
import pandas as pd
from typer.testing import CliRunner
from fraudshield.cli import app

runner = CliRunner()

@pytest.fixture
def dummy_pipeline_env(tmp_path):
    # Create dummy raw data
    raw_path = tmp_path / "dummy_raw.csv"
    processed_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "artifacts"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'step': list(range(1, 101)),
        'type': ['PAYMENT', 'TRANSFER'] * 50,
        'amount': [100.0, 500.0] * 50,
        'nameOrig': ['A123'] * 100,
        'oldbalanceOrg': [1000.0, 500.0] * 50,
        'newbalanceOrig': [900.0, 0.0] * 50,
        'nameDest': ['M123'] * 100,
        'oldbalanceDest': [0.0, 100.0] * 50,
        'newbalanceDest': [0.0, 600.0] * 50,
        'isFraud': [0, 1] * 50,
        'isFlaggedFraud': [0, 0] * 50
    })
    df.to_csv(raw_path, index=False)
    
    # Create data config
    data_cfg = {
        'raw_data_path': str(raw_path),
        'processed_data_path': str(processed_dir),
        'report_path': str(report_dir),
        'target_column': 'isFraud',
        'temporal_column': 'step',
        'train_ratio': 0.6,
        'validation_ratio': 0.2,
        'test_ratio': 0.2,
        'random_seed': 42,
        'required_columns': list(df.columns)
    }
    data_cfg_path = tmp_path / "data.yaml"
    with open(data_cfg_path, "w") as f:
        yaml.dump(data_cfg, f)
        
    # Create baseline config
    baseline_cfg = {
        'random_seed': 42,
        'target_column': 'isFraud',
        'temporal_column': 'step',
        'model_artifact_dir': str(artifact_dir),
        'report_dir': str(report_dir),
        'data_config_path': str(data_cfg_path),
        'feature_sets': {
            'core': ['amount', 'oldbalanceOrg'],
            'engineered': ['amount', 'oldbalanceOrg']
        },
        'false_positive_cost': 1,
        'false_negative_cost': 100,
        'min_recall_targets': [0.8, 0.9]
    }
    baseline_cfg_path = tmp_path / "baseline.yaml"
    with open(baseline_cfg_path, "w") as f:
        yaml.dump(baseline_cfg, f)
        
    return {
        'data_config': str(data_cfg_path),
        'baseline_config': str(baseline_cfg_path),
        'artifact_dir': str(artifact_dir),
        'report_dir': str(report_dir)
    }

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "FraudShield ML CLI" in result.output

def test_cli_validate_missing_config():
    result = runner.invoke(app, ["validate-data"])
    assert result.exit_code != 0
    assert "Missing option" in result.output

def test_full_pipeline_cli(dummy_pipeline_env):
    data_cfg = dummy_pipeline_env['data_config']
    base_cfg = dummy_pipeline_env['baseline_config']
    
    # 1. Validate
    res = runner.invoke(app, ["validate-data", "--config", data_cfg])
    if res.exit_code != 0: print(res.output)
    assert res.exit_code == 0
    
    # 2. Build Features
    res = runner.invoke(app, ["build-features", "--config", data_cfg])
    if res.exit_code != 0: print(res.output)
    assert res.exit_code == 0
    
    # 3. Split Data
    res = runner.invoke(app, ["split-data", "--config", data_cfg])
    if res.exit_code != 0: print(res.output)
    assert res.exit_code == 0
    
    # 4. Train Baseline
    res = runner.invoke(app, ["train-baseline", "--config", base_cfg])
    if res.exit_code != 0: print(res.output)
    assert res.exit_code == 0
    
    # Verify outputs
    runs = os.listdir(dummy_pipeline_env['artifact_dir'])
    assert len(runs) == 1
    
    run_path = os.path.join(dummy_pipeline_env['artifact_dir'], runs[0])
    assert os.path.exists(os.path.join(run_path, "pipeline.joblib"))
    assert os.path.exists(os.path.join(run_path, "all_val_metrics.json"))
    assert os.path.exists(os.path.join(run_path, "test_metrics.json"))
    
    assert os.path.exists(os.path.join(dummy_pipeline_env['report_dir'], "baseline_results.md"))
