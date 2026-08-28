import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from fraudshield.models.lightgbm_model import calculate_scale_pos_weight, get_lightgbm_pipeline
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.cli import app
from fraudshield.schemas.config import LightGBMConfig

def test_calculate_scale_pos_weight():
    # 90 negative, 10 positive -> weight should be 9.0
    y = np.array([0]*90 + [1]*10)
    weight = calculate_scale_pos_weight(y)
    assert weight == 9.0

    # 0 positive -> should return 1.0 (fallback)
    y_zeros = np.array([0]*100)
    weight_zeros = calculate_scale_pos_weight(y_zeros)
    assert weight_zeros == 1.0

def test_get_lightgbm_pipeline():
    pipeline = get_lightgbm_pipeline(
        numeric_features=['amount', 'oldbalanceOrg'],
        categorical_features=['type'],
        params={'n_estimators': 10},
        is_weighted=True,
        scale_pos_weight=5.0
    )
    
    assert pipeline is not None
    assert 'preprocessor' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps
    
    classifier = pipeline.named_steps['classifier']
    assert classifier.n_estimators == 10
    assert classifier.scale_pos_weight == 5.0

def test_inner_temporal_split():
    # Create mock outer train data
    df = pd.DataFrame({
        'step': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 10 records, each different step
        'amount': np.random.rand(10)
    })
    
    # 80% train, 20% val
    train_df, val_df, test_df = temporal_train_val_test_split(df, 'step', 0.8, 0.2)
    
    assert len(test_df) == 0
    assert len(train_df) == 9 # indices 0 to 8
    assert len(val_df) == 1   # index 9
    
    # Verify temporal order
    assert train_df['step'].max() < val_df['step'].min()

@pytest.fixture
def mock_lightgbm_environment(tmp_path):
    # Setup dummy data and configs
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    
    # Create dummy train and val, NO TEST
    np.random.seed(42)
    n_samples = 100
    df_train = pd.DataFrame({
        'step': np.random.randint(1, 10, n_samples),
        'type': np.random.choice(['PAYMENT', 'TRANSFER'], n_samples),
        'amount': np.random.rand(n_samples) * 1000,
        'oldbalanceOrg': np.random.rand(n_samples) * 1000,
        'isFraud': np.random.randint(0, 2, n_samples)
    })
    df_train.sort_values('step', inplace=True)
    df_train.to_parquet(processed_dir / "train.parquet", index=False)
    
    df_val = pd.DataFrame({
        'step': np.random.randint(10, 15, 20),
        'type': np.random.choice(['PAYMENT', 'TRANSFER'], 20),
        'amount': np.random.rand(20) * 1000,
        'oldbalanceOrg': np.random.rand(20) * 1000,
        'isFraud': np.random.randint(0, 2, 20)
    })
    df_val.sort_values('step', inplace=True)
    df_val.to_parquet(processed_dir / "val.parquet", index=False)
    
    # We explicitly do NOT create test.parquet to ensure it's not needed/used
    
    # Configs
    data_config_path = tmp_path / "data.yaml"
    with open(data_config_path, "w") as f:
        f.write(f"""
raw_data_path: dummy
processed_data_path: {processed_dir}
report_path: dummy
target_column: isFraud
temporal_column: step
train_ratio: 0.7
validation_ratio: 0.15
test_ratio: 0.15
random_seed: 42
required_columns: []
        """)
        
    lightgbm_config_path = tmp_path / "lightgbm.yaml"
    run_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    
    with open(lightgbm_config_path, "w") as f:
        f.write(f"""
random_seed: 42
objective: binary
primary_metric: average_precision
n_trials: 1
tuning_max_rows: 50
inner_validation_ratio: 0.20
early_stopping_rounds: 5
data_config_path: {data_config_path}
model_artifact_dir: {run_dir}
report_dir: {report_dir}
feature_sets:
  core:
    - step
    - type
    - amount
    - oldbalanceOrg
false_positive_cost: 1.0
false_negative_cost: 10.0
min_recall_targets: [0.80]
param_distributions:
  learning_rate:
    type: loguniform
    low: 0.01
    high: 0.1
  n_estimators:
    type: categorical
    choices: [10]
  num_leaves:
    type: int
    low: 10
    high: 20
        """)
        
    return lightgbm_config_path, run_dir, report_dir

def test_train_lightgbm_cli(mock_lightgbm_environment):
    config_path, run_dir, report_dir = mock_lightgbm_environment
    
    runner = CliRunner()
    result = runner.invoke(app, ["train-lightgbm", "--config", str(config_path)])
    
    assert result.exit_code == 0
    
    # Check if artifacts were created
    assert run_dir.exists()
    
    # Find the specific run directory
    runs = [d for d in run_dir.iterdir() if d.is_dir()]
    assert len(runs) == 1
    
    specific_run_dir = runs[0]
    assert (specific_run_dir / "model.joblib").exists()
    assert (specific_run_dir / "best_params.json").exists()
    assert (specific_run_dir / "optuna_trials_LightGBM_unweighted_core.csv").exists()
    assert (specific_run_dir / "validation_metrics.json").exists()
    assert (specific_run_dir / "threshold.json").exists()
    assert (specific_run_dir / "feature_schema.json").exists()
    assert (specific_run_dir / "run_metadata.json").exists()
    assert (specific_run_dir / "figures" / "lightgbm" / "pr_curve.png").exists()
    
    # Check report
    assert (report_dir / "lightgbm_results.md").exists()
    assert (report_dir / "metrics" / "lightgbm_validation_metrics.json").exists()
