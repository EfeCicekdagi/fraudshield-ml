import typer
import pandas as pd
from pathlib import Path

from fraudshield.schemas.config import DataConfig
from fraudshield.logging_config import setup_logger
from fraudshield.data.loader import load_raw_data
from fraudshield.data.validation import validate_raw_data
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.features.builder import build_all_features

app = typer.Typer(help="FraudShield ML CLI")
logger = setup_logger(__name__)

@app.command()
def validate_data(config: str = typer.Option(..., help="Path to the config yaml file")):
    """Validate the raw dataset."""
    try:
        conf = DataConfig.load_from_yaml(config)
        df = load_raw_data(conf.raw_data_path)
        
        results = validate_raw_data(df, required_columns=conf.required_columns)
        
        if not results.get('has_all_required_columns', False):
            logger.error("Validation failed: Missing required columns.")
            raise typer.Exit(code=1)
            
        logger.info("Validation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise typer.Exit(code=1)

@app.command()
def build_features(config: str = typer.Option(..., help="Path to the config yaml file")):
    """Build features from the raw dataset and save to processed data path."""
    try:
        conf = DataConfig.load_from_yaml(config)
        df = load_raw_data(conf.raw_data_path)
        
        df_features = build_all_features(df)
        
        out_path = Path(conf.processed_data_path)
        out_path.mkdir(parents=True, exist_ok=True)
        
        out_file = out_path / "features.parquet"
        logger.info(f"Saving features to {out_file}")
        
        # Save as parquet for efficiency
        df_features.to_parquet(out_file, index=False)
        logger.info("Features built and saved successfully.")
        
    except Exception as e:
        logger.error(f"Error during feature building: {e}")
        raise typer.Exit(code=1)

@app.command()
def split_data(config: str = typer.Option(..., help="Path to the config yaml file")):
    """Split the feature dataset into train, validation, and test sets."""
    try:
        conf = DataConfig.load_from_yaml(config)
        
        feature_file = Path(conf.processed_data_path) / "features.parquet"
        if not feature_file.exists():
            logger.error(f"Features file not found at {feature_file}. Run build-features first.")
            raise typer.Exit(code=1)
            
        logger.info(f"Loading features from {feature_file}")
        df_features = pd.read_parquet(feature_file)
        
        train_df, val_df, test_df = temporal_train_val_test_split(
            df_features,
            temporal_column=conf.temporal_column,
            train_size=conf.train_ratio,
            val_size=conf.validation_ratio
        )
        
        out_path = Path(conf.processed_data_path)
        logger.info(f"Saving splits to {out_path}")
        
        train_df.to_parquet(out_path / "train.parquet", index=False)
        val_df.to_parquet(out_path / "val.parquet", index=False)
        test_df.to_parquet(out_path / "test.parquet", index=False)
        
        logger.info("Data split and saved successfully.")
        
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise typer.Exit(code=1)

@app.command()
def train_baseline(
    config: Path = typer.Option(..., "--config", "-c", help="Path to baseline config file (e.g., configs/baseline.yaml)")
):
    """Train baseline models and evaluate thresholds."""
    from fraudshield.pipelines.training_pipeline import run_baseline_pipeline
    
    logger.info("Starting baseline training pipeline...")
    run_baseline_pipeline(str(config))
    logger.info("Baseline training pipeline finished.")

@app.command()
def train_lightgbm(
    config: Path = typer.Option(..., "--config", "-c", help="Path to LightGBM config file (e.g., configs/lightgbm.yaml)")
):
    """Train LightGBM models with hyperparameter optimization."""
    from fraudshield.pipelines.lightgbm_pipeline import run_lightgbm_pipeline
    
    logger.info("Starting LightGBM training pipeline...")
    run_lightgbm_pipeline(str(config))
    logger.info("LightGBM training pipeline finished.")

@app.command()
def train_mlp(
    config: Path = typer.Option(
        "configs/mlp.yaml",
        help="Path to the MLP training config file"
    )
):
    """
    Trains PyTorch MLP Deep Learning models and hyperparameter tuning using Optuna.
    """
    from fraudshield.pipelines.mlp_pipeline import run_mlp_pipeline
    
    logger.info("Starting MLP training pipeline...")
    run_mlp_pipeline(str(config))
    logger.info("MLP training pipeline finished.")

@app.command()
def finalize_model(
    config: Path = typer.Option(
        "configs/final_model.yaml",
        help="Path to the finalization config file"
    )
):
    """
    Executes the final model selection, calibration, thresholding, and locked test evaluation.
    """
    from fraudshield.pipelines.finalization_pipeline import run_finalization_pipeline
    
    logger.info("Starting Finalization pipeline...")
    run_finalization_pipeline(str(config))
    logger.info("Finalization pipeline finished.")

@app.command()
def predict(
    input_file: Path = typer.Option(..., "--input", "-i", help="Path to input JSON transaction"),
    config: Path = typer.Option("configs/inference.yaml", help="Path to inference config")
):
    """Run real-time inference on a single transaction JSON."""
    import json
    from fraudshield.inference.predictor import FraudPredictor
    from fraudshield.inference.schemas import TransactionRequest
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    predictor = FraudPredictor(str(config))
    req = TransactionRequest(**data)
    res = predictor.predict_single(req)
    
    print(json.dumps(res.model_dump(), indent=2))

@app.command()
def predict_batch(
    input_file: Path = typer.Option(..., "--input", "-i", help="Path to input CSV transactions"),
    output_file: Path = typer.Option(..., "--output", "-o", help="Path to output CSV predictions"),
    config: Path = typer.Option("configs/inference.yaml", help="Path to inference config")
):
    """Run batch inference on a CSV file."""
    import pandas as pd
    from fraudshield.inference.predictor import FraudPredictor
    from fraudshield.inference.schemas import BatchTransactionRequest, TransactionRequest
    
    logger.info(f"Loading batch data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Convert DF to list of requests
    reqs = []
    for record in df.to_dict("records"):
        reqs.append(TransactionRequest(**record))
        
    batch_req = BatchTransactionRequest(transactions=reqs)
    predictor = FraudPredictor(str(config))
    
    import time
    t0 = time.time()
    results = predictor.predict_batch(batch_req)
    t1 = time.time()
    
    logger.info(f"Batch processed {len(results)} rows in {t1-t0:.2f} seconds.")
    
    # Save to CSV
    res_df = pd.DataFrame([r.model_dump() if not isinstance(r, dict) else r for r in results])
    res_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

@app.command()
def model_info(
    config: Path = typer.Option("configs/inference.yaml", help="Path to inference config")
):
    """Display loaded model metadata and policy rules."""
    from fraudshield.inference.predictor import FraudPredictor
    
    predictor = FraudPredictor(str(config))
    env = predictor.env
    
    print("=== FraudShield Model Info ===")
    print(f"Model Version: {predictor.model_version}")
    print(f"Model Type: {env.metadata.get('champion')}")
    print(f"Calibration Method: {env.metadata.get('calibrator')}")
    print(f"Decision Threshold: {env.operational_threshold}")
    print(f"Risk Policy Boundaries: {env.risk_levels}")
    print(f"Feature Contract: {env.feature_schema}")
    print("Explanation Method: Captum Integrated Gradients (Pre-Calibration Logit)")
    print("Intended Use: Real-time pre-transaction fraud scoring.")
    print("Known Limitations: Not a completely independent holdout prediction (Phase 3 evaluation leakage). Synthetic PaySim artifacts excluded.")

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host IP to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    log_level: str = typer.Option("info", help="Log level")
):
    """Start the FastAPI Production Inference Service."""
    import uvicorn
    logger.info(f"Starting API on {host}:{port} with {workers} workers")
    uvicorn.run(
        "fraudshield.api.app:create_app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level.lower(),
        factory=True
    )

if __name__ == "__main__":
    app()
