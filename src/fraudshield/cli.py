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

if __name__ == "__main__":
    app()
