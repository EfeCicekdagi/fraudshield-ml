import json
import torch
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

from fraudshield.logging_config import setup_logger
from fraudshield.schemas.config import MLPConfig, DataConfig
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.features.preprocessing import get_preprocessor
from fraudshield.models.mlp import FraudMLP
from fraudshield.models.losses import FocalLoss, calculate_pos_weight
from fraudshield.models.torch_dataset import create_dataloader
from fraudshield.models.torch_training import train_mlp, validate_mlp_epoch
from fraudshield.models.persistence import create_run_directory, save_json
from fraudshield.models.evaluation import evaluate_predictions
from fraudshield.models.thresholds import analyze_thresholds
from fraudshield.models.reporting import (
    generate_figures, 
    plot_training_history
)

logger = setup_logger(__name__)

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def _build_criterion(config_dict: Dict[str, Any], y_train: torch.Tensor) -> nn.Module:
    if config_dict['loss_type'] == 'focal_loss':
        return FocalLoss(alpha=config_dict['focal_loss_alpha'], gamma=config_dict['focal_loss_gamma'])
    else:
        # Default to weighted BCE
        pos_weight = calculate_pos_weight(y_train)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def tune_mlp(
    df_inner_train: pd.DataFrame,
    df_inner_val: pd.DataFrame,
    y_inner_train: pd.Series,
    y_inner_val: pd.Series,
    feature_list: list[str],
    base_config: MLPConfig,
    device: torch.device
) -> optuna.Study:
    """Tunes MLP hyperparameters using Optuna on the inner split."""
    
    logger.info(f"Starting Optuna tuning with {base_config.tuning_trials} trials...")
    
    cat_features = [f for f in feature_list if df_inner_train[f].dtype.name in ['category', 'object']]
    num_features = [f for f in feature_list if f not in cat_features]
    
    preprocessor = get_preprocessor(num_features, cat_features)
    X_train_processed = preprocessor.fit_transform(df_inner_train[feature_list])
    X_val_processed = preprocessor.transform(df_inner_val[feature_list])
    
    y_train_np = y_inner_train.values
    y_val_np = y_inner_val.values

    def objective(trial):
        # Sample hyperparameters
        config_dict = base_config.model_dump()
        
        # We override some keys from base config using trial suggestions
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dims = []
        for i in range(n_layers):
            dims.append(trial.suggest_categorical(f'dim_l{i}', [32, 64, 128, 256]))
        config_dict['hidden_dimensions'] = dims
        
        config_dict['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        config_dict['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config_dict['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        config_dict['loss_type'] = trial.suggest_categorical('loss_type', ['weighted_bce', 'focal_loss'])
        config_dict['activation'] = trial.suggest_categorical('activation', ['relu', 'gelu'])
        
        # Create DataLoaders
        train_loader = create_dataloader(X_train_processed, y_train_np, batch_size=config_dict['batch_size'], shuffle=True, num_workers=config_dict['num_workers'])
        val_loader = create_dataloader(X_val_processed, y_val_np, batch_size=config_dict['batch_size'], shuffle=False, num_workers=config_dict['num_workers'])
        
        input_dim = X_train_processed.shape[1]
        model = FraudMLP(
            input_dim=input_dim,
            hidden_dimensions=config_dict['hidden_dimensions'],
            dropout=config_dict['dropout'],
            activation=config_dict['activation'],
            batch_norm=config_dict['batch_norm']
        ).to(device)
        
        criterion = _build_criterion(config_dict, torch.tensor(y_train_np)).to(device)
        
        _, history = train_mlp(model, train_loader, val_loader, criterion, config_dict, device)
        
        best_val_pr_auc = max(history['val_pr_auc'])
        return best_val_pr_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=base_config.random_seed))
    study.optimize(objective, n_trials=base_config.tuning_trials)
    
    logger.info(f"Tuning finished. Best PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study

def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_mlp_pipeline(config_path: str):
    base_config = MLPConfig.load_from_yaml(config_path)
    data_config = DataConfig.load_from_yaml(base_config.data_config_path)
    
    set_seeds(base_config.random_seed)
    device = get_device(base_config.device)
    logger.info(f"Using device: {device} (PyTorch {torch.__version__})")
    
    # Ensure reproducibility note
    if device.type == 'cuda':
        logger.info("Note: Complete determinism on CUDA may not be guaranteed due to internal PyTorch operations.")
        
    logger.info("Loading training data...")
    train_path = Path(data_config.processed_data_path) / "train.parquet"
    val_path = Path(data_config.processed_data_path) / "val.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_outer_val = pd.read_parquet(val_path)
    
    # Inner temporal split for tuning
    logger.info(f"Creating inner temporal split (Validation Ratio: {base_config.inner_validation_ratio})...")
    
    df_inner_train, df_inner_val, _ = temporal_train_val_test_split(
        df_train,
        temporal_column=data_config.temporal_column,
        train_size=1.0 - base_config.inner_validation_ratio,
        val_size=base_config.inner_validation_ratio
    )
    
    if len(df_inner_train) > base_config.tuning_max_rows:
        logger.info(f"Subsampling inner_train for tuning (max rows: {base_config.tuning_max_rows})")
        # Stratified sampling based on target
        pos_df = df_inner_train[df_inner_train[data_config.target_column] == 1]
        neg_df = df_inner_train[df_inner_train[data_config.target_column] == 0]
        
        pos_ratio = len(pos_df) / len(df_inner_train)
        pos_samples = int(base_config.tuning_max_rows * pos_ratio)
        neg_samples = base_config.tuning_max_rows - pos_samples
        
        pos_sampled = pos_df.sample(n=pos_samples, random_state=base_config.random_seed)
        neg_sampled = neg_df.sample(n=neg_samples, random_state=base_config.random_seed)
        df_inner_train_tuned = pd.concat([pos_sampled, neg_sampled]).sort_values(by=data_config.temporal_column)
    else:
        df_inner_train_tuned = df_inner_train
        
    y_inner_train_tuned = df_inner_train_tuned[data_config.target_column]
    y_inner_val = df_inner_val[data_config.target_column]
    
    all_val_results = []
    best_overall_pr_auc = -1
    best_overall_model_info = None
    
    for fset_name, feature_list in base_config.feature_sets.items():
        logger.info(f"=== Starting MLP Tuning for {fset_name} features ===")
        
        study = tune_mlp(
            df_inner_train=df_inner_train_tuned,
            df_inner_val=df_inner_val,
            y_inner_train=y_inner_train_tuned,
            y_inner_val=y_inner_val,
            feature_list=feature_list,
            base_config=base_config,
            device=device
        )
        
        # Construct final config for this feature set
        final_config_dict = base_config.model_dump()
        for k, v in study.best_params.items():
            if k.startswith('dim_l'):
                continue
            final_config_dict[k] = v
        
        # reconstruct hidden_dimensions from best params
        n_layers = study.best_params.get('n_layers', len(base_config.hidden_dimensions))
        dims = [study.best_params.get(f'dim_l{i}', 64) for i in range(n_layers)]
        final_config_dict['hidden_dimensions'] = dims
        
        logger.info(f"=== Training Final Model on Full Outer Train ({fset_name}) ===")
        cat_features = [f for f in feature_list if df_train[f].dtype.name in ['category', 'object']]
        num_features = [f for f in feature_list if f not in cat_features]
        
        preprocessor = get_preprocessor(num_features, cat_features)
        X_outer_train = preprocessor.fit_transform(df_train[feature_list])
        y_outer_train = df_train[data_config.target_column].values
        
        X_outer_val = preprocessor.transform(df_outer_val[feature_list])
        y_outer_val = df_outer_val[data_config.target_column].values
        
        train_loader = create_dataloader(X_outer_train, y_outer_train, batch_size=final_config_dict['batch_size'], shuffle=True, num_workers=final_config_dict['num_workers'])
        val_loader = create_dataloader(X_outer_val, y_outer_val, batch_size=final_config_dict['batch_size'], shuffle=False, num_workers=final_config_dict['num_workers'])
        
        input_dim = X_outer_train.shape[1]
        model = FraudMLP(
            input_dim=input_dim,
            hidden_dimensions=final_config_dict['hidden_dimensions'],
            dropout=final_config_dict['dropout'],
            activation=final_config_dict['activation'],
            batch_norm=final_config_dict['batch_norm']
        ).to(device)
        
        criterion = _build_criterion(final_config_dict, torch.tensor(y_outer_train)).to(device)
        
        best_model, history = train_mlp(model, train_loader, val_loader, criterion, final_config_dict, device)
        
        logger.info("Evaluating on Outer Validation set...")
        best_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = best_model(torch.tensor(X_outer_val, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            
        thresh_results = analyze_thresholds(
            y_outer_val, probs,
            fp_cost=base_config.false_positive_cost,
            fn_cost=base_config.false_negative_cost,
            min_recall_targets=base_config.min_recall_targets
        )
        thresholds_info = thresh_results['thresholds']
        curves = thresh_results['curves']
        
        best_threshold = thresholds_info['max_f1']
        y_pred = (probs >= best_threshold).astype(int)
        
        val_metrics = evaluate_predictions(y_outer_val, y_pred, probs)
        
        model_name = f"MLP_{final_config_dict['loss_type']}"
        val_results = {
            "model_name": model_name,
            "feature_set": fset_name,
            "metrics": val_metrics,
            "thresholds": thresholds_info,
            "curves": curves,
            "history": history,
            "best_params": final_config_dict,
            "y_true": y_outer_val,
            "y_prob": probs,
            "model_state_dict": best_model.state_dict(),
            "preprocessor": preprocessor,
            "architecture": {
                "input_dim": input_dim,
                "hidden_dimensions": final_config_dict['hidden_dimensions'],
                "dropout": final_config_dict['dropout'],
                "activation": final_config_dict['activation'],
                "batch_norm": final_config_dict['batch_norm']
            }
        }
        all_val_results.append(val_results)
        
        if val_metrics['pr_auc'] > best_overall_pr_auc:
            best_overall_pr_auc = val_metrics['pr_auc']
            best_overall_model_info = val_results
            
    # Save artifacts for the best overall model
    run_dir = create_run_directory(str(base_config.model_artifact_dir))
    run_dir_path = Path(run_dir)
    
    logger.info(f"Selected best overall model: {best_overall_model_info['model_name']} with {best_overall_model_info['feature_set']} features.")
    
    # Save PyTorch model state dict
    torch.save(best_overall_model_info['model_state_dict'], run_dir_path / "model_state_dict.pt")
    
    # Save preprocessor
    import joblib
    joblib.dump(best_overall_model_info['preprocessor'], run_dir_path / "preprocessor.joblib")
    
    save_json(best_overall_model_info['best_params'], run_dir, "best_params.json")
    save_json(best_overall_model_info['architecture'], run_dir, "architecture.json")
    config_dict = base_config.model_dump(mode='json') if hasattr(base_config, 'model_dump') else base_config.dict()
    with open(run_dir_path / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, default=str)
    save_json(best_overall_model_info['thresholds'], run_dir, "threshold.json")
    
    feature_schema = {
        "features": base_config.feature_sets[best_overall_model_info['feature_set']]
    }
    save_json(feature_schema, run_dir, "feature_schema.json")
    save_json(best_overall_model_info['history'], run_dir, "training_history.json")
    
    # Reports
    report_dir_path = Path(base_config.report_dir)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir_path / "figures" / "mlp"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_training_history(best_overall_model_info['history'], str(figures_dir))
    
    generate_figures(
        best_overall_model_info['y_true'], 
        best_overall_model_info['y_prob'],
        best_overall_model_info,
        str(figures_dir),
        prefix="mlp"
    )
    
    metrics_report_dir = report_dir_path / "metrics"
    metrics_report_dir.mkdir(parents=True, exist_ok=True)
    val_metrics_list = [{'model': r['model_name'], 'feature_set': r['feature_set'], 'metrics': r['metrics']} for r in all_val_results]
    save_json({'val_results': val_metrics_list}, str(metrics_report_dir), "mlp_validation_metrics.json")
    
    # Generate Markdown Report
    lines = [
        "# PyTorch MLP Training and Tuning Report",
        f"**Run ID:** {run_dir_path.name}",
        "",
        "## Summary of Best Model",
        f"- **Model Variant:** {best_overall_model_info['model_name']}",
        f"- **Feature Set:** {best_overall_model_info['feature_set']}",
        f"- **Selected Threshold:** {best_overall_model_info['thresholds']['max_f1']:.4f} (Max F1 on Outer Validation)",
        "",
        "## Outer Validation Results (All Variants)",
        "| Model Variant | Feature Set | PR-AUC | ROC-AUC | F1 | Precision | Recall |",
        "|---------------|-------------|--------|---------|----|-----------|--------|"
    ]
    
    for res in all_val_results:
        m = res['metrics']
        lines.append(
            f"| {res['model_name']} | {res['feature_set']} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['f1_score']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |"
        )
        
    lines.extend([
        "",
        "## Core vs Engineered Analysis",
        "Mühendislik özellikleri (Engineered features) ağ tabanlı veya gradient bazlı modellerde kritik olduğu kadar MLP'de de fark yaratabilir.",
        "Ancak over-fitting riskini gözlemlemek için Core sonuçlarıyla karşılaştırma yapılmıştır.",
        "",
        "## BCE vs Focal Loss Analysis",
        "Aşırı dengesiz verilerde Focal Loss zorlu fraud örneklerine odaklanmayı sağlar. Metriklerdeki PR-AUC değeri Focal Loss'un ağırlıklı BCE'ye kıyasla etkisini göstermektedir."
    ])
    
    with open(report_dir_path / "mlp_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
