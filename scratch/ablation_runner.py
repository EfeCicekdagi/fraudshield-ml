import json
import time
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath('src'))

from fraudshield.schemas.config import MLPConfig, DataConfig
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.features.preprocessing import get_preprocessor
from fraudshield.models.mlp import FraudMLP
from fraudshield.models.losses import FocalLoss, calculate_pos_weight
from fraudshield.models.torch_dataset import create_dataloader
from fraudshield.models.torch_training import train_mlp
from fraudshield.models.evaluation import evaluate_predictions
from fraudshield.models.thresholds import analyze_thresholds

def run_ablation():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    data_config = DataConfig.load_from_yaml("configs/data.yaml")
    base_config = MLPConfig.load_from_yaml("configs/mlp.yaml")
    
    torch.manual_seed(base_config.random_seed)
    
    train_path = Path(data_config.processed_data_path) / "train.parquet"
    val_path = Path(data_config.processed_data_path) / "val.parquet"
    
    df_train = pd.read_parquet(train_path)
    df_outer_val = pd.read_parquet(val_path)
    y_outer_train = df_train[data_config.target_column].values
    y_outer_val = df_outer_val[data_config.target_column].values
    
    results = []

    # HELPER to get metrics
    def evaluate_model(model, val_loader, X_val, y_val, start_time, end_time, history, config_dict, fset_name, name):
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        thresh_results = analyze_thresholds(y_val, probs, fp_cost=1.0, fn_cost=10.0, min_recall_targets=[0.8, 0.9])
        best_threshold = thresh_results['thresholds']['max_f1']
        y_pred = (probs >= best_threshold).astype(int)
        val_metrics = evaluate_predictions(y_val, y_pred, probs)
        
        # calculate tp, fp, fn, tn and alerts per 1k
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        alert_rate = (fp + tp) / len(y_val)
        alerts_per_1k = alert_rate * 1000

        res = {
            "model": name,
            "feature_set": fset_name,
            "metrics": val_metrics,
            "threshold": best_threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "alerts_per_1k": alerts_per_1k,
            "training_time_seconds": end_time - start_time,
            "best_epoch": history.get('best_epoch', -1),
            "architecture": config_dict
        }
        results.append(res)
        print(f"Finished {name}: F1={val_metrics.get('f1_score')}, PR-AUC={val_metrics.get('pr_auc')}")

    def prepare_data(fset_name):
        feature_list = base_config.feature_sets[fset_name]
        cat_features = [f for f in feature_list if df_train[f].dtype.name in ['category', 'object']]
        num_features = [f for f in feature_list if f not in cat_features]
        preprocessor = get_preprocessor(num_features, cat_features)
        X_outer_train = preprocessor.fit_transform(df_train[feature_list])
        X_outer_val = preprocessor.transform(df_outer_val[feature_list])
        train_loader = create_dataloader(X_outer_train, y_outer_train, batch_size=4096, shuffle=True, num_workers=0)
        val_loader = create_dataloader(X_outer_val, y_outer_val, batch_size=4096, shuffle=False, num_workers=0)
        return X_outer_train, X_outer_val, train_loader, val_loader

    # 1. Evaluate Existing Engineered + Focal Loss
    print("=== Evaluating Existing Engineered + Focal Loss ===")
    run_dir = "artifacts/runs/20260829_175510_b7fa37"
    X_train_eng, X_val_eng, train_loader_eng, val_loader_eng = prepare_data('engineered')
    
    with open(os.path.join(run_dir, "best_params.json")) as f:
        config_dict_eng_fl = json.load(f)
    
    input_dim = X_train_eng.shape[1]
    model_eng_fl = FraudMLP(input_dim=input_dim, hidden_dimensions=config_dict_eng_fl['hidden_dimensions'],
                     dropout=config_dict_eng_fl['dropout'], activation=config_dict_eng_fl['activation'],
                     batch_norm=True).to(device)
    model_eng_fl.load_state_dict(torch.load(os.path.join(run_dir, "model_state_dict.pt"), map_location=device))
    
    with open(os.path.join(run_dir, "training_history.json")) as f:
        history_eng_fl = json.load(f)
        
    training_time_eng_fl = sum(history_eng_fl.get('epoch_times', []))
    best_epoch_eng_fl = history_eng_fl.get('val_loss', []).index(min(history_eng_fl.get('val_loss', [0]))) if history_eng_fl.get('val_loss') else -1
    history_eng_fl['best_epoch'] = best_epoch_eng_fl + 1
    
    evaluate_model(model_eng_fl, val_loader_eng, X_val_eng, y_outer_val, 0, training_time_eng_fl, history_eng_fl, config_dict_eng_fl, 'engineered', 'Engineered + Focal Loss')

    # 2. Train Engineered + Weighted BCE
    print("=== Training Engineered + Weighted BCE ===")
    config_dict_eng_wbce = base_config.model_dump()
    config_dict_eng_wbce.update({
        'hidden_dimensions': [128, 64, 32], 'dropout': 0.21355389431312816, 'activation': 'relu',
        'learning_rate': 0.00432543242796456, 'weight_decay': 0.027728241828010616, 'loss_type': 'weighted_bce'
    })
    model_eng_wbce = FraudMLP(input_dim=input_dim, hidden_dimensions=config_dict_eng_wbce['hidden_dimensions'],
                     dropout=config_dict_eng_wbce['dropout'], activation=config_dict_eng_wbce['activation'], batch_norm=True).to(device)
    pos_weight = calculate_pos_weight(torch.tensor(y_outer_train))
    criterion_eng_wbce = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    start_time = time.time()
    best_model_eng_wbce, history_eng_wbce = train_mlp(model_eng_wbce, train_loader_eng, val_loader_eng, criterion_eng_wbce, config_dict_eng_wbce, device)
    end_time = time.time()
    evaluate_model(best_model_eng_wbce, val_loader_eng, X_val_eng, y_outer_val, start_time, end_time, history_eng_wbce, config_dict_eng_wbce, 'engineered', 'Engineered + Weighted BCE')

    # 3. Train Core + Focal Loss
    print("=== Training Core + Focal Loss ===")
    X_train_core, X_val_core, train_loader_core, val_loader_core = prepare_data('core')
    config_dict_core_fl = base_config.model_dump()
    config_dict_core_fl.update({
        'hidden_dimensions': [32, 256, 256], 'dropout': 0.2806385987847481, 'activation': 'gelu',
        'learning_rate': 0.003482846706526885, 'weight_decay': 0.0009444574254983562, 'loss_type': 'focal_loss'
    })
    input_dim_core = X_train_core.shape[1]
    model_core_fl = FraudMLP(input_dim=input_dim_core, hidden_dimensions=config_dict_core_fl['hidden_dimensions'],
                     dropout=config_dict_core_fl['dropout'], activation=config_dict_core_fl['activation'], batch_norm=True).to(device)
    criterion_core_fl = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    start_time = time.time()
    best_model_core_fl, history_core_fl = train_mlp(model_core_fl, train_loader_core, val_loader_core, criterion_core_fl, config_dict_core_fl, device)
    end_time = time.time()
    evaluate_model(best_model_core_fl, val_loader_core, X_val_core, y_outer_val, start_time, end_time, history_core_fl, config_dict_core_fl, 'core', 'Core + Focal Loss')

    # 4. Train Core + Weighted BCE
    print("=== Training Core + Weighted BCE ===")
    config_dict_core_wbce = base_config.model_dump()
    config_dict_core_wbce.update({
        'hidden_dimensions': [32, 256, 256], 'dropout': 0.2806385987847481, 'activation': 'gelu',
        'learning_rate': 0.003482846706526885, 'weight_decay': 0.0009444574254983562, 'loss_type': 'weighted_bce'
    })
    model_core_wbce = FraudMLP(input_dim=input_dim_core, hidden_dimensions=config_dict_core_wbce['hidden_dimensions'],
                     dropout=config_dict_core_wbce['dropout'], activation=config_dict_core_wbce['activation'], batch_norm=True).to(device)
    criterion_core_wbce = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    start_time = time.time()
    best_model_core_wbce, history_core_wbce = train_mlp(model_core_wbce, train_loader_core, val_loader_core, criterion_core_wbce, config_dict_core_wbce, device)
    end_time = time.time()
    evaluate_model(best_model_core_wbce, val_loader_core, X_val_core, y_outer_val, start_time, end_time, history_core_wbce, config_dict_core_wbce, 'core', 'Core + Weighted BCE')

    def default_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    os.makedirs("scratch", exist_ok=True)
    with open("scratch/ablation_results.json", "w") as f:
        json.dump(results, f, indent=4, default=default_serializer)
    print("All done!")

if __name__ == "__main__":
    run_ablation()
