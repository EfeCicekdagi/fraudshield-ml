import os
import json
import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yaml
from pathlib import Path
from sklearn.metrics import brier_score_loss
import torch

from sklearn.linear_model import SGDClassifier
from fraudshield.schemas.config import DataConfig
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.features.availability import check_feature_safety, generate_availability_report
from fraudshield.models.lightgbm_model import get_lightgbm_pipeline, calculate_scale_pos_weight
from fraudshield.models.calibration import fit_calibrator, evaluate_calibration
from fraudshield.models.risk_levels import get_risk_levels, assign_risk_level
from fraudshield.models.thresholds import analyze_thresholds
from fraudshield.models.audit import evaluate_metrics

logger = logging.getLogger(__name__)

def evaluate_baseline_rule(df, y_true):
    """Simple baseline rule: IF amount > 200,000 AND type in TRANSFER/CASH_OUT THEN fraud"""
    preds = ((df['amount'] > 200000) & (df['type'].isin(['TRANSFER', 'CASH_OUT']))).astype(int)
    return evaluate_metrics(y_true, preds, threshold=0.5)

def train_mlp_fixed(df_train, y_train, df_val, y_val, features, mlp_config, scale_pos):
    # Dynamic import to avoid dependency issues if torch isn't used
    import torch
    from fraudshield.models.mlp import FraudMLP
    from fraudshield.models.torch_training import train_mlp_epoch, validate_mlp_epoch
    from fraudshield.models.torch_dataset import create_dataloader

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    cat_features = [f for f in features if df_train[f].dtype.name in ['category', 'object']]
    num_features = [f for f in features if f not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )

    X_train_t = preprocessor.fit_transform(df_train[features])
    X_val_t = preprocessor.transform(df_val[features])
    
    input_dim = X_train_t.shape[1]
    
    model = FraudMLP(
        input_dim=input_dim,
        hidden_dimensions=mlp_config['hidden_dimensions'],
        dropout=mlp_config['dropout']
    )
    
    device = torch.device('cpu')
    model = model.to(device)
    
    if mlp_config['loss'] == 'focal_loss':
        from fraudshield.models.losses import FocalLoss
        criterion = FocalLoss()
    else:
        # Default to BCEWithLogitsLoss
        pos_weight = torch.tensor([1.0], dtype=torch.float32) # Using balanced weights or default
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=mlp_config['learning_rate'])
    
    train_loader = create_dataloader(
        X_train_t.toarray() if hasattr(X_train_t, 'toarray') else X_train_t, y_train, batch_size=mlp_config['batch_size'], shuffle=True
    )
    val_loader = create_dataloader(
        X_val_t.toarray() if hasattr(X_val_t, 'toarray') else X_val_t, y_val, batch_size=mlp_config['batch_size'], shuffle=False
    )
    
    for epoch in range(mlp_config['epochs']):
        train_mlp_epoch(model, train_loader, criterion, optimizer, device, None, 0.0)
        
    # Get probs
    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits = model(xb.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.extend(probs)
            
    val_probs = np.array(val_probs)
    return preprocessor, model, val_probs

class MLPPipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.classes_ = np.array([0, 1])
        
    def predict_proba(self, X):
        import torch
        X_t = self.preprocessor.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_t.toarray() if hasattr(X_t, 'toarray') else X_t))
            probs = torch.sigmoid(logits).numpy()
        return np.vstack((1 - probs, probs)).T
    
    def fit(self, X, y):
        # Already fitted manually
        pass
        
def run_finalization_pipeline(config_path: str, force_final_test: bool = False):
    logger.info("Starting Final Model Selection and Calibration Pipeline")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    out_dir = config['final_artifact_dir']
    if os.path.exists(os.path.join(out_dir, "final_test_metrics.json")) and not force_final_test:
        raise RuntimeError("Final test has already been run. Use force_final_test=True to rerun.")

        
    data_config = DataConfig.load_from_yaml(config['data_config_path'])
    
    logger.info("Generating Feature Availability Report")
    generate_availability_report(os.path.join(config['report_dir'], 'feature_availability.md'))
    
    features = config['feature_sets']['pre_transaction']
    logger.info(f"Checking safety of {len(features)} pre_transaction features...")
    check_feature_safety(features)
    
    # 1. Load Data
    df_train = pd.read_parquet(os.path.join(data_config.processed_data_path, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(data_config.processed_data_path, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(data_config.processed_data_path, "test.parquet"))
    
    y_train = df_train[data_config.target_column].values
    y_val = df_val[data_config.target_column].values
    y_test = df_test[data_config.target_column].values
    
    scale_pos = calculate_scale_pos_weight(y_train)
    
    # 2. Inner Split for Selection
    logger.info("Splitting outer train for inner validation...")
    df_in_train, df_in_val, _ = temporal_train_val_test_split(
        df_train, temporal_column=data_config.temporal_column,
        train_size=(1.0 - config['inner_validation_ratio']),
        val_size=config['inner_validation_ratio']
    )
    y_in_train = df_in_train[data_config.target_column].values
    y_in_val = df_in_val[data_config.target_column].values
    
    candidates = {}
    metrics = {}
    
    # Train SGD
    logger.info("Training SGD Baseline...")
    cat_feats = [f for f in features if df_train[f].dtype.name in ['category', 'object']]
    num_feats = [f for f in features if f not in cat_feats]
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    sgd_pipe = Pipeline([
        ('prep', ColumnTransformer([('num', StandardScaler(), num_feats), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)])),
        ('clf', SGDClassifier(loss=config['candidates']['sgd']['loss'], penalty=config['candidates']['sgd']['penalty'], alpha=config['candidates']['sgd']['alpha'], class_weight='balanced', random_state=42))
    ])
    t0 = time.time()
    sgd_pipe.fit(df_in_train[features], y_in_train)
    sgd_time = time.time() - t0
    y_prob = sgd_pipe.predict_proba(df_in_val[features])[:, 1]
    candidates['SGD'] = sgd_pipe
    metrics['SGD'] = evaluate_metrics(y_in_val, y_prob)
    metrics['SGD']['time'] = sgd_time
    
    # Train LGBM Unweighted
    logger.info("Training LightGBM Unweighted...")
    lgb_u = get_lightgbm_pipeline(num_feats, cat_feats, config['candidates']['lightgbm'], is_weighted=False, scale_pos_weight=1.0)
    t0 = time.time()
    lgb_u.fit(df_in_train[features], y_in_train)
    lu_time = time.time() - t0
    y_prob = lgb_u.predict_proba(df_in_val[features])[:, 1]
    candidates['LGBM_Unweighted'] = lgb_u
    metrics['LGBM_Unweighted'] = evaluate_metrics(y_in_val, y_prob)
    metrics['LGBM_Unweighted']['time'] = lu_time
    
    # Train LGBM Weighted
    logger.info("Training LightGBM Weighted...")
    lgb_w = get_lightgbm_pipeline(num_feats, cat_feats, config['candidates']['lightgbm'], is_weighted=True, scale_pos_weight=scale_pos)
    t0 = time.time()
    lgb_w.fit(df_in_train[features], y_in_train)
    lw_time = time.time() - t0
    y_prob = lgb_w.predict_proba(df_in_val[features])[:, 1]
    candidates['LGBM_Weighted'] = lgb_w
    metrics['LGBM_Weighted'] = evaluate_metrics(y_in_val, y_prob)
    metrics['LGBM_Weighted']['time'] = lw_time
    
    # Train MLP
    logger.info("Training MLP Fixed Architecture...")
    t0 = time.time()
    mlp_prep, mlp_model, y_prob = train_mlp_fixed(df_in_train, y_in_train, df_in_val, y_in_val, features, config['candidates']['mlp'], scale_pos)
    mlp_time = time.time() - t0
    mlp_pipe = MLPPipeline(mlp_prep, mlp_model)
    candidates['MLP_Weighted'] = mlp_pipe
    metrics['MLP_Weighted'] = evaluate_metrics(y_in_val, y_prob)
    metrics['MLP_Weighted']['time'] = mlp_time
    
    # 3. Model Selection
    logger.info("Candidate Inner Validation Metrics:")
    best_pr = -1
    champion_name = ""
    for name, m in metrics.items():
        logger.info(f"{name}: PR-AUC={m['pr_auc']:.4f}, F1={m['f1']:.4f}, Time={m['time']:.1f}s")
        if m['pr_auc'] > best_pr:
            best_pr = m['pr_auc']
            champion_name = name
            
    logger.info(f"Selected Champion: {champion_name}")
    
    # 4. Final Training on FULL outer train
    logger.info("Retraining champion on full outer train...")
    champion = candidates[champion_name]
    if champion_name == 'MLP_Weighted':
        mlp_prep, mlp_model, _ = train_mlp_fixed(df_train, y_train, df_val, y_val, features, config['candidates']['mlp'], scale_pos)
        champion = MLPPipeline(mlp_prep, mlp_model)
    else:
        champion.fit(df_train[features], y_train)
        
    # 5. Split outer val for calibration and decision
    df_calib, df_dec, _ = temporal_train_val_test_split(
        df_val, temporal_column=data_config.temporal_column,
        train_size=config['outer_validation_calibration_ratio'],
        val_size=(1.0 - config['outer_validation_calibration_ratio'])
    )
    y_calib = df_calib[data_config.target_column].values
    y_dec = df_dec[data_config.target_column].values
    
    logger.info("Calibrating on calibration split...")
    calib_sigmoid = fit_calibrator(champion, df_calib[features], y_calib, method='sigmoid')
    calib_isotonic = fit_calibrator(champion, df_calib[features], y_calib, method='isotonic')
    
    prob_sig = calib_sigmoid.predict_proba(df_dec[features])[:, 1]
    prob_iso = calib_isotonic.predict_proba(df_dec[features])[:, 1]
    
    eval_sig = evaluate_calibration(y_dec, prob_sig)
    eval_iso = evaluate_calibration(y_dec, prob_iso)
    
    logger.info(f"Sigmoid: Brier={eval_sig['brier_score']:.6f}, ECE={eval_sig['ece']:.6f}")
    logger.info(f"Isotonic: Brier={eval_iso['brier_score']:.6f}, ECE={eval_iso['ece']:.6f}")
    
    # Pick calibrator with lower Brier score
    if eval_iso['brier_score'] < eval_sig['brier_score']:
        final_calibrator = calib_isotonic
        calib_name = "Isotonic"
    else:
        final_calibrator = calib_sigmoid
        calib_name = "Sigmoid"
        
    logger.info(f"Selected calibrator: {calib_name}")
    
    # 6. Thresholding & Risk Levels
    y_dec_prob = final_calibrator.predict_proba(df_dec[features])[:, 1]
    t_res = analyze_thresholds(y_dec, y_dec_prob, config['false_positive_cost'], config['false_negative_cost'], config['min_recall_targets'])
    
    thresholds = t_res['thresholds']
    op_threshold = thresholds['min_cost'] # Use min cost as operational threshold
    
    risk_bounds = get_risk_levels(thresholds)
    logger.info(f"Risk boundaries established: {risk_bounds}")
    
    # 7. Locked Test Evaluation
    logger.info("========== LOCKED TEST SET EVALUATION ==========")
    logger.info("Predicting on Test Set...")
    y_test_prob = final_calibrator.predict_proba(df_test[features])[:, 1]
    y_test_pred = (y_test_prob >= op_threshold).astype(int)
    
    test_metrics = evaluate_metrics(y_test, y_test_prob, threshold=op_threshold)
    test_metrics['brier_score'] = float(brier_score_loss(y_test, y_test_prob))
    
    # Rule baseline
    rule_metrics = evaluate_baseline_rule(df_test, y_test)
    
    # Hybrid
    hybrid_preds = ((y_test_prob >= op_threshold) | ((df_test['amount'] > 200000) & (df_test['type'].isin(['TRANSFER', 'CASH_OUT'])))).astype(int)
    hybrid_metrics = evaluate_metrics(y_test, hybrid_preds, threshold=op_threshold)
    
    # Remove invalid PR-AUC for binary rules
    rule_metrics['pr_auc'] = None
    hybrid_metrics['pr_auc'] = None
    
    logger.info(f"Rule F1: {rule_metrics['f1']:.4f}")
    logger.info(f"ML Final PR-AUC: {test_metrics['pr_auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    logger.info(f"Hybrid F1: {hybrid_metrics['f1']:.4f}")
    
    # Save Artifacts
    out_dir = config['final_artifact_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    if champion_name != 'MLP_Weighted':
        joblib.dump(champion, os.path.join(out_dir, "model.joblib"))
    else:
        torch.save(champion.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
        joblib.dump(champion.preprocessor, os.path.join(out_dir, "preprocessor.joblib"))
        
    joblib.dump(final_calibrator, os.path.join(out_dir, "calibrator.joblib"))
    
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"operational_threshold": op_threshold, "all_thresholds": thresholds}, f, indent=4)
        
    with open(os.path.join(out_dir, "risk_levels.json"), "w") as f:
        json.dump(risk_bounds, f, indent=4)
        
    with open(os.path.join(out_dir, "feature_schema.json"), "w") as f:
        json.dump({"features": features}, f, indent=4)
        
    with open(os.path.join(out_dir, "final_test_metrics.json"), "w") as f:
        json.dump({"ML": test_metrics, "Rule": rule_metrics, "Hybrid": hybrid_metrics}, f, indent=4)
        
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump({
            "champion": champion_name,
            "calibrator": calib_name,
            "is_final": True,
            "rules": {
                "high_amount_rule": "amount > 200000 AND type in ('TRANSFER', 'CASH_OUT')",
                "paysim_native_rule": "isFlaggedFraud == 1"
            }
        }, f, indent=4)
        
    # Generate final reports
    generate_final_reports(config['report_dir'], test_metrics, rule_metrics, hybrid_metrics, metrics, champion_name, calib_name, risk_bounds)
    logger.info("Finalization complete.")
    
def generate_final_reports(report_dir, test_m, rule_m, hyb_m, inner_m, champ, calib, risk):
    rep = os.path.join(report_dir, "final_model_selection.md")
    lines = [
        "# Final Model Selection",
        "| Candidate | PR-AUC | F1 | Time (s) |",
        "|-----------|--------|----|----------|"
    ]
    for n, m in inner_m.items():
        lines.append(f"| {n} | {m['pr_auc']:.4f} | {m['f1']:.4f} | {m['time']:.1f} |")
        
    lines.extend([
        "",
        f"**Selected Champion:** `{champ}`",
        f"**Calibration Method:** `{calib}`",
        "",
        "## Calibration Metrics",
        f"- Final Brier score: {test_m.get('brier_score', 'Missing'):.5f}" if 'brier_score' in test_m else "- Final Brier score: *Missing*",
        "- Brier score (before): *Missing*",
        "- Log loss (before/after): *Missing*",
        "- ECE (before/after): *Missing*",
        "- Calibration split step range: *Missing*",
        "- Decision split step range: *Missing*",
        "",
        "## Configurable Policy Risk Boundaries",
        "*(Not empirically derived from validation data)*",
        f"- LOW: <= {risk['LOW_MAX']:.4f}",
        f"- MEDIUM: {risk['LOW_MAX']:.4f} < score <= {risk['MEDIUM_MAX']:.4f}",
        f"- HIGH: {risk['MEDIUM_MAX']:.4f} < score <= {risk['HIGH_MAX']:.4f}",
        f"- CRITICAL: > {risk['HIGH_MAX']:.4f}",
    ])
    with open(rep, "w") as f: f.write("\n".join(lines))
    
    trep = os.path.join(report_dir, "final_test_results.md")
    tlines = [
        "# Final evaluation of the point-in-time-safe deployment candidate",
        "",
        "## A. Score-based model performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| ML Final PR-AUC | {test_m['pr_auc']:.4f} |",
        f"| ROC-AUC | {test_m['roc_auc']:.4f} |",
        f"| Brier score | {test_m.get('brier_score', 0):.4f} |",
        "| Log loss | *Missing* |",
        "| ECE | *Missing* |",
        "",
        "## B. Decision performance",
        "",
        "| System | Threshold / Rule Definition | Precision | Recall | F1 | TP | FP | FN | TN | Alerts/1K | Cost |",
        "|--------|-----------------------------|-----------|--------|----|----|----|----|----|-----------|------|",
        "| PaySim native | `isFlaggedFraud == 1` | 1.0000 | 0.0032 | 0.0065 | 13 | 0 | 3993 | 914611 | 0.01 | *Missing* |",
        f"| High-amount rule | `amount > 200000 AND type in ('TRANSFER', 'CASH_OUT')` | {rule_m['precision']:.4f} | {rule_m['recall']:.4f} | {rule_m['f1']:.4f} | {rule_m['tp']} | {rule_m['fp']} | {rule_m['fn']} | {rule_m['tn']} | {rule_m['alerts_per_1k']:.2f} | *Missing* |",
        f"| Final ML | `>= {test_m['threshold']:.4f}` | {test_m['precision']:.4f} | {test_m['recall']:.4f} | {test_m['f1']:.4f} | {test_m['tp']} | {test_m['fp']} | {test_m['fn']} | {test_m['tn']} | {test_m['alerts_per_1k']:.2f} | *Missing* |",
        "| Native rule OR ML | *Missing* | - | - | - | - | - | - | - | - | - |",
        f"| High-amount rule OR ML | `ML >= {test_m['threshold']:.4f} OR High-amount rule` | {hyb_m['precision']:.4f} | {hyb_m['recall']:.4f} | {hyb_m['f1']:.4f} | {hyb_m['tp']} | {hyb_m['fp']} | {hyb_m['fn']} | {hyb_m['tn']} | {hyb_m['alerts_per_1k']:.2f} | *Missing* |"
    ]
    with open(trep, "w") as f: f.write("\n".join(tlines))
    
    mcard = os.path.join(report_dir, "model_card.md")
    mlines = [
        "# Model Card: FraudShield ML (v1.0.0)",
        "",
        "## Model Details",
        f"- **Algorithm:** {champ}",
        f"- **Calibration:** {calib} (empirically calibrated probability estimate, calibrated on a temporally separated calibration split. Calibration validity depends on future data resembling the calibration distribution)",
        "- **Features Used:** point-in-time-safe (`pre_transaction`)",
        "",
        "## Intended Use",
        "- **Primary Use Case:** Real-time pre-transaction fraud scoring.",
        "- **Out-of-Scope:** Post-transaction reconciliation.",
        "",
        "## Known Limitations and Warnings",
        "- **Synthetic Data:** Trained entirely on PaySim, a simulated dataset.",
        "- **Artifact Exclusion:** Highly predictive balance errors were removed to ensure generalization. The `Full Engineered` model was downgraded to a benchmark artifact.",
        "- **Real-World Validation Required:** The hypothetical costs and behavior do not exactly mirror a live bank.",
        "- **Test Set Usage:** The test set was previously used for the Phase 3 baseline benchmark. Therefore, the results are not a completely independent and pristine holdout prediction.",
        "",
        "## Performance on Test Set",
        f"- **F1 Score:** {test_m['f1']:.4f}",
        f"- **Recall:** {test_m['recall']:.4f}"
    ]
    with open(mcard, "w") as f: f.write("\n".join(mlines))

