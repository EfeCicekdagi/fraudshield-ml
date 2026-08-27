import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from fraudshield.schemas.config import BaselineConfig, DataConfig
from fraudshield.models.training import (
    get_preprocessor, get_dummy_pipeline, 
    get_sgd_unweighted_pipeline, get_sgd_weighted_pipeline
)
from fraudshield.models.evaluation import evaluate_predictions
from fraudshield.models.thresholds import analyze_thresholds
from fraudshield.models.persistence import create_run_directory, save_pipeline, save_json
from fraudshield.models.reporting import (
    plot_pr_curve, plot_roc_curve, plot_threshold_metrics, plot_threshold_cost,
    plot_confusion_matrix, generate_markdown_report
)

logger = logging.getLogger(__name__)

def run_baseline_pipeline(config_path: str):
    logger.info(f"Starting Baseline Pipeline using config {config_path}")
    start_time = datetime.now()
    
    # 1. Load Config
    config = BaselineConfig.load_from_yaml(config_path)
    data_config = DataConfig.load_from_yaml(config.data_config_path)
    
    # 2 & 3 & 4. Load Data (assuming split-data has already been run)
    train_path = os.path.join(data_config.processed_data_path, "train.parquet")
    val_path = os.path.join(data_config.processed_data_path, "val.parquet")
    test_path = os.path.join(data_config.processed_data_path, "test.parquet")
    
    logger.info("Loading train, validation, and test sets...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    y_train = df_train[config.target_column].values
    y_val = df_val[config.target_column].values
    y_test = df_test[config.target_column].values
    
    rule_val = df_val['isFlaggedFraud'].values
    rule_test = df_test['isFlaggedFraud'].values
    
    # Define models
    model_definitions = {
        'dummy': get_dummy_pipeline,
        'sgd_unweighted': get_sgd_unweighted_pipeline,
        'sgd_weighted': get_sgd_weighted_pipeline
    }
    
    all_val_results = []
    best_model_info = None
    best_pr_auc = -1
    
    run_dir = create_run_directory(config.model_artifact_dir)
    logger.info(f"Created run directory: {run_dir}")
    
    # 5, 6, 7. Preprocess, Train, and Evaluate on Validation
    for fset_name, feature_list in config.feature_sets.items():
        # Identify numeric vs categorical
        cat_features = [f for f in feature_list if df_train[f].dtype.name in ['category', 'object']]
        num_features = [f for f in feature_list if f not in cat_features]
        
        preprocessor = get_preprocessor(num_features, cat_features)
        
        for model_name, model_fn in model_definitions.items():
            logger.info(f"Training {model_name} with {fset_name} features...")
            pipeline = model_fn(preprocessor)
            
            # Fit only on train
            pipeline.fit(df_train[feature_list], y_train)
            
            # Predict on val
            y_val_prob = pipeline.predict_proba(df_val[feature_list])[:, 1]
            
            # Default threshold 0.5 metrics for comparison table
            y_val_pred_default = (y_val_prob >= 0.5).astype(int)
            val_metrics = evaluate_predictions(y_val, y_val_pred_default, y_val_prob)
            
            logger.info(f"[{model_name} | {fset_name}] PR-AUC: {val_metrics['pr_auc']:.4f}")
            
            # 8. Threshold Analysis
            thresh_results = analyze_thresholds(
                y_val, y_val_prob, 
                config.false_positive_cost, config.false_negative_cost,
                config.min_recall_targets
            )
            
            all_val_results.append({
                'model_name': model_name,
                'feature_set': fset_name,
                'pipeline': pipeline,
                'metrics': val_metrics,
                'thresholds': thresh_results['thresholds'],
                'curves': thresh_results['curves'],
                'feature_list': feature_list,
                'val_prob': y_val_prob
            })
            
            # 9. Model Selection
            if val_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['pr_auc']
                best_model_info = all_val_results[-1]

    # Select threshold (using max_f1 as primary, or min_cost based on business logic)
    selected_threshold = best_model_info['thresholds']['max_f1']
    threshold_reason = "Max F1 Score on Validation Set"
    
    logger.info(f"Selected best model: {best_model_info['model_name']} with {best_model_info['feature_set']} features.")
    
    # 10. Test Set Evaluation (Once)
    best_pipeline = best_model_info['pipeline']
    feature_list = best_model_info['feature_list']
    
    logger.info("Evaluating selected model on test set...")
    y_test_prob = best_pipeline.predict_proba(df_test[feature_list])[:, 1]
    y_test_pred = (y_test_prob >= selected_threshold).astype(int)
    
    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)
    
    # Hybrid System Comparison
    hybrid_pred = np.logical_or(y_test_pred, rule_test).astype(int)
    hybrid_metrics = evaluate_predictions(y_test, hybrid_pred, y_test_prob) # prob doesn't matter for hard metrics
    
    rule_metrics = evaluate_predictions(y_test, rule_test, np.zeros_like(rule_test))
    
    hybrid_results = {
        'Rule Only (isFlaggedFraud)': rule_metrics,
        'ML Baseline Only': test_metrics,
        'Hybrid (Rule OR ML)': hybrid_metrics
    }
    
    # 11 & 12. Save Artifacts and Generate Report
    save_pipeline(best_pipeline, run_dir)
    save_json({'val_results': [
        {'model': r['model_name'], 'feature_set': r['feature_set'], 'metrics': r['metrics']}
        for r in all_val_results
    ]}, run_dir, "all_val_metrics.json")
    save_json(test_metrics, run_dir, "test_metrics.json")
    save_json(best_model_info['thresholds'], run_dir, "thresholds.json")
    save_json(feature_list, run_dir, "feature_schema.json")
    
    metadata = {
        'run_id': os.path.basename(run_dir),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'selected_model': best_model_info['model_name'],
        'selected_feature_set': best_model_info['feature_set'],
        'selected_threshold': selected_threshold
    }
    save_json(metadata, run_dir, "run_metadata.json")
    
    # Generate Plots
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    curves = best_model_info['curves']
    plot_pr_curve(curves['recalls'], curves['precisions'], best_model_info['metrics']['pr_auc'], os.path.join(figures_dir, "pr_curve.png"))
    plot_roc_curve(curves['fprs'], curves['tprs'], best_model_info['metrics']['roc_auc'], os.path.join(figures_dir, "roc_curve.png"))
    plot_threshold_metrics(curves['thresholds'], curves['precisions'], curves['recalls'], curves['f1_scores'], os.path.join(figures_dir, "threshold_metrics.png"))
    plot_threshold_cost(curves['thresholds'], curves['costs'], os.path.join(figures_dir, "threshold_costs.png"))
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(figures_dir, "confusion_matrix.png"))
    
    report_path = os.path.join(config.report_dir, "baseline_results.md")
    os.makedirs(config.report_dir, exist_ok=True)
    
    report_info = {
        'model_name': best_model_info['model_name'],
        'feature_set': best_model_info['feature_set'],
        'threshold': selected_threshold,
        'threshold_reason': threshold_reason
    }
    generate_markdown_report(run_dir, report_info, all_val_results, test_metrics, hybrid_results, report_path)
    
    logger.info(f"Pipeline completed in {datetime.now() - start_time}. Results saved to {run_dir} and {report_path}")
