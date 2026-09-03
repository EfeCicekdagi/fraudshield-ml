import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from fraudshield.schemas.config import LightGBMConfig, DataConfig, BaselineConfig
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.models.lightgbm_model import get_lightgbm_pipeline, calculate_scale_pos_weight
from fraudshield.models.tuning import tune_lightgbm
from fraudshield.models.evaluation import evaluate_predictions
from fraudshield.models.thresholds import analyze_thresholds
from fraudshield.models.persistence import create_run_directory, save_pipeline, save_json
from fraudshield.models.reporting import (
    plot_pr_curve, plot_roc_curve, plot_threshold_metrics, plot_threshold_cost
)

logger = logging.getLogger(__name__)

def generate_lightgbm_markdown_report(run_dir, report_info, all_val_results, report_path, baseline_path=None):
    """Generates the Markdown report for LightGBM tuning and evaluation."""
    lines = [
        "# LightGBM Training and Tuning Report",
        f"**Run ID:** {report_info.get('run_id', 'Unknown')}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary of Best Model",
        f"- **Model Variant:** {report_info['model_name']}",
        f"- **Feature Set:** {report_info['feature_set']}",
        f"- **Selected Threshold:** {report_info['threshold']:.4f} ({report_info['threshold_reason']})",
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
        "## Feature Ablation Analysis",
        "Core ve Engineered feature setleri arasındaki farklar, simülasyona özgü (bakiye hatası vb.) değişkenlerin modele katkısını gösterir.",
        "Eğer Engineered özellikleri çıkarıldığında PR-AUC'de ciddi bir düşüş yaşanıyorsa, model büyük oranda bu simülasyon hatalarını ezberliyor olabilir.",
        "Gerçek hayata genelleme yaparken Core modelin daha tutarlı çalışması beklenebilir.",
        "",
        "## Optuna Tuning Information",
        "Optuna tuning aşamasında, veri seti kısıtlı bir örneklem üzerinden değerlendirilerek hiperparametre araması yapılmıştır.",
        "Test seti bu aşamada kesinlikle yüklenmemiş, yalnızca train ve validation verileri kullanılmıştır.",
        ""
    ])
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_lightgbm_pipeline(config_path: str):
    logger.info(f"Starting LightGBM Pipeline using config {config_path}")
    start_time = datetime.now()
    
    # 1. Load Configs
    config = LightGBMConfig.load_from_yaml(config_path)
    data_config = DataConfig.load_from_yaml(config.data_config_path)
    
    # 2. Load Data (ONLY train and val, NEVER test)
    train_path = os.path.join(data_config.processed_data_path, "train.parquet")
    val_path = os.path.join(data_config.processed_data_path, "val.parquet")
    
    logger.info(f"Loading train ({train_path}) and validation ({val_path}) sets...")
    logger.info("CRITICAL: Test set will NOT be loaded during this phase.")
    df_outer_train = pd.read_parquet(train_path)
    df_outer_val = pd.read_parquet(val_path)
    
    y_outer_train = df_outer_train[data_config.target_column].values
    y_outer_val = df_outer_val[data_config.target_column].values
    
    logger.info(f"Data Sizes - Outer Train: {len(df_outer_train)}, Outer Validation: {len(df_outer_val)}")
    
    # Calculate scale_pos_weight on outer train
    scale_pos_weight = calculate_scale_pos_weight(y_outer_train)
    logger.info(f"Calculated scale_pos_weight from outer train: {scale_pos_weight:.4f}")
    
    # Inner Temporal Split for Tuning
    # Outer train'i kronolojik olarak inner_train (%80) ve inner_val (%20) olarak bölüyoruz.
    inner_train_ratio = 1.0 - config.inner_validation_ratio
    
    logger.info("Performing inner temporal split on outer train for Optuna tuning...")
    # using temporal split function from data.splitting (which takes train_size and val_size and splits)
    # Since we are splitting outer_train into two, we pass train_size=inner_train_ratio, val_size=inner_validation_ratio
    df_inner_train, df_inner_val, _ = temporal_train_val_test_split(
        df_outer_train, 
        temporal_column=data_config.temporal_column,
        train_size=inner_train_ratio,
        val_size=config.inner_validation_ratio
    )
    
    # Subsample for tuning to avoid extremely long times
    logger.info(f"Subsampling for tuning (max rows = {config.tuning_max_rows})")
    total_inner_rows = len(df_inner_train) + len(df_inner_val)
    if total_inner_rows > config.tuning_max_rows:
        sample_frac = config.tuning_max_rows / total_inner_rows
        # Stratified sampling to maintain fraud ratio
        df_inner_train = df_inner_train.groupby(data_config.target_column, group_keys=False).sample(
            frac=sample_frac, random_state=config.random_seed
        )
        df_inner_val = df_inner_val.groupby(data_config.target_column, group_keys=False).sample(
            frac=sample_frac, random_state=config.random_seed
        )

    y_inner_train = df_inner_train[data_config.target_column].values
    y_inner_val = df_inner_val[data_config.target_column].values
    
    logger.info(f"Tuning Data Sizes - Inner Train: {len(df_inner_train)}, Inner Validation: {len(df_inner_val)}")
    
    run_dir = create_run_directory(config.model_artifact_dir)
    logger.info(f"Created run directory: {run_dir}")
    
    variants = [
        {"is_weighted": False, "name": "unweighted"},
        {"is_weighted": True, "name": "weighted"}
    ]
    
    all_val_results = []
    best_overall_pr_auc = -1
    best_overall_model_info = None
    
    for fset_name, feature_list in config.feature_sets.items():
        for variant in variants:
            model_name = f"LightGBM_{variant['name']}"
            logger.info(f"=== Starting {model_name} with {fset_name} features ===")
            
            # Optuna tuning
            study = tune_lightgbm(
                df_inner_train=df_inner_train,
                df_inner_val=df_inner_val,
                y_inner_train=y_inner_train,
                y_inner_val=y_inner_val,
                feature_list=feature_list,
                param_distributions=config.param_distributions,
                n_trials=config.n_trials,
                early_stopping_rounds=config.early_stopping_rounds,
                is_weighted=variant['is_weighted'],
                scale_pos_weight=scale_pos_weight,
                random_state=config.random_seed
            )
            
            # Save Optuna history
            study_df = study.trials_dataframe()
            study_csv_path = os.path.join(run_dir, f"optuna_trials_{model_name}_{fset_name}.csv")
            study_df.to_csv(study_csv_path, index=False)
            
            best_params = study.best_params
            logger.info(f"[{model_name} | {fset_name}] Best Tuning Params: {best_params}")
            
            # Retrain on full outer train
            logger.info(f"[{model_name} | {fset_name}] Retraining on full outer train with best params...")
            cat_features = [f for f in feature_list if df_outer_train[f].dtype.name in ['category', 'object']]
            num_features = [f for f in feature_list if f not in cat_features]
            
            pipeline = get_lightgbm_pipeline(
                numeric_features=num_features,
                categorical_features=cat_features,
                params=best_params,
                is_weighted=variant['is_weighted'],
                scale_pos_weight=scale_pos_weight,
                random_state=config.random_seed
            )
            
            start_train = datetime.now()
            pipeline.fit(df_outer_train[feature_list], y_outer_train)
            train_time = (datetime.now() - start_train).total_seconds()
            logger.info(f"[{model_name} | {fset_name}] Training time: {train_time:.2f} seconds")
            
            # Evaluate on outer validation
            y_val_prob = pipeline.predict_proba(df_outer_val[feature_list])[:, 1]
            y_val_pred_default = (y_val_prob >= 0.5).astype(int)
            val_metrics = evaluate_predictions(y_outer_val, y_val_pred_default, y_val_prob)
            
            logger.info(f"[{model_name} | {fset_name}] Outer Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
            
            thresh_results = analyze_thresholds(
                y_outer_val, y_val_prob,
                config.false_positive_cost, config.false_negative_cost,
                config.min_recall_targets
            )
            
            result_info = {
                'model_name': model_name,
                'feature_set': fset_name,
                'pipeline': pipeline,
                'best_params': best_params,
                'metrics': val_metrics,
                'thresholds': thresh_results['thresholds'],
                'curves': thresh_results['curves'],
                'feature_list': feature_list,
                'train_time_seconds': train_time
            }
            
            all_val_results.append(result_info)
            
            if val_metrics['pr_auc'] > best_overall_pr_auc:
                best_overall_pr_auc = val_metrics['pr_auc']
                best_overall_model_info = result_info

    # Save artifacts for the best model
    logger.info(f"Selected best overall model: {best_overall_model_info['model_name']} with {best_overall_model_info['feature_set']} features.")
    
    # Save Pipeline and JSONs
    save_pipeline(best_overall_model_info['pipeline'], run_dir, "model.joblib")
    save_json(best_overall_model_info['best_params'], run_dir, "best_params.json")
    
    # Save config, ensure Path objects are serialized
    config_dict = config.model_dump(mode='json') if hasattr(config, 'model_dump') else config.dict()
    # If mode='json' is not supported in this pydantic version, fallback to a custom encoder or yaml
    with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, default=str)
        
    save_json(best_overall_model_info['thresholds'], run_dir, "threshold.json")
    save_json(best_overall_model_info['feature_list'], run_dir, "feature_schema.json")
    
    val_metrics_list = [{'model': r['model_name'], 'feature_set': r['feature_set'], 'metrics': r['metrics']} for r in all_val_results]
    save_json({'val_results': val_metrics_list}, run_dir, "validation_metrics.json")
    
    metadata = {
        'run_id': os.path.basename(run_dir),
        'start_time': start_time.isoformat(),
        'end_time': datetime.now().isoformat(),
        'selected_model': best_overall_model_info['model_name'],
        'selected_feature_set': best_overall_model_info['feature_set']
    }
    save_json(metadata, run_dir, "run_metadata.json")
    
    # Figures
    figures_dir = os.path.join(run_dir, "figures", "lightgbm")
    os.makedirs(figures_dir, exist_ok=True)
    
    curves = best_overall_model_info['curves']
    plot_pr_curve(curves['recalls'], curves['precisions'], best_overall_model_info['metrics']['pr_auc'], os.path.join(figures_dir, "pr_curve.png"))
    plot_threshold_metrics(curves['thresholds'], curves['precisions'], curves['recalls'], curves['f1_scores'], os.path.join(figures_dir, "threshold_metrics.png"))
    
    # Rapor
    report_path = os.path.join(config.report_dir, "lightgbm_results.md")
    os.makedirs(config.report_dir, exist_ok=True)
    
    report_info = {
        'run_id': os.path.basename(run_dir),
        'model_name': best_overall_model_info['model_name'],
        'feature_set': best_overall_model_info['feature_set'],
        'threshold': best_overall_model_info['thresholds']['max_f1'],
        'threshold_reason': "Max F1 on Outer Validation"
    }
    generate_lightgbm_markdown_report(run_dir, report_info, all_val_results, report_path)
    
    # Additional metric export to reports/metrics
    metrics_report_dir = os.path.join(config.report_dir, "metrics")
    os.makedirs(metrics_report_dir, exist_ok=True)
    save_json({'val_results': val_metrics_list}, metrics_report_dir, "lightgbm_validation_metrics.json")
    
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"LightGBM Pipeline completed in {total_time:.2f} seconds. Results saved to {run_dir}")
