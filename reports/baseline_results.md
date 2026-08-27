# Baseline Model Evaluation Report

**Run ID:** 20260827_171226_3dd8cf
**Selected Model:** sgd_unweighted
**Selected Feature Set:** engineered
**Selected Threshold:** 0.0853 (Reason: Max F1 Score on Validation Set)

## Validation Experiments (All Models)

| Model | Feature Set | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | Alert Rate |
|---|---|---|---|---|---|---|---|
| dummy | core | 0.0006 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sgd_unweighted | core | 0.4930 | 0.9620 | 0.2038 | 0.1135 | 1.0000 | 0.0001 |
| sgd_weighted | core | 0.0611 | 0.9819 | 0.0114 | 0.9255 | 0.0058 | 0.0926 |
| dummy | engineered | 0.0006 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sgd_unweighted | engineered | 0.7482 | 0.9957 | 0.6024 | 0.4379 | 0.9648 | 0.0003 |
| sgd_weighted | engineered | 0.6174 | 0.9983 | 0.0531 | 0.9911 | 0.0273 | 0.0209 |

## Final Test Set Results

| Metric | Value |
|---|---|
| pr_auc | 0.8596 |
| roc_auc | 0.9926 |
| precision | 0.9781 |
| recall | 0.6787 |
| f1_score | 0.8014 |
| accuracy | 0.9985 |
| tn | 914550 |
| fp | 61 |
| fn | 1287 |
| tp | 2719 |
| alert_rate | 0.0030 |
| alerts_per_1k | 3.0263 |

## System Comparison (Rule vs ML vs Hybrid) on Test Set

| System | Precision | Recall | F1 | TP | FP | FN | TN | Alerts/1K |
|---|---|---|---|---|---|---|---|---|
| Rule Only (isFlaggedFraud) | 1.0000 | 0.0032 | 0.0065 | 13 | 0 | 3993 | 914611 | 0.01 |
| ML Baseline Only | 0.9781 | 0.6787 | 0.8014 | 2719 | 61 | 1287 | 914550 | 3.03 |
| Hybrid (Rule OR ML) | 0.9782 | 0.6820 | 0.8036 | 2732 | 61 | 1274 | 914550 | 3.04 |

## Warning regarding PaySim
> **Note:** PaySim is a synthetic dataset. Engineered features based on balance errors (e.g. `error_balance_orig`) often act as perfect indicators of fraud due to simulation artifacts. Ensure real-world data reflects these discrepancies before deploying models reliant on them.
