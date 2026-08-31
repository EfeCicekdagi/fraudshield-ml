# Final evaluation of the point-in-time-safe deployment candidate

## A. Score-based model performance

| Metric | Value |
|--------|-------|
| ML Final PR-AUC | 0.7442 |
| ROC-AUC | 0.9921 |
| Brier score | 0.0021 |
| Log loss | *Missing* |
| ECE | *Missing* |

## B. Decision performance

| System | Threshold / Rule Definition | Precision | Recall | F1 | TP | FP | FN | TN | Alerts/1K | Cost |
|--------|-----------------------------|-----------|--------|----|----|----|----|----|-----------|------|
| PaySim native | `isFlaggedFraud == 1` | 1.0000 | 0.0032 | 0.0065 | 13 | 0 | 3993 | 914611 | 0.01 | *Missing* |
| High-amount rule | `amount > 200000 AND type in ('TRANSFER', 'CASH_OUT')` | 0.0167 | 0.6792 | 0.0326 | 2721 | 160118 | 1285 | 754493 | 177.27 | *Missing* |
| Final ML | `>= 0.0210` | 0.4521 | 0.7574 | 0.5662 | 3034 | 3677 | 972 | 910934 | 7.31 | *Missing* |
| Native rule OR ML | *Missing* | - | - | - | - | - | - | - | - | - |
| High-amount rule OR ML | `ML >= 0.0210 OR High-amount rule` | 0.0194 | 0.7968 | 0.0378 | 3192 | 161729 | 814 | 752882 | 179.53 | *Missing* |