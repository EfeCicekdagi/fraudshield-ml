# LightGBM Training and Tuning Report
**Run ID:** 20260828_230111_725bd4
**Date:** 2026-08-28 23:03:41

## Summary of Best Model
- **Model Variant:** LightGBM_weighted
- **Feature Set:** engineered
- **Selected Threshold:** 0.9999 (Max F1 on Outer Validation)

## Outer Validation Results (All Variants)
| Model Variant | Feature Set | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---------------|-------------|--------|---------|----|-----------|--------|
| LightGBM_unweighted | core | 0.6597 | 0.9708 | 0.3364 | 0.2246 | 0.6702 |
| LightGBM_weighted | core | 0.4644 | 0.9991 | 0.0949 | 0.0498 | 0.9965 |
| LightGBM_unweighted | engineered | 0.2567 | 0.9303 | 0.2962 | 0.1764 | 0.9238 |
| LightGBM_weighted | engineered | 1.0000 | 1.0000 | 0.9658 | 0.9338 | 1.0000 |

## Feature Ablation Analysis
Core ve Engineered feature setleri arasındaki farklar, simülasyona özgü (bakiye hatası vb.) değişkenlerin modele katkısını gösterir.
Eğer Engineered özellikleri çıkarıldığında PR-AUC'de ciddi bir düşüş yaşanıyorsa, model büyük oranda bu simülasyon hatalarını ezberliyor olabilir.
Gerçek hayata genelleme yaparken Core modelin daha tutarlı çalışması beklenebilir.

## Optuna Tuning Information
Optuna tuning aşamasında, veri seti kısıtlı bir örneklem üzerinden değerlendirilerek hiperparametre araması yapılmıştır.
Test seti bu aşamada kesinlikle yüklenmemiş, yalnızca train ve validation verileri kullanılmıştır.
