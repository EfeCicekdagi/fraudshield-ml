# Baseline Modeling and Evaluation Results

## Model Özeti
- **Kullanılan Feature Listesi:** ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'hour_of_day', 'day', 'week', 'is_risky_type', 'amount_log1p', 'amount_is_zero', 'orig_balance_delta', 'error_balance_orig', 'abs_error_balance_orig', 'orig_oldbalance_is_zero', 'orig_newbalance_is_zero', 'amount_to_oldbalance_orig_ratio', 'dest_balance_delta', 'error_balance_dest', 'abs_error_balance_dest', 'dest_oldbalance_is_zero', 'dest_newbalance_is_zero', 'amount_to_oldbalance_dest_ratio', 'type', 'orig_account_type', 'dest_account_type']
- **Veri Boyutları:** Train (4463587), Validation (980416), Test (918617)

## Validation Metrikleri (Threshold: 0.5)
|                     |      PR-AUC |   ROC-AUC |   Precision |   Recall |   F1-score |   Alerts per 1000 |
|:--------------------|------------:|----------:|------------:|---------:|-----------:|------------------:|
| Dummy               | 0.000575266 |  0.5      |   0         | 0        |  0         |          0        |
| Unweighted Logistic | 0.744243    |  0.995415 |   0.964     | 0.427305 |  0.592138  |          0.254994 |
| Weighted Logistic   | 0.618302    |  0.998249 |   0.0254763 | 0.991135 |  0.0496756 |         22.3803   |

## Seçim Gerekçesi
- **Seçilen Model:** Unweighted Logistic (En yüksek PR-AUC'ye sahip olması sebebiyle seçilmiştir).
- **Seçilen Operasyon Threshold'u:** 0.0682 (Validation seti üzerinde F1 skorunu maksimize eden noktadır).

## Test Seti Karşılaştırması
|                            |   Precision |     Recall |   F1-score |   True Positive |   False Positive |   False Negative |   Alerts per 1000 |
|:---------------------------|------------:|-----------:|-----------:|----------------:|-----------------:|-----------------:|------------------:|
| Rule Only (isFlaggedFraud) |    1        | 0.00324513 | 0.00646927 |              13 |                0 |             3993 |         0.0141517 |
| ML Only (Baseline)         |    0.976403 | 0.681727   | 0.802881   |            2731 |               66 |             1275 |         3.04479   |
| Hybrid (Rule OR ML)        |    0.976512 | 0.684973   | 0.805164   |            2744 |               66 |             1262 |         3.05895   |

**Sonuç:** ML modeli (özellikle Hybrid sistem), mevcut sadece-kural tabanlı yaklaşıma göre Recall (Duyarlılık) oranını binlerce kat artırmış, makul bir Precision ile Fraud vakalarının büyük kısmını yakalamıştır.
