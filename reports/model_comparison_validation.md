# Model Comparison (Validation Set)

Bu doküman, projede değerlendirilen temel modellerin yalnızca **validation (doğrulama)** seti üzerindeki performanslarını karşılaştırmaktadır. Test setine herhangi bir tahmin işlemi uygulanmamıştır.

## Validation Results

Aşağıdaki tablo her algoritma için elde edilen en iyi model (hyperparameter tuning ve ablation testleri sonucu) metriklerini içermektedir:

| Model | Feature Set | Loss/Objective | PR-AUC | ROC-AUC | F1-Score | Precision | Recall |
|-------|-------------|----------------|--------|---------|----------|-----------|--------|
| **SGD Baseline** | Engineered | Unweighted | 0.7482 | 0.9957 | 0.6024 | 0.9648 | 0.4379 |
| **MLP** | Engineered | Weighted BCE | 0.8146 | 0.9987 | 0.8298 | 0.9604 | 0.7305 |
| **LightGBM** | Engineered | Weighted | 1.0000 | 1.0000 | 0.9658 | 0.9338 | 1.0000 |

## Final Model Seçimi

LightGBM, validation seti üzerinde kusursuz bir performans (PR-AUC: 1.0000, Recall: 1.0000) sergilemiştir. MLP ise PR-AUC'yi 0.8146'ya çıkararak tatminkâr bir sonuç üretse de ağaç tabanlı LightGBM'in performansına ulaşamamıştır.

Bu bağlamda final değerlendirmesi ve production senaryoları için **LightGBM (Engineered + Weighted)** modeli en güçlü aday olarak öne çıkmaktadır.

> [!WARNING]
> **Genelleme Riski:** Engineered bakiye özelliklerinin (ör. `error_balance_orig`) PaySim veri setine ve simülasyon mantığına aşırı uyum (overfit) sağladığı açıkça görülmektedir. LightGBM'in kusursuz metriklere ulaşması, modelin fraud tespitinden ziyade veri setindeki deterministik bir hatayı ezberlediğine işaret edebilir. Gerçek hayat finansal verisinde bu özellikler dikkatli bir şekilde test edilmelidir.
