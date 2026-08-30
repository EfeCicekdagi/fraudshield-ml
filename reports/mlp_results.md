# PyTorch MLP Training and Ablation Report

## Configuration and Hardware
- **PyTorch Version**: 2.12.0+cpu
- **Device**: CPU
- **Test Set**: Test setine tahmin yapılmamış olup, tüm değerlendirmeler validation seti üzerinde gerçekleştirilmiştir.

## Best Model Architecture & Hyperparameters (Engineered + Weighted BCE)
- **Architecture Variant**: MLP_weighted_bce (Engineered Features)
- **Best Hidden Dimensions**: `[128, 64, 32]`
- **Dropout Rate**: 0.2135
- **Learning Rate**: 0.0043
- **Batch Size**: 4096
- **Loss Function Parameters**: `Weighted BCE (fp_cost=1.0, fn_cost=10.0)`
- **Total Training Time**: 1104 seconds

## Outer Validation Results

| Model Variant | Feature Set | PR-AUC | ROC-AUC | F1 | Precision | Recall | TP | FP | FN | TN | Alerts / 1K | Training Time |
|---------------|-------------|--------|---------|----|-----------|--------|----|----|----|----|-------------|---------------|
| MLP_focal_loss | Core | 0.5952 | 0.9969 | 0.5762 | 0.7731 | 0.4592 | 259 | 76 | 305 | 979776 | 0.34 | 1547s |
| MLP_weighted_bce | Core | 0.6477 | 0.9980 | 0.6249 | 0.6980 | 0.5656 | 319 | 138 | 245 | 979714 | 0.47 | 1968s |
| MLP_focal_loss | Engineered | 0.8033 | 0.9985 | 0.8218 | 0.9803 | 0.7074 | 399 | 8 | 165 | 979844 | 0.42 | 2402s |
| MLP_weighted_bce | Engineered | 0.8146 | 0.9987 | 0.8298 | 0.9604 | 0.7305 | 412 | 17 | 152 | 979835 | 0.44 | 1104s |

> [!WARNING]
> **Genelleme Riski:** Engineered bakiye özelliklerinin PaySim simülasyonuna özgü olabileceği ve gerçek finansal veriye (production ortamına) genelleme riski taşıdığı dikkate alınmalıdır.

## Feature Ablation Analysis
Mühendislik özellikleri (Engineered features) ile temel özellikler (Core features) arasındaki fark, her iki kayıp fonksiyonunda da tutarlı bir şekilde gözlemlenmektedir:
- Core özelliklerden Engineered özelliklere geçiş, Focal Loss kullanıldığında PR-AUC'de yaklaşık %20 (0.595'ten 0.803'e), Weighted BCE kullanıldığında ise yaklaşık %17 (0.648'den 0.815'e) artış sağlamaktadır.
- Bu artış, bağımsız değişken olarak ele alındığında feature engineering adımlarının performansa olan etkisini desteklemektedir. Ancak yukarıda belirtilen genelleme riski göz önünde bulundurulmalıdır.

## Loss Ablation Analysis
Loss fonksiyonlarının (Focal Loss vs. Weighted BCE) etkileri izole edildiğinde, aşağıdaki bulgular elde edilmiştir:
- Core feature set üzerinde değerlendirildiğinde Weighted BCE (PR-AUC: 0.6477), Focal Loss'a (PR-AUC: 0.5952) kıyasla daha yüksek bir performans sergilemektedir.
- Benzer şekilde Engineered feature set üzerinde de Weighted BCE (PR-AUC: 0.8146), Focal Loss'tan (PR-AUC: 0.8033) daha yüksek bir PR-AUC değeri ile sonuçlanmaktadır.
- Sonuç olarak, iki değişken aynı anda değişirken performans artışının sadece loss fonksiyonuna bağlanamayacağı, Feature Engineering etkisinin ağırlıklı olduğu anlaşılmaktadır. Bu veri setinde, hiperparametre aramasından elde edilen `fp_cost=1.0, fn_cost=10.0` parametreli Weighted BCE, Focal Loss'tan genel olarak daha yüksek metrikler üretmiştir.