# Temporal Data Split Summary

Bu rapor, verinin leakage (sızıntı) engellenerek zamansal (temporal) olarak nasıl bölündüğünü göstermektedir.

## Train Seti
- **Satır Sayısı:** 4463587
- **Zaman Aralığı (Step):** 1 - 323
- **Fraud İşlem Sayısı:** 3643
- **Fraud Oranı:** %0.082

## Validation Seti
- **Satır Sayısı:** 980416
- **Zaman Aralığı (Step):** 324 - 378
- **Fraud İşlem Sayısı:** 564
- **Fraud Oranı:** %0.058

## Test Seti
- **Satır Sayısı:** 918617
- **Zaman Aralığı (Step):** 379 - 743
- **Fraud İşlem Sayısı:** 4006
- **Fraud Oranı:** %0.436

**Uyarı:** Validasyon ve Test setlerine kesinlikle oversampling (SMOTE vb.) veya undersampling uygulanmamalıdır. Sadece Train seti üzerinde bu işlemler yapılmalıdır.
