# Exploratory Data Analysis (EDA) Özeti

Bu rapor, fraudshield-ml projesinin ilk aşaması olan veri doğrulama ve EDA sürecinin bulgularını özetlemektedir. Veri seti üzerinde makine öğrenmesi modelleri geliştirilmeden önce verinin yapısı, kalitesi ve olası sorunlar incelenmiştir.

## 1. Veri Setinin Genel Boyutu ve Özellikleri
- **Toplam Kayıt (Satır) Sayısı:** 6362620
- **Toplam Değişken (Sütun) Sayısı:** 11
- **Eksik Veri (Missing Values):** Tüm değişkenlerde 0 eksik değer bulundu (veya: {'step': 0, 'type': 0, 'amount': 0, 'nameOrig': 0, 'oldbalanceOrg': 0, 'newbalanceOrig': 0, 'nameDest': 0, 'oldbalanceDest': 0, 'newbalanceDest': 0, 'isFraud': 0, 'isFlaggedFraud': 0}).
- **Tekrar Eden Kayıt (Duplicates):** 0 adet birebir aynı satır bulundu.

## 2. Fraud (Dolandırıcılık) Analizi
- **Toplam Fraud İşlem Sayısı:** 8213
- **Genel Fraud Oranı:** %0.129
- Sınıf dengesizliği oldukça yüksektir. Modeller eğitilirken SMOTE veya undersampling gibi yöntemler kullanılması zorunludur.

## 3. İşlem Türleri ve Fraud İlişkisi
- Veri setinde genel olarak şu işlem türleri mevcuttur: {'CASH_OUT': 2237500, 'PAYMENT': 2151495, 'CASH_IN': 1399284, 'TRANSFER': 532909, 'DEBIT': 41432}
- **Fraud Görülen İşlem Türleri:** {'CASH_OUT': 4116, 'TRANSFER': 4097, 'CASH_IN': 0, 'DEBIT': 0, 'PAYMENT': 0}
- Fraud işlemleri yalnızca belirli türlerde (genellikle TRANSFER ve CASH_OUT) gerçekleşmiştir.

## 4. isFlaggedFraud Mekanizmasının Durumu
- `isFlaggedFraud` kuralı tüm veri setinde incelendiğinde şu sonuçları vermiştir:
  - **True Negative (TN):** 6,354,407
  - **False Positive (FP):** 0
  - **False Negative (FN):** 8,197
  - **True Positive (TP):** 16
- **Precision (Kesinlik):** 1.0 (hiç yanlış alarm vermiyor)
- **Recall (Duyarlılık):** %0.19 (çok düşük)
- Mevcut `isFlaggedFraud` kuralı (200.000 üzeri yasadışı transfer girişimi gibi sabit bir kural), tespit ettiği vakalarda (16 adet) %100 doğru çalışsa da, toplam fraud vakalarının (8.213 adet) çok küçük bir kısmını yakalamaktadır. Precision'ı yüksek, ancak Recall çok düşük olduğu için tek başına model veya güçlü bir kural olarak kullanılması yetersizdir.

## 5. Veri Kalitesi Sorunları
- **Eksik Sütunlar:** []
- **Negatif Değerler:** Miktarlarda (0) veya bakiyelerde negatif değer var mı?
- Veri kalitesi genel olarak iyidir, ancak bakiye güncellemelerinde mantıksal tutarsızlıklar (eski bakiye - işlem tutarı != yeni bakiye) modelleme aşamasında özellik çıkarımı (feature engineering) için kullanılabilir.

## 6. Modelleme Aşaması İçin Öneriler ve Riskler
1. **Sınıf Dengesizliği:** Fraud sınıfı çok nadir görüldüğü için ağaç tabanlı algoritmalar (RandomForest, XGBoost) tercih edilebilir.
2. **Feature Engineering:**
   - step (saatlik zaman dilimi) değişkeni günün saatlerine veya haftanın günlerine dönüştürülebilir.
   - Origin ve Destination bakiyelerindeki matematiksel uyuşmazlıklar dolandırıcılığın çok güçlü bir göstergesi olabilir (alance_error_orig, alance_error_dest gibi yeni sütunlar üretilmeli).
3. **Modelleme Kapsamı:** Modelin sadece dolandırıcılık gözlemlenen işlem türleri (TRANSFER ve CASH_OUT) üzerine eğitilmesi düşünülebilir, diğer türler kurallı olarak (fraud = 0) filtrelenebilir.
