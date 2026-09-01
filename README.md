# FraudShield ML

FraudShield ML, finansal işlemler için dolandırıcılık olasılığı ve risk skoru üreten uçtan uca bir makine öğrenmesi projesidir. İlk veri kaynağı olarak **PaySim** kullanılacaktır.

## Proje Hedefi

- Finansal işlemleri `normal` veya `fraud` olarak sınıflandırmak
- Dengesiz sınıf dağılımını doğru yöntemlerle ele almak
- Her işlem için açıklanabilir bir risk skoru üretmek
- Model başarısını yalnızca accuracy ile değil; PR-AUC, recall, precision ve F1 ile değerlendirmek
- İlerleyen aşamalarda modeli API ve dashboard üzerinden kullanılabilir hâle getirmek

## Neden Temporal Split Kullanıyoruz?
Finansal işlemler zamana bağlıdır. Gelecekteki bir işlemdeki kalıpların, geçmiş işlemleri eğitirken sızdırılması (Data Leakage) modelin gerçek dünya performansını yanıltıcı derecede yüksek gösterebilir. Bu yüzden veriyi `step` (saat) sütununa göre kronolojik olarak bölüyoruz.

## Proje Mimarisi (v0.1.0)
Proje, araştırma notebook'larından çıkarak modüler ve kurulabilir bir Python paketi (`fraudshield`) haline getirilmiştir. İş mantığı, veri yükleme, özellik çıkarma ve modelleme işlemleri tamamen modüller içerisindedir. Notebook'lar yalnızca EDA (Keşifçi Veri Analizi) ve raporlama amaçlıdır, iş mantığı içermezler. Detaylar için [docs/architecture.md](docs/architecture.md) dosyasına bakabilirsiniz.

## Kurulum

Projeyi kurulabilir bir paket olarak kullanıyoruz.

```bash
# Sanal ortam oluştur
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# For LightGBM support (Phase 4):
pip install -e ".[boosting]"

# For Deep Learning / PyTorch support (Phase 5):
pip install -e ".[deep-learning]"

# For API and Dashboard (Phase 8 & 9)
pip install -e ".[api,dashboard]"
```

## Veri Seti

PaySim veri seti büyük olduğu için GitHub reposuna yüklenmez. CSV dosyasını yerel olarak aşağıdaki konuma yerleştirin:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```
`data/raw/` içeriği `.gitignore` tarafından korunmaktadır. Büyük çıktı dosyaları veya ara işlenmiş dosyalar (Parquet) Git'e eklenmez.

## CLI Kullanımı (Komut Satırı Arayüzü)

İşlemleri terminal üzerinden tetikleyebilirsiniz. (Konfigürasyonlar `configs/data.yaml` içinden okunur.)

```bash
# Veriyi doğrula
fraudshield validate-data --config configs/data.yaml

# Feature'ları (özellikleri) oluştur
fraudshield build-features --config configs/data.yaml

# Veriyi zaman serisine göre Train/Val/Test olarak böl
fraudshield split-data --config configs/data.yaml

# Inference API Sunucusunu Başlat
fraudshield serve --host 127.0.0.1 --port 8000
# Alternatif: uvicorn fraudshield.api.app:create_app --factory
```

## Testlerin Çalıştırılması

Tüm testleri (Birim ve Entegrasyon) çalıştırmak için `pytest` kullanabilirsiniz:

```bash
pytest
```

## Planlanan Sonraki Aşamalar

1. ~~Veri doğrulama ve keşifçi veri analizi~~ (Tamamlandı)
2. ~~Feature engineering ve Temporal Split~~ (Tamamlandı)
3. ~~Baseline model karşılaştırmaları~~ (Tamamlandı)
4. ~~LightGBM optimizasyonları~~ (Tamamlandı)
5. ~~Deep Learning (PyTorch MLP) entegrasyonu~~ (Tamamlandı)
6. ~~Threshold ve Calibration ayarlamaları~~ (Tamamlandı)
7. ~~Version-Independent Inference Bundle~~ (Tamamlandı)
8. ~~Production-Oriented FastAPI Inference Service~~ (Tamamlandı)
9. Streamlit dashboard (Planlanıyor)

## Çalıştırma (Inference API ve Dashboard)

Bu proje, production-oriented inference API ve analist operasyonları için bir Streamlit Dashboard sunmaktadır. Dashboard tamamen API-driven olup, hiçbir model objesini belleğe doğrudan yüklemez.

> **Uyarı:** `orig_account_type` ve `dest_account_type` gibi alanlar upstream (güvenilir) sistemler tarafından doğrulanmalı veya üretilmelidir. Halka açık (untrusted) istemcilerden doğrudan gelen bu verilere güvenilmemelidir. 

**API ve Dashboard'u Başlatma (İki ayrı terminalde çalıştırın):**

Terminal 1 (Backend API):
```powershell
$env:FRAUDSHIELD_API_KEY="my-secret-key"
fraudshield serve
```

Terminal 2 (Streamlit Dashboard):
```powershell
$env:FRAUDSHIELD_API_KEY="my-secret-key"
fraudshield dashboard --api-url http://127.0.0.1:8000/api/v1
```

**Örnek (PowerShell):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/predict" `
  -Method Post `
  -Headers @{"X-API-Key"="your-secret-key"} `
  -Body '{"step":1,"type":"PAYMENT","amount":100,"oldbalanceOrg":1000,"orig_account_type":"C","dest_account_type":"M"}' `
  -ContentType "application/json"
```
