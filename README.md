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
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pip install -e ".[boosting]"
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
4. Dengesiz veri ve threshold optimizasyonunun geliştirilmesi (Planlanıyor)
5. Deep learning / Gelişmiş Ağaç modelleri (XGBoost vs.)
6. Açıklanabilir risk skoru (SHAP vs.)
7. FastAPI servis katmanı
8. Streamlit dashboard
