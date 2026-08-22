# FraudShield ML

FraudShield ML, finansal işlemler için dolandırıcılık olasılığı ve risk skoru üreten uçtan uca bir makine öğrenmesi projesidir. İlk veri kaynağı olarak **PaySim** kullanılacaktır.

## Proje hedefi

- Finansal işlemleri `normal` veya `fraud` olarak sınıflandırmak
- Dengesiz sınıf dağılımını doğru yöntemlerle ele almak
- Her işlem için açıklanabilir bir risk skoru üretmek
- Model başarısını yalnızca accuracy ile değil; PR-AUC, recall, precision ve F1 ile değerlendirmek
- İlerleyen aşamalarda modeli API ve dashboard üzerinden kullanılabilir hâle getirmek

## Planlanan aşamalar

1. Veri doğrulama ve keşifçi veri analizi
2. Feature engineering
3. Baseline model karşılaştırmaları
4. Dengesiz veri ve threshold optimizasyonu
5. Deep learning modeli
6. Açıklanabilir risk skoru
7. FastAPI servis katmanı
8. Streamlit dashboard

## Veri seti

PaySim veri seti büyük olduğu için GitHub reposuna yüklenmez. CSV dosyasını yerel olarak aşağıdaki konuma yerleştirin:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

`data/raw/` içeriği `.gitignore` tarafından korunmaktadır. Veri dosyası bilgisayarda kalır ve commit edilmez.

## Kurulum

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## İlk kontrol

Veri dosyasını `data/raw/` içine koyduktan sonra:

```bash
python -m fraudshield.data.validate
```

## Proje yapısı

```text
fraudshield-ml/
├── config/              # Proje ve model ayarları
├── data/                # Yerel veri alanları; veri dosyaları Git'e girmez
├── notebooks/           # EDA ve deney notebook'ları
├── reports/             # Grafikler ve analiz sonuçları
├── src/fraudshield/     # Uygulama kaynak kodları
├── tests/               # Otomatik testler
├── .env.example         # Ortam değişkeni örneği
├── .gitignore
└── requirements.txt
```

## Durum

Proje temiz bir repo yapısıyla başlatıldı. İlk geliştirme adımı PaySim veri doğrulaması ve EDA çalışmasıdır.

