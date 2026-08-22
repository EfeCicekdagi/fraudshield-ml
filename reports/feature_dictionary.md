# Feature Dictionary

Bu doküman, `fraudshield-ml` projesi kapsamında geliştirilen makine öğrenmesi modellerine girdi olarak sağlanacak özellikleri (features) açıklar. Olası sızıntı (leakage) durumları ve özelliklerin iş anlamları belirtilmiştir.

## Zaman Özellikleri (Temporal Features)
- **`hour_of_day`**: (Tipi: `int8`) `step` alanından türetilir. İşlemin günün hangi saatinde yapıldığını (0-23 arası) gösterir. Gece yapılan işlemlerdeki dolandırıcılık riskini yakalamak için üretilmiştir.
- **`day`**: (Tipi: `int16`) `step` alanından türetilir. Ayın hangi gününde işlem yapıldığını ifade eder.
- **`week`**: (Tipi: `int8`) `day` üzerinden hesaplanmıştır.

## İşlem Özellikleri (Transaction Features)
- **`type`**: (Tipi: `category`) İşlemin ham türü (CASH_IN, CASH_OUT, TRANSFER vb.). Target leakage riski taşımaz.
- **`is_risky_type`**: (Tipi: `int8`) İşlem türü `TRANSFER` veya `CASH_OUT` ise 1, diğer hallerde 0 değerini alır. Dolandırıcılık yalnızca bu türlerde görüldüğü için ağaç modellerine doğrudan sinyal verir.
- **`amount`**: (Tipi: `float32`) İşlemin orijinal tutarı.
- **`amount_log1p`**: (Tipi: `float32`) İşlem tutarının doğal logaritmasının (log(1+x)) alınmış hali. Uç değerlerin (outliers) etkisini baskılar.
- **`amount_is_zero`**: (Tipi: `int8`) İşlem tutarı sıfırsa 1, değilse 0'dır. Anormal sistem test işlemlerini tespit eder.

## Kaynak Hesap Özellikleri (Origin Balance Features)
- **`orig_balance_delta`**: (Tipi: `float32`) `oldbalanceOrg - newbalanceOrig`. İşlem sonrası hesabın net değişim miktarını gösterir.
- **`error_balance_orig`**: (Tipi: `float32`) `oldbalanceOrg - amount - newbalanceOrig`. **Önemli:** PaySim verisetinde dolandırıcılar hesapları boşaltırken sistem tutarsızlıkları (bakiye güncellenmemesi) yaratır. Sıfırdan farklı olan büyük hatalar fraud sinyali olabilir.
- **`abs_error_balance_orig`**: (Tipi: `float32`) Bakiye hata miktarının mutlak değeridir.
- **`orig_oldbalance_is_zero`**: (Tipi: `int8`) İşlem öncesi kaynak bakiye 0 ise 1'dir.
- **`orig_newbalance_is_zero`**: (Tipi: `int8`) İşlem sonrası bakiye sıfırlanmışsa 1'dir. Hesabın boşaltıldığını gösterir.
- **`amount_to_oldbalance_orig_ratio`**: (Tipi: `float32`) Transfer edilen tutarın, mevcut bakiyeye oranı. Tüm hesabı boşaltma eğilimi olan durumları (%100 veya 1.0) yakalar. (Sıfıra bölmede 0 döner).

## Hedef Hesap Özellikleri (Destination Balance Features)
- **`dest_balance_delta`**: (Tipi: `float32`) `newbalanceDest - oldbalanceDest`.
- **`error_balance_dest`**: (Tipi: `float32`) `oldbalanceDest + amount - newbalanceDest`. Beklenen hedef bakiye ile gerçekleşen hedef bakiye arasındaki farktır.
- **`abs_error_balance_dest`**: (Tipi: `float32`) Hedef hesap hata miktarının mutlak değeridir.
- **`dest_oldbalance_is_zero`**: (Tipi: `int8`) Hedef hesaba ilk defa mı para geliyor (0) kontrolüdür.
- **`dest_newbalance_is_zero`**: (Tipi: `int8`) İşlem sonrası hedef bakiye 0 ise 1'dir.
- **`amount_to_oldbalance_dest_ratio`**: (Tipi: `float32`) Tutarın hedef bakiyeye oranı.

## Hesap Tipi Özellikleri (Account Type Features)
- **`orig_account_type`**: (Tipi: `category`) `nameOrig` alanının ilk harfidir ('C' = Customer, 'M' = Merchant).
- **`dest_account_type`**: (Tipi: `category`) `nameDest` alanının ilk harfidir ('C' = Customer, 'M' = Merchant).

## Sızıntı Önlemleri ve Dışlanan Özellikler (Leakage & Dropped Features)
- `nameOrig` ve `nameDest` (Tam metin olarak): Modeli ezbere yöneltme ve kardinalitenin çok yüksek olması sebebiyle kaldırıldı.
- `isFraud`: Hedef değişkendir. Baseline listesinde yer almaz (eğitimde kullanılmaz).
- `isFlaggedFraud`: Sadece değerlendirme (evaluation) aşamasında kıyaslamak için tutulmuştur, modele (X) girdi olarak verilemez.
