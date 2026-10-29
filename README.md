# ERGİN TEKNİK ATÖLYE · Tigerk
## Saf Python ile Yapay Sinir Ağı

Tigerk, Ergin Teknik Atölye içindeki eğitim amaçlı sinir ağı deneyidir. İleri besleme, sigmoid aktivasyonu, geri yayılım ve ortalama kare hata (MSE) gibi temel parçaları harici makine öğrenmesi kütüphanesi olmadan görünür kılar.

## Gereksinim

Python 3.9+; harici paket gerekmez.

## Çalıştırma

Depo kök dizininde:

```bash
python main.py
python mantik.py
python denetci.py
python -m unittest test_tigerk -v
```

`main.py` AND kapısını eğitir, ardından mantığı ve denetimi basar.

## Mantık

Anlatım, o çalıştırmanın kendi sayılarıdır. Formül metne sonradan yapıştırılmaz; basılan delta, kodun o adımda ürettiği delta ile karşılaştırılır.

- **Geri yayılım neden kullanılır.** Hata çıkışta ölçülür. Sayısal türev her ağırlık için ayrı bir ileri geçiş ister. Geri yayılım aynı türevi zincir kuralıyla, çıkıştan girişe tek geçişte taşır.
- **Türev nasıl hesaplanır.** Eğitim kaybı `L = ½(y - hedef)²` olduğu için `∂L/∂y = y - hedef`. Sigmoid türevi çıktıdan yazılır: `y·(1-y)`. Delta bunların çarpımıdır. Ağırlık türevi `delta · x`, eşik türevi `delta`dır. Gizli katmanda delta, sonraki ağırlıklarla geri toplanır.
- **Öğrenme oranı sonucu nasıl değiştirir.** Güncelleme `w ← w - oran · ∂L/∂w` biçimindedir. Oran yön seçmez, adımın boyunu değiştirir. Küçük oran yavaş iner. Fazla büyük oran tek bir örneğin hatasını düşürebilir; dört örnek birlikte eğitilince aynı oran hatayı büyütüp doğru sınıf sayısını düşürebilir.

## Denetim

`denetci.py` çalışan geri yayılımı, bu `L` kaybının sayısal türeviyle karşılaştırır. Üç taslak bilerek bozuktur. Tanı taslağın adına bakmaz: gözlenen türevi bilinen hesaplarla örtüştürür, doğru türevi yerine koyar ve sayısal farkın düştüğünü gösterir.

## Modüller

- `ag.py`: ağ ve geri yayılım
- `mantik.py`: geri yayılım, türev ve öğrenme oranı
- `denetci.py`: sayısal türev, hatalı taslağı bulma ve düzeltme
- `katman.py`, `noron.py`: ağ yapısı
- `aktivasyon.py`, `kayip.py`: sigmoid ve MSE
- `egitim.py`: eğitim sarmalayıcısı
- `ve_kapisi.py`: AND kapısı eğitim verisi

## Sınırlar

Bu, küçük ve açıklanabilir bir eğitim uygulamasıdır; üretim makine öğrenmesi çerçevesi veya genel amaçlı dil modeli değildir. AND kapısına ait test, yalnızca bu küçük örneğin beklenen davranışını doğrular.

**Atölye:** Ergin Teknik Atölye
