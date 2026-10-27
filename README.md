# ERGİN TEKNİK ATÖLYE · Tigerk
## Saf Python ile Yapay Sinir Ağı

Tigerk, Ergin Teknik Atölye içindeki eğitim amaçlı sinir ağı deneyidir. İleri besleme, sigmoid aktivasyonu, geri yayılım ve ortalama kare hata (MSE) gibi temel parçaları harici makine öğrenmesi kütüphanesi olmadan görünür kılar.

## Gereksinim

Python 3.9+; harici paket gerekmez.

## Çalıştırma

Depo kök dizininde:

```bash
python main.py
python -m unittest test_tigerk -v
```

## Modüller

- `ag.py`: ağ ve geri yayılım
- `katman.py`, `noron.py`: ağ yapısı
- `aktivasyon.py`, `kayip.py`: sigmoid ve MSE
- `egitim.py`: eğitim sarmalayıcısı
- `ve_kapisi.py`: AND kapısı eğitim verisi

## Sınırlar

Bu, küçük ve açıklanabilir bir eğitim uygulamasıdır; üretim makine öğrenmesi çerçevesi veya genel amaçlı dil modeli değildir. AND kapısına ait test, yalnızca bu küçük örneğin beklenen davranışını doğrular.

**Atölye:** Ergin Teknik Atölye
