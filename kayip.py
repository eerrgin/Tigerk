"""Kayıp fonksiyonları."""


def mse(tahmin, hedef):
    """Tahmin ve hedef listeleri arasındaki ortalama kare hata.

    Eğitim sırasında ekrana basılan ölçüdür. Geri yayılımın türevini aldığı
    kayıp bu değildir; o kayıp `yari_kare` içindedir. İkisi aynı yönü verir.
    """
    if len(tahmin) != len(hedef):
        raise ValueError("Tahmin ve hedef uzunlukları aynı olmalı.")
    if len(tahmin) == 0:
        raise ValueError("Boş listeler için MSE hesaplanamaz.")
    return sum((t - h) ** 2 for t, h in zip(tahmin, hedef)) / len(tahmin)


def yari_kare(tahmin, hedef):
    """L = ½ Σ (tahmin - hedef)².

    ∂L/∂tahmin = tahmin - hedef. Baştaki ½, türevin getireceği 2'yi götürür.
    """
    if len(tahmin) != len(hedef):
        raise ValueError("Tahmin ve hedef uzunlukları aynı olmalı.")
    if len(tahmin) == 0:
        raise ValueError("Boş listeler için kayıp hesaplanamaz.")
    return 0.5 * sum((t - h) ** 2 for t, h in zip(tahmin, hedef))
