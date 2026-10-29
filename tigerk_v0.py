"""Tigerk'in tek dosyalık öğretici sürümü; AND kapısını öğrenir."""
import math
import random

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

class Ag:
    def __init__(self, giris_sayisi, katman_yapisi, tohum=7):
        self.rng = random.Random(tohum)
        self.katmanlar = []
        onceki = giris_sayisi
        for adet in katman_yapisi:
            katman = []
            for _ in range(adet):
                katman.append({
                    "w": [self.rng.uniform(-1.0, 1.0) for _ in range(onceki)],
                    "b": self.rng.uniform(-1.0, 1.0),
                    "y": 0.0,
                })
            self.katmanlar.append(katman)
            onceki = adet

    def ileri(self, girdiler):
        a = list(girdiler)
        for katman in self.katmanlar:
            yeni = []
            for n in katman:
                n["y"] = sigmoid(sum(x * w for x, w in zip(a, n["w"])) + n["b"])
                yeni.append(n["y"])
            a = yeni
        return a

    def egit(self, veri, tur_sayisi=6000, oran=0.7):
        for _ in range(tur_sayisi):
            sirali = list(veri)
            self.rng.shuffle(sirali)
            for x, hedef in sirali:
                aktivasyonlar = [list(x)]
                for katman in self.katmanlar:
                    a = aktivasyonlar[-1]
                    yeni = []
                    for n in katman:
                        n["y"] = sigmoid(sum(v * w for v, w in zip(a, n["w"])) + n["b"])
                        yeni.append(n["y"])
                    aktivasyonlar.append(yeni)

                # L = ½(y-hedef)² ⇒ delta = ∂L/∂z = (y-hedef) · y · (1-y)
                # Gizli katman bu deltayı sonraki ağırlıklarla geri taşır.
                deltalar = [None] * len(self.katmanlar)
                deltalar[-1] = [
                    (n["y"] - h) * n["y"] * (1.0 - n["y"])
                    for n, h in zip(self.katmanlar[-1], hedef)
                ]
                for li in range(len(self.katmanlar) - 2, -1, -1):
                    mevcut = self.katmanlar[li]
                    sonraki = self.katmanlar[li + 1]
                    deltalar[li] = [
                        n["y"] * (1.0 - n["y"]) *
                        sum(sonraki[j]["w"][i] * deltalar[li + 1][j]
                            for j in range(len(sonraki)))
                        for i, n in enumerate(mevcut)
                    ]

                # w ← w - oran·delta·girdi. oran adım boyudur, yönü türev belirler.
                for li, katman in enumerate(self.katmanlar):
                    girisler = aktivasyonlar[li]
                    for n, delta in zip(katman, deltalar[li]):
                        for i, deger in enumerate(girisler):
                            n["w"][i] -= oran * delta * deger
                        n["b"] -= oran * delta

if __name__ == "__main__":
    veri = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [0.0]),
        ([1.0, 0.0], [0.0]),
        ([1.0, 1.0], [1.0]),
    ]
    ag = Ag(2, [4, 1], tohum=7)
    ag.egit(veri, tur_sayisi=6000, oran=0.7)
    print("=== Tigerk | tek dosyalık AND örneği ===")
    for x, y in veri:
        print(f"{x} -> {ag.ileri(x)[0]:.4f} (hedef {int(y[0])})")
    print()
    print("Geri yayılım, çıkıştaki hatayı zincir kuralıyla her ağırlığa taşır.")
    print("delta = (y - hedef) * y * (1 - y)")
    print("w -= oran * delta * girdi    b -= oran * delta")
    print("Öğrenme oranı yön seçmez; adımın boyunu değiştirir.")
    print("Sayılarla anlatım ve hatalı türevi düzeltme: python main.py")
