"""Küçük, tam bağlantılı ve sigmoid aktivasyonlu sinir ağı."""
import random
from katman import Katman
from aktivasyon import sigmoid_turev_cikti
from kayip import mse, yari_kare


class Ag:
    def __init__(self, giris_sayisi, katman_yapisi, tohum=None):
        if giris_sayisi < 1:
            raise ValueError("Girdi sayısı pozitif olmalı.")
        if not katman_yapisi or any(n < 1 for n in katman_yapisi):
            raise ValueError("En az bir katman ve her katmanda pozitif nöron sayısı olmalı.")
        self.rng = random.Random(tohum)
        self.katmanlar = []
        onceki = giris_sayisi
        for noron_sayisi in katman_yapisi:
            self.katmanlar.append(Katman(noron_sayisi, onceki, rng=self.rng))
            onceki = noron_sayisi

    def anlik_goruntu(self):
        """Ağırlık ve eşikleri, adımı geri almak için kopyalar."""
        return [
            [(n.agirliklar[:], n.esik) for n in k.noronlar]
            for k in self.katmanlar
        ]

    def yukle(self, goruntu):
        for katman, kopya in zip(self.katmanlar, goruntu):
            for noron, (agirliklar, esik) in zip(katman.noronlar, kopya):
                noron.agirliklar[:] = agirliklar
                noron.esik = esik

    def ileri(self, girdiler):
        cikti = list(girdiler)
        for katman in self.katmanlar:
            cikti = katman.ileri(cikti)
        return cikti

    def tahmin(self, girdiler):
        return self.ileri(girdiler)

    def egit_ornegi(self, girdiler, hedef, oran=0.3, iz=False):
        """Tek örneği eğitir. iz=True ise ara değerleri de döndürür; güncelleme yine yapılır."""
        if oran <= 0:
            raise ValueError("Öğrenme oranı pozitif olmalı.")
        if len(hedef) != len(self.katmanlar[-1].noronlar):
            raise ValueError("Hedef boyutu çıkış katmanıyla eşleşmeli.")
        aktivasyonlar = [list(girdiler)]
        for katman in self.katmanlar:
            aktivasyonlar.append(katman.ileri(aktivasyonlar[-1]))

        # L = ½ Σ (y - hedef)²
        # ∂L/∂y = y - hedef
        # delta = ∂L/∂z = ∂L/∂y · y·(1-y)
        # Gizli katman: ∂L/∂y, sonraki katmanın ağırlıklarıyla taşınır.
        tum_deltalar = [None] * len(self.katmanlar)
        aktarilanlar = [None] * len(self.katmanlar)
        son = self.katmanlar[-1]
        aktarilanlar[-1] = [
            noron.son_cikti - beklenen
            for noron, beklenen in zip(son.noronlar, hedef)
        ]
        tum_deltalar[-1] = [
            akt * sigmoid_turev_cikti(noron.son_cikti)
            for noron, akt in zip(son.noronlar, aktarilanlar[-1])
        ]
        for li in range(len(self.katmanlar) - 2, -1, -1):
            mevcut = self.katmanlar[li]
            sonraki = self.katmanlar[li + 1]
            deltalar = []
            akt_liste = []
            for i, noron in enumerate(mevcut.noronlar):
                aktarilan = sum(
                    sonraki.noronlar[j].agirliklar[i] * tum_deltalar[li + 1][j]
                    for j in range(len(sonraki.noronlar))
                )
                akt_liste.append(aktarilan)
                deltalar.append(aktarilan * sigmoid_turev_cikti(noron.son_cikti))
            aktarilanlar[li] = akt_liste
            tum_deltalar[li] = deltalar

        noron_iz = []
        for li, katman in enumerate(self.katmanlar):
            onceki_aktivasyonlar = aktivasyonlar[li]
            katman_iz = []
            for ni, (noron, delta) in enumerate(zip(katman.noronlar, tum_deltalar[li])):
                # ∂L/∂w = delta · x ,  ∂L/∂b = delta
                # w ← w - oran · ∂L/∂w   (oran yalnız adım boyunu değiştirir)
                turev_w = [delta * deger for deger in onceki_aktivasyonlar]
                turev_b = delta
                once_w = noron.agirliklar[:]
                once_b = noron.esik
                for i in range(len(turev_w)):
                    noron.agirliklar[i] -= oran * turev_w[i]
                noron.esik -= oran * turev_b
                if iz:
                    katman_iz.append({
                        "w_once": once_w,
                        "b_once": once_b,
                        "w_sonra": noron.agirliklar[:],
                        "b_sonra": noron.esik,
                        "y": noron.son_cikti,
                        "delta": delta,
                        "turev_w": turev_w,
                        "turev_b": turev_b,
                        "aktarilan": aktarilanlar[li][ni],
                    })
            if iz:
                noron_iz.append(katman_iz)

        hata = mse(aktivasyonlar[-1], hedef)
        if not iz:
            return hata
        return {
            "hata": hata,
            "kayip": yari_kare(aktivasyonlar[-1], hedef),
            "girdiler": list(girdiler),
            "hedef": list(hedef),
            "oran": oran,
            "aktivasyonlar": aktivasyonlar,
            "deltalar": tum_deltalar,
            "noronlar": noron_iz,
        }

    def egit(self, veri, tur_sayisi=6000, oran=0.3, rapor=1000):
        if not veri:
            raise ValueError("Eğitim verisi boş olamaz.")
        if tur_sayisi < 1:
            raise ValueError("Tur sayısı en az 1 olmalı.")
        for tur in range(1, tur_sayisi + 1):
            sirali = list(veri)
            self.rng.shuffle(sirali)
            for girdiler, hedef in sirali:
                if len(hedef) != len(self.katmanlar[-1].noronlar):
                    raise ValueError("Hedef boyutu çıkış katmanıyla eşleşmeli.")
                self.egit_ornegi(girdiler, hedef, oran)
            if rapor and (tur == 1 or tur % rapor == 0 or tur == tur_sayisi):
                hata = sum(mse(self.ileri(x), y) for x, y in veri) / len(veri)
                print(f"Tur {tur:5d} | MSE: {hata:.6f}")
