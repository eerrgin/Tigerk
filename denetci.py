"""Geri yayılımı sayısal türevle yoklar; bozuk bir taslağı tanıyıp düzeltir.

Tanı, taslağın adına bakmaz. Gözlenen türevleri bilinen hesaplarla ve
sayısal türevle karşılaştırır.
"""
from ag import Ag
from kayip import yari_kare


def _anahtarlar(ag):
    for li, katman in enumerate(ag.katmanlar):
        for ni, noron in enumerate(katman.noronlar):
            for wi in range(len(noron.agirliklar)):
                yield ("w", li, ni, wi)
            yield ("b", li, ni, None)


def _oku(ag, anahtar):
    tur, li, ni, wi = anahtar
    noron = ag.katmanlar[li].noronlar[ni]
    if tur == "w":
        return noron.agirliklar[wi]
    return noron.esik


def _yaz(ag, anahtar, deger):
    tur, li, ni, wi = anahtar
    noron = ag.katmanlar[li].noronlar[ni]
    if tur == "w":
        noron.agirliklar[wi] = deger
    else:
        noron.esik = deger


def _ad(anahtar):
    tur, li, ni, wi = anahtar
    if tur == "b":
        return f"eşik, katman {li} nöron {ni}"
    return f"ağırlık, katman {li} nöron {ni} girdi {wi}"


def en_buyuk_fark(a, b):
    anahtarlar = set(a) | set(b)
    if not anahtarlar:
        return 0.0
    return max(abs(a[k] - b[k]) for k in anahtarlar)


def izden_turev(iz):
    turev = {}
    for li, katman in enumerate(iz["noronlar"]):
        for ni, noron in enumerate(katman):
            for wi, deger in enumerate(noron["turev_w"]):
                turev[("w", li, ni, wi)] = deger
            turev[("b", li, ni, None)] = noron["turev_b"]
    return turev


def sayisal_turevler(ag, girdiler, hedef, eps=1e-5):
    """L = ½ Σ (y-hedef)² için merkezi fark. Ağırlıkları yerinde bırakır."""
    goruntu = ag.anlik_goruntu()
    try:
        turev = {}
        for anahtar in _anahtarlar(ag):
            eski = _oku(ag, anahtar)
            _yaz(ag, anahtar, eski + eps)
            arti = yari_kare(ag.ileri(girdiler), hedef)
            _yaz(ag, anahtar, eski - eps)
            eksi = yari_kare(ag.ileri(girdiler), hedef)
            _yaz(ag, anahtar, eski)
            turev[anahtar] = (arti - eksi) / (2.0 * eps)
        return turev
    finally:
        ag.yukle(goruntu)


def katalog_turevleri(ag, girdiler, hedef, kural="dogru"):
    """Bilinen doğru hesap ve sık görülen bozuk hesaplar. Ağırlık güncellemez."""
    aktivasyonlar = [list(girdiler)]
    for katman in ag.katmanlar:
        aktivasyonlar.append(katman.ileri(aktivasyonlar[-1]))

    carpan = 2.0 if kural == "iki_katsayisi" else 1.0
    isaret = -1.0 if kural == "isaret_ters" else 1.0

    def sig(y):
        if kural == "sigmoid_turevi_yok":
            return 1.0
        return y * (1.0 - y)

    n_kat = len(ag.katmanlar)
    deltalar = [None] * n_kat
    son = ag.katmanlar[-1]
    deltalar[-1] = [
        isaret * carpan * (noron.son_cikti - beklenen) * sig(noron.son_cikti)
        for noron, beklenen in zip(son.noronlar, hedef)
    ]
    for li in range(n_kat - 2, -1, -1):
        mevcut = ag.katmanlar[li]
        sonraki = ag.katmanlar[li + 1]
        yeni = []
        for i, noron in enumerate(mevcut.noronlar):
            akt = sum(
                sonraki.noronlar[j].agirliklar[i] * deltalar[li + 1][j]
                for j in range(len(sonraki.noronlar))
            )
            yeni.append(akt * sig(noron.son_cikti))
        deltalar[li] = yeni

    turev = {}
    for li, katman in enumerate(ag.katmanlar):
        xs = aktivasyonlar[li]
        for ni, delta in enumerate(deltalar[li]):
            for wi, x in enumerate(xs):
                if kural == "girdiyle_carpilmamis":
                    turev[("w", li, ni, wi)] = delta
                else:
                    turev[("w", li, ni, wi)] = delta * x
            if kural == "esik_atlandi":
                turev[("b", li, ni, None)] = 0.0
            else:
                turev[("b", li, ni, None)] = delta
    return turev


KURALLAR = (
    "dogru",
    "sigmoid_turevi_yok",
    "isaret_ters",
    "girdiyle_carpilmamis",
    "esik_atlandi",
    "iki_katsayisi",
)

YORUM = {
    "dogru": "kural doğru; türevde düzeltilecek bir şey yok",
    "sigmoid_turevi_yok": "sigmoid türevi çarpımı eksik",
    "isaret_ters": "hatanın işareti ters; iniş yerine çıkış olur",
    "girdiyle_carpilmamis": "ağırlık türevi girdi ile çarpılmamış",
    "esik_atlandi": "eşik türevi sıfır bırakılmış",
    "iki_katsayisi": "½ kare yerine tam karenin türevi kullanılmış (fazladan 2)",
}

DUZELTME = {
    "dogru": "Değişiklik yok.",
    "sigmoid_turevi_yok": "Düzeltme: delta = (y - hedef) · y · (1-y). Gizli katmanda da y·(1-y) ile çarp.",
    "isaret_ters": "Düzeltme: (hedef - y) değil (y - hedef). İşaret ters olursa ağırlık, hatayı azaltan yönün tersine kayar.",
    "girdiyle_carpilmamis": "Düzeltme: ağırlık türevi delta değil, delta · x olmalı.",
    "esik_atlandi": "Düzeltme: eşik türevi 0 değil, delta. Eşik de w gibi güncellenir.",
    "iki_katsayisi": "Düzeltme: ∂L/∂y hesabında fazladan 2 var. L = ½(y-hedef)² ise çarpan (y-hedef) kalır.",
}


def tani(gozlenen, hipotezler, esik=1e-5):
    """Gözlenen türeve en yakın hesabın adını döndürür. Örtüşmezse None."""
    skorlar = []
    for ad, turev in hipotezler.items():
        skorlar.append((en_buyuk_fark(gozlenen, turev), ad))
    skorlar.sort()
    fark, ad = skorlar[0]
    if fark < esik:
        return ad, fark
    return None, fark


def gercek_fark(tohum=7, girdiler=None, hedef=None, oran=0.7):
    """Tigerk'in kendi adımındaki türev ile sayısal türev arasındaki en büyük fark."""
    if girdiler is None:
        girdiler = [0.3, 0.8]
    if hedef is None:
        hedef = [0.0]
    ag = Ag(2, [4, 1], tohum=tohum)
    goruntu = ag.anlik_goruntu()
    iz = ag.egit_ornegi(girdiler, hedef, oran=oran, iz=True)
    ag.yukle(goruntu)
    return en_buyuk_fark(izden_turev(iz), sayisal_turevler(ag, girdiler, hedef))


def _ileri_aktivasyon(ag, girdiler):
    aktivasyonlar = [list(girdiler)]
    for katman in ag.katmanlar:
        aktivasyonlar.append(katman.ileri(aktivasyonlar[-1]))
    return aktivasyonlar


def taslak_sigmoid_yok(ag, girdiler, hedef):
    """Taslak geri geçiş. Kasıtlı hata içerir; türü fonksiyon adından okunup karar verilmez."""
    aktivasyonlar = _ileri_aktivasyon(ag, girdiler)
    n_kat = len(ag.katmanlar)
    deltalar = [None] * n_kat
    deltalar[-1] = [
        noron.son_cikti - beklenen
        for noron, beklenen in zip(ag.katmanlar[-1].noronlar, hedef)
    ]
    for li in range(n_kat - 2, -1, -1):
        sonraki = ag.katmanlar[li + 1]
        yeni = []
        for i, _noron in enumerate(ag.katmanlar[li].noronlar):
            yeni.append(sum(
                sonraki.noronlar[j].agirliklar[i] * deltalar[li + 1][j]
                for j in range(len(sonraki.noronlar))
            ))
        deltalar[li] = yeni
    return _turev_haritasi(aktivasyonlar, deltalar, esik_sifir=False, girdi_yok=False)


def taslak_isaret_ters(ag, girdiler, hedef):
    """İkinci taslak. Kasıtlı hata içerir."""
    aktivasyonlar = _ileri_aktivasyon(ag, girdiler)
    n_kat = len(ag.katmanlar)
    deltalar = [None] * n_kat
    deltalar[-1] = [
        (beklenen - noron.son_cikti) * noron.son_cikti * (1.0 - noron.son_cikti)
        for noron, beklenen in zip(ag.katmanlar[-1].noronlar, hedef)
    ]
    for li in range(n_kat - 2, -1, -1):
        sonraki = ag.katmanlar[li + 1]
        yeni = []
        for i, noron in enumerate(ag.katmanlar[li].noronlar):
            akt = sum(
                sonraki.noronlar[j].agirliklar[i] * deltalar[li + 1][j]
                for j in range(len(sonraki.noronlar))
            )
            y = noron.son_cikti
            yeni.append(akt * y * (1.0 - y))
        deltalar[li] = yeni
    return _turev_haritasi(aktivasyonlar, deltalar, esik_sifir=False, girdi_yok=False)


def taslak_girdi_yok(ag, girdiler, hedef):
    """Üçüncü taslak. Kasıtlı hata içerir."""
    aktivasyonlar = _ileri_aktivasyon(ag, girdiler)
    n_kat = len(ag.katmanlar)
    deltalar = [None] * n_kat
    deltalar[-1] = [
        (noron.son_cikti - beklenen) * noron.son_cikti * (1.0 - noron.son_cikti)
        for noron, beklenen in zip(ag.katmanlar[-1].noronlar, hedef)
    ]
    for li in range(n_kat - 2, -1, -1):
        sonraki = ag.katmanlar[li + 1]
        yeni = []
        for i, noron in enumerate(ag.katmanlar[li].noronlar):
            akt = sum(
                sonraki.noronlar[j].agirliklar[i] * deltalar[li + 1][j]
                for j in range(len(sonraki.noronlar))
            )
            y = noron.son_cikti
            yeni.append(akt * y * (1.0 - y))
        deltalar[li] = yeni
    return _turev_haritasi(aktivasyonlar, deltalar, esik_sifir=False, girdi_yok=True)


def _turev_haritasi(aktivasyonlar, deltalar, esik_sifir, girdi_yok):
    turev = {}
    # aktivasyonlar[0] giriş, deltalar katmanlarla hizalı; katman sayısı = len(deltalar)
    for li, delta_kat in enumerate(deltalar):
        xs = aktivasyonlar[li]
        for ni, delta in enumerate(delta_kat):
            for wi, x in enumerate(xs):
                turev[("w", li, ni, wi)] = delta if girdi_yok else delta * x
            turev[("b", li, ni, None)] = 0.0 if esik_sifir else delta
    return turev


def incele(ag, girdiler, hedef, taslak, oran=0.7):
    """Taslağın türevini tanır, doğru türevle değiştirir, sayısal türevle doğrular."""
    hipotez = {ad: katalog_turevleri(ag, girdiler, hedef, ad) for ad in KURALLAR}
    gozlenen = taslak(ag, girdiler, hedef)
    ad, tani_farki = tani(gozlenen, hipotez)
    sayisal = sayisal_turevler(ag, girdiler, hedef)
    dogru = hipotez["dogru"]
    fark_once = en_buyuk_fark(gozlenen, sayisal)
    fark_sonra = en_buyuk_fark(dogru, sayisal)
    ayrica = max(dogru, key=lambda k: abs(gozlenen[k] - dogru[k]))
    eski = None
    # Ağırlık, katalog ileri geçişinden sonra da aynıdır; güncelleme uygulanmaz.
    # Eski değeri okumak için ağın parametresine bakarız.
    eski = _oku(ag, ayrica)
    return {
        "ad": ad,
        "tani_farki": tani_farki,
        "fark_once": fark_once,
        "fark_sonra": fark_sonra,
        "parametre": ayrica,
        "eski": eski,
        "taslak_turev": gozlenen[ayrica],
        "dogru_turev": dogru[ayrica],
        "oran": oran,
    }


def _satir_inceleme(baslik, sonuc):
    ad = sonuc["ad"]
    oran = sonuc["oran"]
    eski = sonuc["eski"]
    taslak_yeni = eski - oran * sonuc["taslak_turev"]
    dogru_yeni = eski - oran * sonuc["dogru_turev"]
    if ad is None:
        karar = "Bilinen hesapların hiçbiriyle örtüşmedi; elle bakmak gerek."
        duzeltme = "Otomatik düzeltme uygulanmadı."
    else:
        karar = YORUM[ad]
        duzeltme = DUZELTME[ad]
    return [
        baslik,
        f"Örtüşen hesap: {karar}.",
        f"Sayısal türevle en büyük fark, düzeltmeden önce {sonuc['fark_once']:.2e}, "
        f"sonra {sonuc['fark_sonra']:.2e}.",
        duzeltme,
        f"Örnek parametre ({_ad(sonuc['parametre'])}): "
        f"taslak türev {sonuc['taslak_turev']:.6f}, düzeltilmiş türev {sonuc['dogru_turev']:.6f}.",
        f"oran {oran:g} ile {eski:.6f} → taslak {taslak_yeni:.6f}, düzeltilmiş {dogru_yeni:.6f}.",
    ]


def metin(tohum=11):
    girdiler = [0.3, 0.8]
    hedef = [0.0]
    ag = Ag(2, [4, 1], tohum=tohum)
    fark = gercek_fark(tohum=tohum, girdiler=girdiler, hedef=hedef)
    satirlar = [
        "--- Denetim ---",
        "Tigerk'in kendi geri yayılımı, L = ½ Σ (y-hedef)² kaybının sayısal türeviyle karşılaştırıldı.",
        f"En büyük fark: {fark:.2e}.",
    ]
    if fark < 1e-4:
        satirlar.append("Kendi kuralı tutarlı. Burada düzeltme gerekmedi.")
    else:
        goruntu = ag.anlik_goruntu()
        iz = ag.egit_ornegi(girdiler, hedef, oran=0.7, iz=True)
        ag.yukle(goruntu)
        hipotez = {ad: katalog_turevleri(ag, girdiler, hedef, ad) for ad in KURALLAR}
        ad, _ = tani(izden_turev(iz), hipotez)
        if ad is None:
            satirlar.append("Kendi kuralı bilinen bir bozuk hesapla da örtüşmedi.")
        else:
            satirlar.append(f"Kendi kuralı şu hesapla örtüşüyor: {YORUM[ad]}.")
            satirlar.append(DUZELTME[ad])
        satirlar.append(
            "Aşağıdaki taslaklar ayrıdır; çalışan eğitim kodunun yerine geçmez. "
            "Amaç, bozuk bir türevin nasıl yakalandığını göstermektir."
        )

    if fark < 1e-4:
        satirlar.append("Aşağıdaki taslaklar bilerek bozuk. Etiketleri tanıya verilmez.")

    incelemeler = [
        ("Taslak 1.", taslak_sigmoid_yok),
        ("Taslak 2.", taslak_isaret_ters),
        ("Taslak 3.", taslak_girdi_yok),
    ]
    for baslik, fn in incelemeler:
        sonuc = incele(ag, girdiler, hedef, fn)
        satirlar.append("")
        satirlar.extend(_satir_inceleme(baslik, sonuc))
    return "\n".join(satirlar)


if __name__ == "__main__":
    print(metin())
