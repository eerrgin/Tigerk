"""Geri yayılımın, türevin ve öğrenme oranının somut anlatımı.

Sayılar kodun o an ürettiği değerlerdir; formül metne sonradan yapıştırılmaz.
"""
from aktivasyon import sigmoid, sigmoid_turev_cikti
from ag import Ag
from kayip import mse
from ve_kapisi import VERI


GIRIS = """\
Neden geri yayılım?
Hata yalnız çıkışta ölçülür. Değişecek sayılar ise her katmanın ağırlığı ve eşiğidir.
Bir ağırlığı azıcık oynatıp hatayı yeniden ölçmek türevi verir; buna sayısal türev denir.
Ağırlık başına ayrı bir ileri geçiş ister. Geri yayılım aynı türevi zincir kuralıyla,
çıkıştan girişe tek geçişte taşır. Tigerk bu yüzden geri yayılım kullanır.

Türev nasıl kurulur?
Nöron önce toplar: z = w·x + b. Sonra y = σ(z).
σ'(z) = y·(1-y). Çıktı elde olduğu için kod σ'yu yeniden hesaplamaz.
Eğitim kaybı L = ½(y - hedef)². Baştaki ½, türevin getireceği 2'yi götürür.
Ekrandaki MSE bu ½ olmadan alınmış ortalamadır. İniş yönü aynıdır;
aradaki sabit, öğrenme oranının içine karışır.
  ∂L/∂y = y - hedef
  delta = ∂L/∂z = (y - hedef) · y · (1-y)
  ∂L/∂w = delta · x
  ∂L/∂b = delta
Gizli nörona hata, sonraki katmanın ağırlıklarıyla geri toplanır:
  delta = y·(1-y) · Σ (w_sonraki · delta_sonraki)

Öğrenme oranı neyi değiştirir?
Güncelleme şudur: w ← w - oran · ∂L/∂w
Oran yön seçmez. Yönü türev belirler. Oran yalnız adımın boyunu değiştirir.
Küçük oran kısa adım atar, iniş yavaştır. Büyük oran aynı türevi abartır;
hedefi aşıp hatayı yeniden büyütebilir.
"""


def calisma_izi(tohum=7, ornek=1, oran=0.7):
    """Eğitilmemiş bir ağda tek AND örneğinin adımını çıkarır, ağırlığı geri yükler."""
    ag = Ag(2, [4, 1], tohum=tohum)
    goruntu = ag.anlik_goruntu()
    girdiler, hedef = VERI[ornek]
    iz = ag.egit_ornegi(girdiler, hedef, oran=oran, iz=True)
    ag.yukle(goruntu)
    return iz


def _f(sayi):
    return f"{sayi:.6f}"


def _z(agirliklar, esik, girdiler):
    return sum(w * x for w, x in zip(agirliklar, girdiler)) + esik


def _cumle_adim(once, sonra):
    if sonra > once * 1.02:
        return "hata büyüdü; adım hedefi aştı"
    if sonra < once * 0.98:
        return "hata indi"
    return "hata bu tek adımda neredeyse yerinde kaldı"


def tek_adim_oranlari(tohum=7, ornek=1, oranlar=(0.05, 0.7, 80.0)):
    """Aynı başlangıçtan, aynı örnekte, yalnız oranı değiştirerek bir adım ölçer."""
    satirlar = []
    for oran in oranlar:
        ag = Ag(2, [4, 1], tohum=tohum)
        x, y = VERI[ornek]
        once = mse(ag.ileri(x), y)
        ag.egit_ornegi(x, y, oran=oran)
        sonra = mse(ag.ileri(x), y)
        satirlar.append((oran, once, sonra))
    return satirlar


def kisa_egitim(tohum=7, oranlar=(0.05, 0.7, 80.0), tur=400):
    """Aynı tohumla kısa eğitim; oranlar arasındaki sonucu MSE ile gösterir."""
    satirlar = []
    for oran in oranlar:
        ag = Ag(2, [4, 1], tohum=tohum)
        bas = sum(mse(ag.ileri(x), y) for x, y in VERI) / len(VERI)
        ag.egit(VERI, tur_sayisi=tur, oran=oran, rapor=0)
        bit = sum(mse(ag.ileri(x), y) for x, y in VERI) / len(VERI)
        dogru = sum(int(ag.tahmin(x)[0] >= 0.5) == int(y[0]) for x, y in VERI)
        satirlar.append((oran, bas, bit, dogru))
    return satirlar


def _ornek_bolumu(iz):
    satirlar = []
    girdi = iz["girdiler"]
    hedef = iz["hedef"]
    oran = iz["oran"]
    cikti = iz["noronlar"][-1][0]
    y = cikti["y"]
    z = _z(cikti["w_once"], cikti["b_once"], iz["aktivasyonlar"][-2])
    eps = 1e-5
    sayisal_sig = (sigmoid(z + eps) - sigmoid(z - eps)) / (2.0 * eps)
    analitik_sig = sigmoid_turev_cikti(y)
    formul_delta = (y - hedef[0]) * analitik_sig
    kod_delta = cikti["delta"]

    satirlar.append("Bir örnek, eğitilmemiş ağ (tohum 7).")
    satirlar.append(f"Girdi {_f(girdi[0])}, {_f(girdi[1])}   hedef {_f(hedef[0])}")
    satirlar.append(f"Çıkış: z = {_f(z)}   y = σ(z) = {_f(y)}")
    satirlar.append(
        f"σ' kodda y·(1-y) = {_f(analitik_sig)}; "
        f"sayısal türev (σ(z+ε)-σ(z-ε))/(2ε) = {_f(sayisal_sig)}; "
        f"fark {abs(analitik_sig - sayisal_sig):.2e}"
    )
    satirlar.append(f"∂L/∂y = y - hedef = {_f(y - hedef[0])}")
    satirlar.append(f"delta = ∂L/∂y · σ' = {_f(formul_delta)}")
    if abs(formul_delta - kod_delta) > 1e-9:
        satirlar.append(
            f"Uyarı: kodun deltası {_f(kod_delta)} formülle uyuşmuyor. "
            "Çıkış türevi bozuk olabilir."
        )
    else:
        satirlar.append(f"Kodun yazdığı delta da {_f(kod_delta)}. Formülle aynı.")
    satirlar.append(f"Bu adımdaki MSE = {_f(iz['hata'])}   L = ½(y-hedef)² = {_f(iz['kayip'])}")
    satirlar.append("Çıkış nöronunun ağırlık türevleri (∂L/∂w = delta · x):")
    for i, (x, tw, once, sonra) in enumerate(zip(
        iz["aktivasyonlar"][-2], cikti["turev_w"], cikti["w_once"], cikti["w_sonra"]
    )):
        satirlar.append(
            f"  girdi {_f(x)}  türev {_f(tw)}  "
            f"w {_f(once)} → {_f(sonra)}  (adım {_f(sonra - once)})"
        )
    satirlar.append(
        f"  eşik türevi = delta = {_f(cikti['turev_b'])}  "
        f"b {_f(cikti['b_once'])} → {_f(cikti['b_sonra'])}"
    )
    sifirlar = [i for i, x in enumerate(iz["girdiler"]) if x == 0.0]
    if sifirlar:
        ornek_noron = iz["noronlar"][0][0]
        i = sifirlar[0]
        satirlar.append(
            f"Ham girdi {i} bu örnekte 0. ∂L/∂w = delta·x olduğu için "
            f"gizli nöron 0'ın bu girdisindeki ağırlık kıpırdamaz: "
            f"{_f(ornek_noron['w_once'][i])} → {_f(ornek_noron['w_sonra'][i])}."
        )

    gizli = iz["noronlar"][0]
    secilen = max(range(len(gizli)), key=lambda i: abs(gizli[i]["delta"]))
    n = gizli[secilen]
    formul_g = n["aktarilan"] * n["y"] * (1.0 - n["y"])
    satirlar.append(
        f"Gizli nöron {secilen}: y = {_f(n['y'])}  "
        f"sonraki katmandan gelen = {_f(n['aktarilan'])}"
    )
    satirlar.append(
        f"  delta = gelen · y·(1-y) = {_f(formul_g)}"
    )
    if abs(formul_g - n["delta"]) > 1e-9:
        satirlar.append(
            f"Uyarı: kodun gizli deltası {_f(n['delta'])} bu formülle uyuşmuyor."
        )
    else:
        satirlar.append(f"  Kodun gizli deltası {_f(n['delta'])}. Formülle aynı.")
    diger = ", ".join(f"{i}:{_f(h['delta'])}" for i, h in enumerate(gizli))
    satirlar.append(f"Tüm gizli deltalar: {diger}")

    # Oran, aynı türev üzerinde adımı doğrusal büyütür. Ağırlık güncellenmiş
    # hali izde oran ile kaydedilmiştir; burada türevden yeniden kurulur.
    tw = cikti["turev_w"]
    bag = max(range(len(tw)), key=lambda i: abs(tw[i]))
    satirlar.append(
        f"Aynı türev, üç farklı oran. Bakılan yer: çıkış ağırlığı, girdi {bag}, "
        f"türev {_f(tw[bag])}, eski w {_f(cikti['w_once'][bag])}."
    )
    for deneme in (0.05, oran, oran * 2):
        adim = -deneme * tw[bag]
        satirlar.append(
            f"  oran {_f(deneme)}  adım {_f(adim)}  yeni w {_f(cikti['w_once'][bag] + adim)}"
        )
    satirlar.append("Adım, oran ile doğru orantılı. Yön her satırda aynı.")
    return satirlar


def _oran_bolumu(ornek_no=1):
    satirlar = ["Tek adımda oranın hataya etkisi (aynı tohum, aynı örnek):"]
    tek = tek_adim_oranlari(ornek=ornek_no)
    for oran, once, sonra in tek:
        satirlar.append(
            f"  oran {oran:g}   MSE {_f(once)} → {_f(sonra)}   ({_cumle_adim(once, sonra)})"
        )
    satirlar.append("Kısa eğitimde oranın sonuca etkisi (400 tur, AND kapısının dört örneği):")
    olcum = kisa_egitim()
    for oran, bas, bit, dogru in olcum:
        satirlar.append(
            f"  oran {oran:g}   ortalama MSE {_f(bas)} → {_f(bit)}   doğru sınıf {dogru}/4"
        )
    en_iyi = min(olcum, key=lambda s: s[2])
    en_kotu = max(olcum, key=lambda s: s[2])
    satirlar.append(
        f"Bu tur sayısında en düşük son hata oran {en_iyi[0]:g} ile geldi "
        f"({en_iyi[3]}/4 doğru). En yüksek son hata oran {en_kotu[0]:g} ile kaldı."
    )
    if en_kotu[2] > en_kotu[1]:
        satirlar.append(
            f"Oran {en_kotu[0]:g} hatayı başlangıcın üstüne çıkardı: adım, inişi bozdu."
        )
    tek_ile_inen = {oran for oran, once, sonra in tek if sonra < once}
    turda_bozulan = {oran for oran, bas, bit, _dogru in olcum if bit > bas and oran in tek_ile_inen}
    if turda_bozulan:
        oran = sorted(turda_bozulan)[0]
        satirlar.append(
            f"Oran {oran:g} tek örnekte o örneğin hatasını düşürdü; "
            "dört örnek birlikte eğitilince hatayı büyüttü. "
            "Bir örneğe aşırı uyan adım diğerlerini bozar."
        )
    return satirlar


def metin(tohum=7, ornek=1, oran=0.7):
    iz = calisma_izi(tohum=tohum, ornek=ornek, oran=oran)
    parcalar = ["--- Kodun mantığı ---", GIRIS.rstrip(), ""]
    parcalar.extend(_ornek_bolumu(iz))
    parcalar.append("")
    parcalar.extend(_oran_bolumu(ornek))
    return "\n".join(parcalar)


if __name__ == "__main__":
    print(metin())
