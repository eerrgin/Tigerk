import unittest
from ag import Ag
from kayip import mse, yari_kare
from ve_kapisi import VERI
import mantik
import denetci


class TigerkTests(unittest.TestCase):
    def test_mse(self):
        self.assertAlmostEqual(mse([0.0, 1.0], [0.0, 0.0]), 0.5)

    def test_yari_kare_turev_kaybi(self):
        self.assertAlmostEqual(yari_kare([1.0], [0.0]), 0.5)
        self.assertAlmostEqual(mse([1.0], [0.0]), 1.0)

    def test_mse_rejects_empty(self):
        with self.assertRaises(ValueError):
            mse([], [])

    def test_and_learns(self):
        ag = Ag(2, [4, 1], tohum=7)
        ag.egit(VERI, tur_sayisi=6000, oran=0.7, rapor=0)
        for x, y in VERI:
            self.assertEqual(int(ag.tahmin(x)[0] >= 0.5), int(y[0]))

    def test_egit_ornegi_iz_yoksa_sayi_doner(self):
        ag = Ag(2, [2, 1], tohum=1)
        deger = ag.egit_ornegi([0.2, 0.4], [1.0], oran=0.1, iz=False)
        self.assertIsInstance(deger, float)

    def test_ogrenme_orani_adimi_olcekler(self):
        def adim(oran):
            ag = Ag(2, [2, 1], tohum=1)
            iz = ag.egit_ornegi([0.3, 0.8], [0.0], oran=oran, iz=True)
            noron = iz["noronlar"][0][0]
            return noron["w_sonra"][1] - noron["w_once"][1]

        self.assertAlmostEqual(adim(0.4), 2 * adim(0.2))

    def test_geri_yayilim_sayisal_turevle_uyumlu(self):
        self.assertLess(denetci.gercek_fark(), 1e-4)

    def test_cok_cikis_sayisal_turev(self):
        ag = Ag(2, [3, 2], tohum=3)
        goruntu = ag.anlik_goruntu()
        x, y = [0.2, -0.4], [1.0, 0.0]
        iz = ag.egit_ornegi(x, y, oran=0.1, iz=True)
        ag.yukle(goruntu)
        fark = denetci.en_buyuk_fark(
            denetci.izden_turev(iz),
            denetci.sayisal_turevler(ag, x, y),
        )
        self.assertLess(fark, 1e-4)

    def test_taslak_hatalari_bulunur_ve_duzelir(self):
        ag = Ag(2, [4, 1], tohum=11)
        x, y = [0.3, 0.8], [0.0]
        beklenen = {
            denetci.taslak_sigmoid_yok: "sigmoid_turevi_yok",
            denetci.taslak_isaret_ters: "isaret_ters",
            denetci.taslak_girdi_yok: "girdiyle_carpilmamis",
        }
        for taslak, ad in beklenen.items():
            sonuc = denetci.incele(ag, x, y, taslak)
            self.assertEqual(sonuc["ad"], ad)
            self.assertGreater(sonuc["fark_once"], 1e-3)
            self.assertLess(sonuc["fark_sonra"], 1e-4)

    def test_mantik_metni_kodun_deltasini_yazar(self):
        iz = mantik.calisma_izi()
        govde = mantik.metin()
        delta = iz["noronlar"][-1][0]["delta"]
        self.assertIn(f"{delta:.6f}", govde)
        self.assertIn("Geri yayılım", govde)
        self.assertIn("Öğrenme oranı", govde)
        self.assertIn("σ'(z) = y·(1-y)", govde)
        self.assertIn("inişi bozdu", govde)

    def test_denetim_metni_duzeltmeyi_yazar(self):
        govde = denetci.metin()
        self.assertIn("sigmoid türevi çarpımı eksik", govde)
        self.assertIn("hatanın işareti ters", govde)
        self.assertIn("Düzeltme:", govde)


if __name__ == "__main__":
    unittest.main()
