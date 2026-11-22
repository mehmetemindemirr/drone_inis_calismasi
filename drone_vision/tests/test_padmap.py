import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pad_vision import Observation, PadMap, PadMapConfig, offset_to_latlon  # noqa: E402

# live_locate'in gercek Gazebo ciktisi (sonuc.csv): gercek pad = (0, 5)
# (t, tahmin_dogu, tahmin_kuzey, hiz, egim, yukseklik)
CSV = [
    (107.44, -0.018, 5.048, 0.714, 0.76, 9.712),
    (109.64, -0.049, 3.861, 3.697, 7.64, 9.756),   # hizli + egimli: 1.14 m hata
    (112.44, -0.045, 5.283, 0.254, 2.92, 10.094),
    (113.04, -0.046, 5.063, 0.483, 2.33, 9.625),
]


def obs(color="kirmizi", e=0.0, n=5.0, **kw):
    base = dict(t=1.0, height_m=10.0, confidence=0.9, tilt_deg=1.0, speed_mps=0.3, pose_age_s=0.1)
    base.update(kw)
    return Observation(color, east_m=e, north_m=n, **base)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.pm = PadMap()

    def reason(self, **kw):
        return self.pm.add(obs(**kw)).reason

    def test_ok(self):
        self.assertEqual(self.reason(), "ok")

    def test_kenar(self):
        self.assertEqual(self.reason(touches_border=True), "kenarda")

    def test_egim(self):
        self.assertEqual(self.reason(tilt_deg=7.6), "egim")

    def test_hiz(self):
        self.assertEqual(self.reason(speed_mps=3.7), "hiz")

    def test_poz_eski(self):
        self.assertEqual(self.reason(pose_age_s=0.9), "poz_eski")

    def test_yukseklik(self):
        self.assertEqual(self.reason(height_m=0.8), "yukseklik")
        self.assertEqual(self.reason(height_m=120.0), "yukseklik")

    def test_guven(self):
        self.assertEqual(self.reason(confidence=0.2), "guven")

    def test_konum_yok(self):
        r = self.pm.add(Observation("kirmizi", t=1.0))
        self.assertEqual(r.reason, "konum_yok")

    def test_bilinmeyen_alan_kapidan_gecer(self):
        r = self.pm.add(Observation("kirmizi", t=1.0, east_m=0.0, north_m=5.0))
        self.assertTrue(r.accepted)

    def test_reddedilen_kayda_girmez(self):
        self.pm.add(obs(speed_mps=9.0))
        self.assertEqual(self.pm.records(), [])


class MedianTests(unittest.TestCase):
    def test_gercek_csv(self):
        pm = PadMap()
        res = [pm.add(Observation("kirmizi", t, e, n, height_m=h, confidence=0.96,
                                  tilt_deg=tilt, speed_mps=v, pose_age_s=0.1))
               for t, e, n, v, tilt, h in CSV]
        # hizli ornek hem egim (7.6 > 4) hem hiz (3.7 > 1.5) kapisina takilir; egim once kontrol edilir
        self.assertEqual([r.reason for r in res], ["ok", "egim", "ok", "ok"])
        rec = pm.best("kirmizi")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.n, 3)
        self.assertTrue(rec.recorded)
        self.assertLess(math.hypot(rec.east_m - 0.0, rec.north_m - 5.0), 0.15)

    def test_medyan_aykiriyi_eler(self):
        pm = PadMap(PadMapConfig(min_observations=3))
        for n in (5.0, 5.05, 4.95, 5.02, 4.98):
            pm.add(obs(n=n))
        pm.add(obs(n=6.5))                      # kume yaricapi icinde ama aykiri
        rec = pm.best("kirmizi")
        self.assertLess(abs(rec.north_m - 5.0), 0.05)

    def test_kaydedildi_esigi(self):
        pm = PadMap(PadMapConfig(min_observations=3))
        pm.add(obs())
        pm.add(obs())
        self.assertIsNone(pm.best("kirmizi"))
        self.assertIsNotNone(pm.best("kirmizi", recorded_only=False))
        pm.add(obs())
        self.assertTrue(pm.best("kirmizi").recorded)

    def test_dagilim(self):
        pm = PadMap()
        for dn in (-0.2, 0.0, 0.2):
            pm.add(obs(n=5.0 + dn))
        self.assertAlmostEqual(pm.best("kirmizi").spread_m, 0.2, places=6)

    def test_bellek_siniri(self):
        pm = PadMap(PadMapConfig(max_per_cluster=10))
        for i in range(50):
            pm.add(obs(t=float(i)))
        self.assertEqual(pm.best("kirmizi").n, 10)


class ClusterTests(unittest.TestCase):
    def test_iki_ayri_pad_ayni_renk(self):
        pm = PadMap(PadMapConfig(cluster_radius_m=2.0))
        for _ in range(4):
            pm.add(obs(e=0.0, n=5.0))
        for _ in range(3):
            pm.add(obs(e=0.0, n=25.0))
        recs = pm.records("kirmizi")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].n, 4)               # en cok gozlemli once
        self.assertAlmostEqual(pm.best("kirmizi").north_m, 5.0)

    def test_renkler_ayri(self):
        pm = PadMap()
        for _ in range(3):
            pm.add(obs("kirmizi"))
            pm.add(obs("mavi", e=10.0))
        self.assertAlmostEqual(pm.best("kirmizi").east_m, 0.0)
        self.assertAlmostEqual(pm.best("mavi").east_m, 10.0)

    def test_latlon_ve_yerel_birlikte(self):
        lat0, lon0 = 41.0, 29.0
        la, lo = offset_to_latlon(lat0, lon0, 3.0, 4.0)
        pm = PadMap()
        for _ in range(3):
            pm.add(Observation("kirmizi", 1.0, 3.0, 4.0, la, lo, 10.0, 0.9))
        rec = pm.best("kirmizi")
        self.assertAlmostEqual(rec.lat_deg, la, places=9)
        self.assertAlmostEqual(rec.lon_deg, lo, places=9)
        self.assertAlmostEqual(rec.east_m, 3.0)

    def test_cerceve_karistirilmaz(self):
        # yalniz yerel olan gozlem, yalniz enlem-boylam tasiyan kumeye katilmaz
        pm = PadMap()
        pm.add(Observation("kirmizi", 1.0, lat_deg=41.0, lon_deg=29.0))
        pm.add(Observation("kirmizi", 1.0, east_m=0.0, north_m=0.0))
        self.assertEqual(len(pm.records("kirmizi")), 2)


class OutputTests(unittest.TestCase):
    def test_json(self):
        pm = PadMap()
        for _ in range(3):
            pm.add(obs())
        data = json.loads(pm.to_json())
        self.assertEqual(len(data), 1)
        for k in ("color", "lat_deg", "lon_deg", "east_m", "north_m", "n", "spread_m", "recorded"):
            self.assertIn(k, data[0])
        self.assertEqual(json.loads(pm.best("kirmizi").to_json())["n"], 3)

    def test_add_sonucu_guncel_kayit(self):
        pm = PadMap()
        r1 = pm.add(obs())
        r2 = pm.add(obs())
        self.assertEqual((r1.record.n, r2.record.n), (1, 2))

    def test_clear(self):
        pm = PadMap()
        pm.add(obs())
        pm.add(obs("mavi"))
        pm.clear("kirmizi")
        self.assertEqual(len(pm.records()), 1)
        pm.clear()
        self.assertEqual(pm.records(), [])


if __name__ == "__main__":
    unittest.main()
