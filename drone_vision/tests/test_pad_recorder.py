"""tools/pad_recorder.py'nin ROS'suz kismi: gozlem uretimi + PadMap ile uctan uca kayit."""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

from pad_vision import (CameraModel, PadMap, ground_offset_to_pixel,  # noqa: E402
                        offset_to_latlon, quaternion_from_euler_deg)

spec = importlib.util.spec_from_file_location("pad_recorder", ROOT / "tools" / "pad_recorder.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)
from live_locate import PoseSnapshot  # noqa: E402

HFOV = 80.0
CAM = CameraModel.from_hfov(640, 480, HFOV)
LEVEL = (0.0, 0.0, 0.0, 1.0)
PAD = (1.5, 5.0)           # gercek pad (Dogu, Kuzey)
LAT0, LON0 = 41.0, 29.0    # drone'un (0,0) yerel noktasinin enlem-boylami


class ObservationTests(unittest.TestCase):
    def test_yukseklik_dusuk(self):
        from adapters.pad_watcher import WatchReport
        rep = WatchReport(0.0, (640, 480), 0.5, (), {})
        o, why = pr.make_observation(rep, PoseSnapshot(0, 0, 0.5, LEVEL), t=1.0, pose_age_s=0.1,
                                     speed=0.1, hfov_deg=HFOV)
        self.assertIsNone(o)
        self.assertIn("yukseklik", why)

    def test_pad_gorunmuyor(self):
        from adapters.pad_watcher import WatchReport
        rep = WatchReport(0.0, (640, 480), 10.0, (), {})
        o, why = pr.make_observation(rep, PoseSnapshot(0, 0, 10, LEVEL), t=1.0, pose_age_s=0.1,
                                     speed=0.1, hfov_deg=HFOV)
        self.assertIsNone(o)
        self.assertIn("gorunmuyor", why)

    def test_gozlem_konum_ve_enlem_boylam(self):
        from pad_vision import PadDetection
        from adapters.pad_watcher import WatchReport
        pose = PoseSnapshot(1.5, 2.0, 9.0, LEVEL)
        u, v = ground_offset_to_pixel(PAD[0] - pose.x, PAD[1] - pose.y, CAM, pose.z, LEVEL)
        det = PadDetection("kirmizi", (u, v), CAM.fx / pose.z, math.pi * (CAM.fx / pose.z) ** 2,
                           0.95, 0.99, False)
        rep = WatchReport(0.0, (640, 480), pose.z, (det,), {})
        o, why = pr.make_observation(rep, pose, t=2.0, pose_age_s=0.1, speed=0.2, hfov_deg=HFOV,
                                     latlon=(LAT0, LON0))
        self.assertEqual(why, "")
        self.assertAlmostEqual(o.east_m, PAD[0], delta=0.05)
        self.assertAlmostEqual(o.north_m, PAD[1], delta=0.05)
        exp_lat, exp_lon = offset_to_latlon(LAT0, LON0, PAD[0] - pose.x, PAD[1] - pose.y)
        self.assertAlmostEqual(o.lat_deg, exp_lat, places=8)
        self.assertAlmostEqual(o.lon_deg, exp_lon, places=8)
        self.assertEqual(o.tilt_deg, 0.0)


class EndToEndTests(unittest.TestCase):
    def _obs_at(self, x, y, z, speed, pad=PAD, tilt_deg=0.0):
        from pad_vision import PadDetection
        from adapters.pad_watcher import WatchReport
        q = quaternion_from_euler_deg(0.0, tilt_deg, 90.0)
        pose = PoseSnapshot(x, y, z, q)
        u, v = ground_offset_to_pixel(pad[0] - x, pad[1] - y, CAM, z, q)
        det = PadDetection("kirmizi", (u, v), CAM.fx / z, math.pi * (CAM.fx / z) ** 2, 0.95, 0.99, False)
        rep = WatchReport(0.0, (640, 480), z, (det,), {})
        return pr.make_observation(rep, pose, t=1.0, pose_age_s=0.1, speed=speed, hfov_deg=HFOV)

    def test_gecis_kaydi_ve_json(self):
        pm = PadMap()
        # drone pad'in yakininda 4 yavas gecis + 1 hizli/egimli gecis (elenmeli)
        for (x, y, spd, tilt) in [(1.5, 2.0, 0.5, 0.0), (1.5, 2.5, 0.6, 0.0), (1.5, 3.0, 0.7, 0.0),
                                  (1.5, 3.5, 3.5, 8.0), (1.5, 4.0, 0.4, 0.0)]:
            o, why = self._obs_at(x, y, 9.0, spd, tilt_deg=tilt)
            self.assertIsNotNone(o, why)
            pm.add(o)
        rec = pm.best("kirmizi")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.n, 4)                       # hizli/egimli elendi
        self.assertLess(math.hypot(rec.east_m - PAD[0], rec.north_m - PAD[1]), 0.1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "kayit.json"
            pr.save_json(pm, str(p))
            data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data[0]["n"], 4)

    def test_ozet_ve_hata_yazar(self):
        pm = PadMap()
        for y in (2.0, 2.5, 3.0):
            o, _ = self._obs_at(1.5, y, 9.0, 0.3)
            pm.add(o)
        s = pr.summary(pm, PAD, {"hiz": 2})
        self.assertIn("KAYDEDILDI", s)
        self.assertIn("HATA=", s)
        self.assertIn("hiz=2", s)

    def test_bos_ozet(self):
        self.assertIn("Hic pad", pr.summary(PadMap(), None, {}))


class ParseTests(unittest.TestCase):
    def test_varsayilanlar(self):
        a = pr.parse([])
        self.assertIsNone(a.pad_east)
        self.assertEqual(a.max_tilt, 4.0)
        self.assertEqual(a.max_speed, 1.5)
        self.assertEqual(a.min_obs, 3)

    def test_gercek_konum(self):
        a = pr.parse(["--pad-east", "0", "--pad-north", "5", "--out", "k.json", "--publish"])
        self.assertEqual((a.pad_east, a.pad_north), (0.0, 5.0))
        self.assertTrue(a.publish)


if __name__ == "__main__":
    unittest.main()
