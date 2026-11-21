"""tools/live_locate.py'nin ROS'suz kisimlari: uctan uca dogrulama, teshis, atlama nedenleri."""
import contextlib
import importlib.util
import io
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import synthetic as syn  # noqa: E402
from adapters.pad_watcher import PadWatcher, WatchReport  # noqa: E402
from pad_vision import (CameraModel, PadDetection, expected_radius_px,  # noqa: E402
                        ground_offset_to_pixel, quaternion_from_euler_deg)


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ll = _load(ROOT / "tools" / "live_locate.py")

HFOV = 80.0
CAM = CameraModel.from_hfov(640, 480, HFOV)
TRUTH = (0.0, 5.0)           # pad: 5 m Kuzeyde


def q(yaw=90.0, pitch=0.0, roll=0.0):
    return quaternion_from_euler_deg(roll, pitch, yaw)


def make_report(u, v, radius=40.0, touches=False, color="kirmizi", frame=(640, 480)):
    det = PadDetection(color=color, center_px=(u, v), radius_px=radius, area_px=5000.0,
                       confidence=0.9, circularity=0.95, touches_border=touches)
    return WatchReport(0.0, frame, None, (det,), {})


def synth_report(pose, truth=TRUTH, mount=0.0, **kw):
    """Gercek konuma gore piksel uretir (ground_offset_to_pixel) ve rapora koyar."""
    u, v = ground_offset_to_pixel(truth[0] - pose.x, truth[1] - pose.y, CAM, pose.z, pose.q, mount)
    return make_report(u, v, **kw)


class EvaluateTests(unittest.TestCase):
    def test_perfect_estimate_has_zero_error(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0))
        fix, reason = ll.evaluate_report(synth_report(pose), pose, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(fix.est_east, 0.0, places=8)
        self.assertAlmostEqual(fix.est_north, 5.0, places=8)
        self.assertAlmostEqual(fix.err_m, 0.0, places=8)
        self.assertAlmostEqual(fix.est_off[1], 5.0, places=8)

    def test_moved_and_tilted_drone(self):
        pose = ll.PoseSnapshot(2.0, 1.0, 8.0, q(yaw=33.0, pitch=4.0, roll=-3.0))
        fix, _ = ll.evaluate_report(synth_report(pose), pose, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertLess(fix.err_m, 1e-7)
        self.assertAlmostEqual(fix.true_off[0], -2.0, places=9)
        self.assertAlmostEqual(fix.true_off[1], 4.0, places=9)

    def test_ground_z_shifts_height(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 12.0, q(90.0))
        rep = synth_report(ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0)))      # gercek yukseklik 10 (zemin z=2)
        fix, _ = ll.evaluate_report(rep, pose, pad_truth=TRUTH, hfov_deg=HFOV, ground_z=2.0)
        self.assertLess(fix.err_m, 1e-7)
        self.assertAlmostEqual(fix.height_m, 10.0, places=9)

    def test_height_from_pad_radius_cross_check(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0))
        rep = synth_report(pose, radius=expected_radius_px(10.0, CAM.fx, 1.0))
        fix, _ = ll.evaluate_report(rep, pose, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertAlmostEqual(fix.height_from_pad_m, 10.0, places=6)

    def test_skip_reasons(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0))
        good = synth_report(pose)
        low = ll.PoseSnapshot(0.0, 0.0, 0.4, q(90.0))
        fix, why = ll.evaluate_report(good, low, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertIsNone(fix); self.assertIn("yukseklik", why)

        fix, why = ll.evaluate_report(good, pose, pad_truth=TRUTH, hfov_deg=HFOV, color="mavi")
        self.assertIsNone(fix); self.assertIn("mavi", why)

        border = synth_report(pose, touches=True)
        fix, why = ll.evaluate_report(border, pose, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertIsNone(fix); self.assertIn("kenar", why)
        fix, _ = ll.evaluate_report(border, pose, pad_truth=TRUTH, hfov_deg=HFOV, allow_border=True)
        self.assertIsNotNone(fix)

        none = WatchReport(0.0, (640, 480), None, (), {})
        fix, why = ll.evaluate_report(none, pose, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertIsNone(fix); self.assertIn("gorunmuyor", why)

        steep = ll.PoseSnapshot(0.0, 0.0, 10.0, q(yaw=90.0, roll=80.0))
        fix, why = ll.evaluate_report(good, steep, pad_truth=TRUTH, hfov_deg=HFOV)
        self.assertIsNone(fix); self.assertIn("geometri", why)


class DiagnosisTests(unittest.TestCase):
    POSITIONS = [(0.0, 0.0), (1.0, -2.0), (-3.0, 1.0), (2.0, 3.0)]

    def _stats(self, true_mount, used_mount, height_used_extra=0.0):
        st = ll.ErrorStats()
        for x, y in self.POSITIONS:
            pose = ll.PoseSnapshot(x, y, 10.0, q(90.0))
            rep = synth_report(pose, mount=true_mount)
            used = ll.PoseSnapshot(x, y, 10.0 + height_used_extra, q(90.0))   # yanlis yukseklik simule et
            fix, why = ll.evaluate_report(rep, used, pad_truth=TRUTH, hfov_deg=HFOV,
                                          mount_yaw_deg=used_mount)
            self.assertIsNotNone(fix, why)
            st.add(fix, 0.0)
        return st

    def test_correct_setup_is_consistent(self):
        st = self._stats(0.0, 0.0)
        yaw, cons = st.suggest_mount_yaw(0.0)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertGreater(cons, 0.99)
        self.assertAlmostEqual(st.scale_ratio(), 1.0, places=6)
        self.assertIn("uyumlu", st.summary(0.0))

    def test_wrong_mount_yaw_is_detected(self):
        for true_mount in (90.0, 180.0, -90.0, 30.0):
            with self.subTest(mount=true_mount):
                st = self._stats(true_mount, 0.0)
                yaw, cons = st.suggest_mount_yaw(0.0)
                self.assertAlmostEqual(_wrap(yaw - true_mount), 0.0, places=6)
                self.assertGreater(cons, 0.99)
                self.assertIn("--mount-yaw", st.summary(0.0))

    def test_suggestion_accounts_for_current_mount(self):
        st = self._stats(90.0, 60.0)             # 60 verildi, gercek 90 -> hala 90 onerilmeli
        yaw, _ = st.suggest_mount_yaw(60.0)
        self.assertAlmostEqual(_wrap(yaw - 90.0), 0.0, places=6)

    def test_height_scale_error_is_detected(self):
        st = self._stats(0.0, 0.0, height_used_extra=2.5)     # 12.5 kullanildi, gercek 10 -> 1.25
        self.assertAlmostEqual(st.scale_ratio(), 1.25, places=6)
        self.assertIn("Olcek", st.summary(0.0))
        self.assertIn("%125", st.summary(0.0))

    def test_no_diagnosis_when_pad_is_below_drone(self):
        st = ll.ErrorStats()
        pose = ll.PoseSnapshot(0.0, 5.0, 10.0, q(90.0))       # pad'in tam ustunde
        fix, _ = ll.evaluate_report(synth_report(pose), pose, pad_truth=TRUTH, hfov_deg=HFOV)
        st.add(fix, 0.0)
        self.assertIsNone(st.suggest_mount_yaw())
        self.assertIsNone(st.scale_ratio())
        self.assertIn("en az", st.summary())

    def test_summary_groups_stationary(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0))
        fix, _ = ll.evaluate_report(synth_report(pose), pose, pad_truth=TRUTH, hfov_deg=HFOV)
        st = ll.ErrorStats(stationary_speed=0.3)
        st.add(fix, 0.05)
        st.add(fix, 2.0)
        st.add(fix, None)
        text = st.summary()
        self.assertIn("durgun: 1", text)
        self.assertIn("[durgun]", text)
        self.assertIn("ortalama hata", text)

    def test_empty_summary(self):
        self.assertIn("Hic gecerli ornek yok", ll.ErrorStats().summary())


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class SpeedTests(unittest.TestCase):
    def test_moving(self):
        s = ll.SpeedEstimator()
        for i in range(11):
            s.add(i * 0.1, i * 0.1 * 1.0, 0.0, 0.0)       # 1 m/s Dogu
        self.assertAlmostEqual(s.speed(), 1.0, places=6)

    def test_still(self):
        s = ll.SpeedEstimator()
        for i in range(11):
            s.add(i * 0.1, 3.0, 4.0, 5.0)
        self.assertAlmostEqual(s.speed(), 0.0, places=9)

    def test_insufficient_data(self):
        s = ll.SpeedEstimator()
        self.assertIsNone(s.speed())
        s.add(0.0, 0.0, 0.0, 0.0)
        self.assertIsNone(s.speed())
        s.add(0.05, 1.0, 0.0, 0.0)                        # aralik cok kisa
        self.assertIsNone(s.speed())

    def test_window_drops_old_samples(self):
        s = ll.SpeedEstimator(window_s=1.0)
        for i in range(20):                                # once hizli, sonra durgun
            s.add(i * 0.1, min(i, 5) * 1.0, 0.0, 0.0)
        self.assertLess(s.speed(), 0.2)


class FormatAndParseTests(unittest.TestCase):
    def test_format_and_csv(self):
        pose = ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0))
        fix, _ = ll.evaluate_report(synth_report(pose), pose, pad_truth=TRUTH, hfov_deg=HFOV)
        line = ll.format_fix(12.3, fix, 0.02, pose)
        self.assertIn("[LOK]", line)
        self.assertIn("KIRMIZI", line)
        self.assertIn("yaw=+90", line)
        row = ll.csv_row(12.3, fix, None, pose)
        self.assertEqual(set(row), set(ll.CSV_FIELDS))
        self.assertEqual(row["speed"], "")

    def test_parse_requires_pad_position(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                ll.parse([])
            with self.assertRaises(SystemExit):
                ll.parse(["--pad-east", "0"])

    def test_parse_defaults(self):
        a = ll.parse(["--pad-east", "0", "--pad-north", "5"])
        self.assertEqual((a.pad_east, a.pad_north), (0.0, 5.0))
        self.assertEqual(a.hfov, 80.0)
        self.assertEqual(a.mount_yaw, 0.0)
        self.assertEqual(a.color, "kirmizi")
        self.assertFalse(a.use_hint)

    def test_quaternion_helpers(self):
        self.assertAlmostEqual(ll.quaternion_yaw_deg(q(90.0)), 90.0, places=6)
        self.assertAlmostEqual(ll.quaternion_yaw_deg(q(-120.0)), -120.0, places=6)
        self.assertAlmostEqual(ll.quaternion_tilt_deg(q(40.0, pitch=10.0)), 10.0, places=6)
        self.assertAlmostEqual(ll.quaternion_tilt_deg(q(0.0, roll=-7.0)), 7.0, places=6)


class EndToEndWithDetectorTests(unittest.TestCase):
    """Sentetik KARE -> gercek dedektor -> geometri -> hata: tum zincir."""

    def _run(self, pose, mount=0.0):
        u, v = ground_offset_to_pixel(TRUTH[0] - pose.x, TRUTH[1] - pose.y, CAM, pose.z, pose.q, mount)
        r = expected_radius_px(pose.z, CAM.fx, 1.0)
        img = syn.disc(syn.blank(640, 480), (u, v), r, syn.RED_BGR)
        w = PadWatcher(ll.HsvPadDetector(), None, every_s=0.0, log=lambda s: None)
        w.on_image(syn.msg_from_bgr(img))
        rep = w.update(None)
        self.assertIsNotNone(rep)
        return ll.evaluate_report(rep, pose, pad_truth=TRUTH, hfov_deg=HFOV, mount_yaw_deg=mount)

    def test_level_drone_10m(self):
        fix, why = self._run(ll.PoseSnapshot(0.0, 0.0, 10.0, q(90.0)))
        self.assertIsNotNone(fix, why)
        self.assertLess(fix.err_m, 0.15)          # piksel kuantalama + 320 px islem olcegi
        self.assertAlmostEqual(fix.height_from_pad_m, 10.0, delta=0.6)

    def test_tilted_and_offset_drone(self):
        # pad bu konumda kadraja tam sigar (daha uzaktan kadraj kenarina tasar ve dedektor reddeder)
        fix, why = self._run(ll.PoseSnapshot(1.5, 2.0, 9.0, q(yaw=70.0, pitch=3.0, roll=-2.0)))
        self.assertIsNotNone(fix, why)
        self.assertLess(fix.err_m, 0.2)


if __name__ == "__main__":
    unittest.main()
