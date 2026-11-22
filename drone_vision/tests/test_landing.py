import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pad_vision import (LandingConfig, LandingController, Observation, TrackerConfig,  # noqa: E402
                        VisionTracker, choose_target, load_record, offset_to_latlon)


# ------------------------------------------------------------------ kayit -> hedef
REC = {"color": "kirmizi", "east_m": 3.0, "north_m": 4.0, "lat_deg": 41.0, "lon_deg": 29.0,
       "n": 5, "spread_m": 0.05, "recorded": True}


class LoadRecordTests(unittest.TestCase):
    def _write(self, data):
        d = tempfile.mkdtemp()
        p = Path(d) / "k.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_en_cok_gozlemli(self):
        a = dict(REC, n=3, east_m=1.0)
        b = dict(REC, n=9, east_m=2.0)
        c = dict(REC, n=50, recorded=False)
        self.assertEqual(load_record(self._write([a, b, c]))["east_m"], 2.0)

    def test_kayitli_yoksa_hata(self):
        with self.assertRaises(ValueError):
            load_record(self._write([dict(REC, recorded=False)]))
        with self.assertRaises(ValueError):
            load_record(self._write([REC]), color="mavi")

    def test_gercek_kayit_dosyasi_bicimi(self):
        real = [{"color": "kirmizi", "east_m": -0.0595, "north_m": 4.8974, "lat_deg": -35.3632182,
                 "lon_deg": 149.1652367, "n": 3, "spread_m": 0.0258, "t_first": 52.7, "t_last": 53.1,
                 "recorded": True}]
        self.assertEqual(load_record(self._write(real))["n"], 3)


class ChooseTargetTests(unittest.TestCase):
    def test_local(self):
        t = choose_target(REC, (0.0, 0.0), None, "local")
        self.assertEqual((t.x, t.y, t.source), (3.0, 4.0, "local"))

    def test_latlon_ofseti(self):
        lat, lon = offset_to_latlon(41.0, 29.0, 10.0, -6.0)      # kayit, GPS'ten (10 D, -6 K)
        rec = dict(REC, lat_deg=lat, lon_deg=lon)
        t = choose_target(rec, (100.0, 50.0), (41.0, 29.0), "latlon")
        self.assertAlmostEqual(t.x, 110.0, delta=0.01)
        self.assertAlmostEqual(t.y, 44.0, delta=0.01)

    def test_auto_uyumluysa_local(self):
        lat, lon = offset_to_latlon(41.0, 29.0, 3.5, 4.0)        # local ile 0.5 m fark
        rec = dict(REC, lat_deg=lat, lon_deg=lon)
        t = choose_target(rec, (0.0, 0.0), (41.0, 29.0), "auto")
        self.assertEqual(t.source, "local")
        self.assertAlmostEqual(t.diff_m, 0.5, delta=0.01)

    def test_auto_uyumsuzsa_latlon_ve_not(self):
        lat, lon = offset_to_latlon(41.0, 29.0, 20.0, 4.0)       # 17 m fark
        rec = dict(REC, lat_deg=lat, lon_deg=lon)
        t = choose_target(rec, (0.0, 0.0), (41.0, 29.0), "auto")
        self.assertEqual(t.source, "latlon")
        self.assertIn("farkli", t.note)

    def test_auto_gps_yoksa_local(self):
        t = choose_target(REC, (0.0, 0.0), None, "auto")
        self.assertEqual(t.source, "local")

    def test_hatalar(self):
        with self.assertRaises(ValueError):
            choose_target(REC, (0, 0), None, "latlon")
        with self.assertRaises(ValueError):
            choose_target({"color": "kirmizi"}, (0, 0), None, "auto")


# ------------------------------------------------------------------ gorsel tahmin
def obs(e=3.0, n=4.0, t=0.0, **kw):
    base = dict(height_m=5.0, confidence=0.9, tilt_deg=1.0, speed_mps=0.2, pose_age_s=0.1)
    base.update(kw)
    return Observation("kirmizi", t, e, n, **base)


class TrackerTests(unittest.TestCase):
    def test_min_ornek(self):
        tr = VisionTracker()
        tr.add(obs(t=0.0), 0.0)
        self.assertIsNone(tr.estimate(0.0))
        tr.add(obs(t=0.2), 0.2)
        self.assertEqual(tr.estimate(0.2), (3.0, 4.0))

    def test_medyan_aykiriyi_eler(self):
        tr = VisionTracker()
        for i, n in enumerate((4.0, 4.02, 3.98, 4.5, 4.01)):
            tr.add(obs(n=n, t=i * 0.2), i * 0.2)
        self.assertAlmostEqual(tr.estimate(1.0)[1], 4.01, places=6)

    def test_pencere_dolar(self):
        tr = VisionTracker(TrackerConfig(window_s=2.0))
        tr.add(obs(t=0.0), 0.0)
        tr.add(obs(t=0.2), 0.2)
        self.assertIsNone(tr.estimate(5.0))

    def test_kapilar(self):
        tr = VisionTracker()
        for kw, why in [(dict(touches_border=True), "kenarda"), (dict(confidence=0.1), "guven"),
                        (dict(tilt_deg=9.0), "egim"), (dict(speed_mps=2.0), "hiz"),
                        (dict(pose_age_s=1.0), "poz_eski"), (dict(height_m=0.1), "yukseklik")]:
            self.assertEqual(tr.add(obs(**kw), 0.0), why)
        self.assertIsNone(tr.estimate(0.0))

    def test_ziplama_ve_sifirlama(self):
        tr = VisionTracker(TrackerConfig(max_jump_m=3.0, reset_after_rejects=3))
        for i in range(4):
            tr.add(obs(t=i * 0.2), i * 0.2)
        # pad gercekten 10 m oteye tasindiysa (ya da kayit yanlissa) pencere sifirlanip yeniye gecer
        res = [tr.add(obs(e=13.0, t=1.0 + i * 0.2), 1.0 + i * 0.2) for i in range(3)]
        self.assertEqual(res, ["ziplama", "ziplama", "ok"])


# ------------------------------------------------------------------ denetleyici (tek adim)
CFG = LandingConfig()


def ctrl(height=5.0, drone=(0.0, 0.0), goal=(0.0, 0.0), cfg=CFG, phase=None, now=0.0):
    c = LandingController(cfg, goal, drone, height, now)
    if phase:
        c.phase = phase
    return c


class ControllerStepTests(unittest.TestCase):
    def test_konik_alcalmayi_durdurur(self):
        c = ctrl(height=4.0, drone=(0.0, 0.0), phase="DESCEND")
        h0 = c.h_target
        cmd = c.update(0.2, (0.0, 0.0), 4.0, est=(2.0, 0.0))        # hata 2 m > izinli 0.6 m
        self.assertEqual(cmd.height_m, h0)
        self.assertIn("ortalaniyor", cmd.note)

    def test_ortaliysa_alcalir(self):
        c = ctrl(height=4.0, phase="DESCEND")
        h0 = c.h_target
        cmd = c.update(0.2, (0.0, 0.0), 4.0, est=(0.1, 0.0))
        self.assertLess(cmd.height_m, h0)

    def test_commit_land(self):
        c = ctrl(height=1.6, phase="DESCEND")
        cmd = c.update(0.2, (0.0, 0.0), 1.55, est=(0.05, 0.0))
        self.assertTrue(cmd.land)
        self.assertEqual(cmd.phase, "LAND")
        self.assertTrue(c.update(0.4, (0.0, 0.0), 1.4, est=None).land)     # LAND kalici

    def test_alt_irtifada_pad_kaybolursa_yukselmez(self):
        c = ctrl(height=2.0, phase="DESCEND")
        c.update(0.0, (0.0, 0.0), 2.0, est=(0.0, 0.0))
        for k in range(1, 40):                                             # 8 sn gorsel tahmin yok
            cmd = c.update(k * 0.2, (0.0, 0.0), 2.0, est=None)
            self.assertNotEqual(cmd.phase, "SEARCH")
        self.assertLessEqual(cmd.height_m, 2.0)

    def test_yuksekte_pad_kaybolursa_arar(self):
        c = ctrl(height=4.0, phase="DESCEND")
        c.update(0.0, (0.0, 0.0), 4.0, est=(0.0, 0.0))
        phases = {c.update(k * 0.2, (0.0, 0.0), 4.0, est=None).phase for k in range(1, 40)}
        self.assertIn("SEARCH", phases)

    def test_xy_hedefi_ani_ziplamaz(self):
        c = ctrl(drone=(0.0, 0.0), goal=(10.0, 0.0))
        cmd = c.update(0.2, (0.0, 0.0), 5.0, est=None)
        self.assertLessEqual(math.hypot(cmd.x, cmd.y), CFG.max_xy_speed_mps * 0.2 + 1e-9)

    def test_gorev_zaman_asimi(self):
        c = ctrl(cfg=LandingConfig(mission_timeout_s=10.0))
        cmd = c.update(11.0, (0.0, 0.0), 5.0, est=None)
        self.assertTrue(cmd.land)


# ------------------------------------------------------------------ kapali dongu benzetimi
def simulate(pad, goal, start=(0.0, 0.0), start_h=5.0, cfg=LandingConfig(), visible=True,
             noise=0.1, seed=1, t_max=200.0, tracker_cfg=None):
    """Basit drone modeli (hedefe orantili hiz, sinirli) + gorsel tahmin = gercek + gurultu.
    Pad, kadraj izdusumundeki (|dx|<0.6h, |dy|<0.45h) ve yukseklik>1.2 ise gorunur."""
    rnd = random.Random(seed)
    dt = 0.2
    x, y, h = start[0], start[1], start_h
    c = LandingController(cfg, goal, (x, y), h, 0.0)
    tr = VisionTracker(tracker_cfg)
    t = 0.0
    cmd = None
    while t < t_max:
        t += dt
        if visible and h > 1.2 and abs(pad[0] - x) < 0.6 * h and abs(pad[1] - y) < 0.45 * h:
            tr.add(Observation("kirmizi", t, pad[0] + rnd.gauss(0, noise), pad[1] + rnd.gauss(0, noise),
                               height_m=h, confidence=0.9, tilt_deg=1.0, speed_mps=0.2, pose_age_s=0.1), t)
        est = tr.estimate(t)
        cmd = c.update(t, (x, y), h, est)
        if cmd.land:
            break
        for axis, tgt, pos in (("x", cmd.x, x), ("y", cmd.y, y)):
            v = max(-2.0, min(2.0, 0.8 * (tgt - pos)))
            if axis == "x":
                x += v * dt
            else:
                y += v * dt
        vz = max(-1.5, min(1.5, 0.8 * (cmd.height_m - h)))
        h = max(0.0, h + vz * dt)
    return c, cmd, (x, y, h), t


class ClosedLoopTests(unittest.TestCase):
    def test_kayit_1_5_m_hatali_olsa_da_pad_merkezine_iner(self):
        pad = (3.0, 4.0)
        goal = (4.2, 4.9)                              # kayit ~1.5 m hatali
        c, cmd, (x, y, h), t = simulate(pad, goal)
        self.assertTrue(cmd.land)
        self.assertTrue(cmd.pad_seen)
        self.assertLess(math.hypot(x - pad[0], y - pad[1]), 0.35, c.log)
        self.assertEqual([l.split()[3] for l in c.log], ["DESCEND", "LAND"])

    def test_kayit_3_m_hatali(self):
        pad = (3.0, 4.0)
        c, cmd, (x, y, h), t = simulate(pad, goal=(5.0, 6.0), t_max=300.0)    # ~2.8 m
        self.assertTrue(cmd.land and cmd.pad_seen)
        self.assertLess(math.hypot(x - pad[0], y - pad[1]), 0.5, c.log)

    def test_baslangicta_kadrajda_degilse_yukselip_bulur(self):
        pad = (3.0, 4.0)
        goal = (3.0, 7.4)                              # kayit 3.4 m hatali, 5 m'de kadrajda degil
        c, cmd, (x, y, h), t = simulate(pad, goal, start=goal, t_max=300.0)
        phases = [l.split()[3] for l in c.log]
        self.assertEqual(phases, ["SEARCH", "DESCEND", "LAND"], c.log)
        self.assertTrue(cmd.pad_seen)
        self.assertLess(math.hypot(x - pad[0], y - pad[1]), 0.35)

    def test_pad_hic_gorunmezse_kor_inis(self):
        pad = (3.0, 4.0)
        goal = (3.4, 4.3)
        c, cmd, (x, y, h), t = simulate(pad, goal, visible=False, t_max=300.0)
        self.assertTrue(cmd.land)
        self.assertFalse(cmd.pad_seen)
        phases = [l.split()[3] for l in c.log]
        self.assertIn("SEARCH", phases)
        self.assertIn("BLIND", phases)
        self.assertLess(math.hypot(x - goal[0], y - goal[1]), 0.5)      # kayitli konuma indi

    def test_gurultulu_tespitle_de_merkezde(self):
        pad = (-2.0, 6.0)
        errs = []
        for seed in range(5):
            c, cmd, (x, y, h), t = simulate(pad, goal=(-1.0, 5.0), noise=0.3, seed=seed, t_max=300.0)
            self.assertTrue(cmd.land)
            errs.append(math.hypot(x - pad[0], y - pad[1]))
        self.assertLess(max(errs), 0.6, errs)


if __name__ == "__main__":
    unittest.main()
