"""tools/pad_lander.py DUMAN TESTI: sahte rclpy/MAVROS + basit drone fizigi ile dugumun TAMAMI
(WAIT_CONNECT -> ... -> MISSION -> LAND -> DONE) calistirilir. Gercek ROS ya da simulasyon DEGILDIR:
yalnizca isimlendirme/akis hatalarini ve temel davranisi yakalar. Zaman sahte saatle ilerler."""
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from adapters.pad_watcher import WatchReport  # noqa: E402
from pad_vision import (CameraModel, PadDetection, ground_offset_to_pixel,  # noqa: E402
                        offset_to_latlon, quaternion_from_yaw)

HFOV = 80.0
CAM = CameraModel.from_hfov(640, 480, HFOV)
ORIGIN = (41.0, 29.0)


# ------------------------------------------------------------------ sahte ROS dunyasi
class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Future:
    def __init__(self, res):
        self._r = res

    def done(self):
        return True

    def result(self):
        return self._r


class World:
    """Basit drone: hedefe birinci derece yaklasir. Servisler etkisini aninda uygular."""

    def __init__(self, clock, pad_xy, z0=0.4):
        self.clock, self.pad = clock, pad_xy
        self.x = self.y = 0.0
        self.z0 = z0
        self.z = z0
        self.yaw = 90.0
        self.armed = False
        self.mode = "STABILIZE"
        self.target = None              # (x, y, z) GUIDED setpoint
        self.takeoff_z = None
        self.node = None
        self.timers = []                # [period, cb, next_t]
        self.next_pose = self.next_gps = self.next_state = 0.0
        self.setpoints = 0
        self.landed_xy = None

    # servis etkileri
    def apply(self, kind, req):
        if kind == "arm":
            self.armed = bool(req.value)
        elif kind == "takeoff":
            self.takeoff_z = self.z0 + req.altitude
            self.target = None
        elif kind == "mode":
            self.mode = req.custom_mode
        return Obj(success=True, mode_sent=True)

    def on_setpoint(self, m):
        p = m.pose.position
        self.target = (p.x, p.y, p.z)
        self.setpoints += 1

    def step(self, dt=0.05):
        c = self.clock
        c.t += dt
        if self.armed:
            tx, ty, tz = self.x, self.y, self.z
            if self.mode == "LAND":
                tz = self.z0 - 0.5
                vz = -0.5
                self.z = max(self.z0, self.z + vz * dt)
                if self.z <= self.z0 + 1e-6:
                    self.armed = False
                    self.landed_xy = (self.x, self.y)
            elif self.mode == "GUIDED":
                if self.target is not None:
                    tx, ty, tz = self.target
                elif self.takeoff_z is not None:
                    tz = self.takeoff_z
                for attr, tgt, vmax in (("x", tx, 2.0), ("y", ty, 2.0), ("z", tz, 1.5)):
                    cur = getattr(self, attr)
                    v = max(-vmax, min(vmax, 0.8 * (tgt - cur)))
                    setattr(self, attr, cur + v * dt)
                self.z = max(self.z0, self.z)
        n = self.node
        if n is None:
            return
        if c.t >= self.next_state:
            n.state_cb(Obj(connected=True, mode=self.mode, armed=self.armed))
            self.next_state = c.t + 0.5
        if c.t >= self.next_pose:
            qx, qy, qz, qw = quaternion_from_yaw(self.yaw)
            n.pose_cb(Obj(pose=Obj(position=Obj(x=self.x, y=self.y, z=self.z),
                                   orientation=Obj(x=qx, y=qy, z=qz, w=qw))))
            self.next_pose = c.t + 0.1
        if c.t >= self.next_gps:
            lat, lon = offset_to_latlon(ORIGIN[0], ORIGIN[1], self.x, self.y)
            n.gps_cb(Obj(status=Obj(status=0), latitude=lat, longitude=lon))
            self.next_gps = c.t + 0.5
        for t in self.timers:
            if c.t >= t[2]:
                t[2] = c.t + t[0]
                t[1]()


class FakeWatcher:
    """PadWatcher yerine: dunyadaki pad'i sentetik 'tespit' olarak dondurur (gercek geometriyle)."""

    def __init__(self, world, clock, every=0.2):
        self.w, self.c, self.every = world, clock, every
        self.last_report = None
        self._last = -1e9

    def on_image(self, msg):
        pass

    def update(self, alt):
        if self.c.t - self._last < self.every:
            return self.last_report
        self._last = self.c.t
        w = self.w
        h = w.z - w.z0
        det = []
        if h > 0.3:
            q = quaternion_from_yaw(w.yaw)
            u, v = ground_offset_to_pixel(w.pad[0] - w.x, w.pad[1] - w.y, CAM, h, q)
            r = CAM.fx * 1.0 / h
            inside = r < u < 640 - r and r < v < 480 - r
            touches = (u - r < 0 or u + r > 640 or v - r < 0 or v + r > 480)
            if (-r < u < 640 + r) and (-r < v < 480 + r) and not (not inside and not touches):
                det.append(PadDetection("kirmizi", (u, v), r, math.pi * r * r, 0.95, 0.99, touches))
        self.last_report = WatchReport(self.c.t, (640, 480), alt, tuple(det), {})
        return self.last_report


def make_stubs(world):
    class Node:
        def __init__(self, name):
            world.node = self
        def get_logger(self):
            return Obj(info=lambda s: None)
        def create_subscription(self, *a, **k):
            return None
        def create_publisher(self, typ, topic, depth):
            return Obj(publish=world.on_setpoint)
        def create_client(self, typ, name):
            kind = {"CommandBool": "arm", "CommandTOL": "takeoff", "SetMode": "mode",
                    "StreamRate": "stream"}[typ.__name__]
            return Obj(wait_for_service=lambda timeout_sec=0: True, service_is_ready=lambda: True,
                       call_async=lambda req: Future(world.apply(kind, req)))
        def create_timer(self, period, cb):
            world.timers.append([period, cb, 0.0])
        def get_clock(self):
            return Obj(now=lambda: Obj(to_msg=lambda: None))
        def destroy_node(self):
            pass

    def mk(name, **attrs):
        return type(name, (), attrs)

    def pose_stamped():
        return Obj(header=Obj(stamp=None, frame_id=""), pose=Obj(position=Obj(x=0, y=0, z=0),
                                                                 orientation=Obj(x=0, y=0, z=0, w=1)))

    PoseStamped = type("PoseStamped", (), {"__init__": lambda self: self.__dict__.update(pose_stamped().__dict__)})
    req = lambda: type("Request", (), {})
    srv = types.SimpleNamespace(CommandBool=mk("CommandBool", Request=req()), CommandTOL=mk("CommandTOL", Request=req()),
                                SetMode=mk("SetMode", Request=req()), StreamRate=mk("StreamRate", Request=req()))
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: None
    rclpy.ok = lambda: True
    rclpy.shutdown = lambda: None
    rclpy.spin_once = lambda node, timeout_sec=0: world.step()
    node_m = types.ModuleType("rclpy.node")
    node_m.Node = Node
    qos = types.ModuleType("rclpy.qos")
    for n in ("ReliabilityPolicy", "HistoryPolicy", "DurabilityPolicy"):
        setattr(qos, n, Obj(RELIABLE=1, TRANSIENT_LOCAL=2, KEEP_LAST=3, BEST_EFFORT=4, VOLATILE=5))
    qos.QoSProfile = lambda **k: Obj(**k)
    qos.qos_profile_sensor_data = Obj()
    gm = types.ModuleType("geometry_msgs"); gmm = types.ModuleType("geometry_msgs.msg"); gmm.PoseStamped = PoseStamped
    mm = types.ModuleType("mavros_msgs"); mmm = types.ModuleType("mavros_msgs.msg"); mmm.State = mk("State")
    mms = types.ModuleType("mavros_msgs.srv")
    for k, v in vars(srv).items():
        setattr(mms, k, v)
    sm = types.ModuleType("sensor_msgs"); smm = types.ModuleType("sensor_msgs.msg"); smm.NavSatFix = mk("NavSatFix")
    return {"rclpy": rclpy, "rclpy.node": node_m, "rclpy.qos": qos, "geometry_msgs": gm,
            "geometry_msgs.msg": gmm, "mavros_msgs": mm, "mavros_msgs.msg": mmm, "mavros_msgs.srv": mms,
            "sensor_msgs": sm, "sensor_msgs.msg": smm}


def load_lander():
    spec = importlib.util.spec_from_file_location("pad_lander", ROOT / "tools" / "pad_lander.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_record(d, east, north, lat_lon_err=(0.0, 0.0)):
    lat, lon = offset_to_latlon(ORIGIN[0], ORIGIN[1], east + lat_lon_err[0], north + lat_lon_err[1])
    p = Path(d) / "kayit.json"
    p.write_text(json.dumps([{"color": "kirmizi", "east_m": east, "north_m": north, "lat_deg": lat,
                              "lon_deg": lon, "n": 5, "spread_m": 0.03, "t_first": 1.0, "t_last": 2.0,
                              "recorded": True}]), encoding="utf-8")
    return str(p)


def fly(pad, rec_xy, extra=(), visible=True, max_steps=40000):
    """Dugumu bastan sona calistir. (world, cikti, args) dondur."""
    clock = FakeClock()
    world = World(clock, pad)
    d = tempfile.mkdtemp()
    rec = write_record(d, *rec_xy)
    mod = load_lander()
    args = mod.parse(["--record", rec, "--no-mavros", "--pad-east", str(pad[0]), "--pad-north", str(pad[1]),
                      "--start-alt", "5", "--log", str(Path(d) / "inis.csv"), *extra])
    buf = io.StringIO()
    stubs = make_stubs(world)
    fake_watcher = FakeWatcher(world, clock)
    if not visible:
        fake_watcher.update = lambda alt: None
    steps = {"n": 0}
    orig_step = world.step

    def counted(dt=0.05):
        steps["n"] += 1
        if steps["n"] > max_steps:
            raise RuntimeError("benzetim bitmedi (dugum DONE olmadi)")
        orig_step(dt)
    world.step = counted
    with mock.patch.dict(sys.modules, stubs), mock.patch.object(mod, "time", clock), \
            mock.patch.object(mod, "build_watcher", lambda a, log=None: fake_watcher), \
            mock.patch.object(mod, "subscribe_image", lambda *a, **k: None), \
            contextlib.redirect_stdout(buf):
        mod.run(args)
    return world, buf.getvalue(), args, Path(d) / "inis.csv"


class LanderSmokeTests(unittest.TestCase):
    def test_tam_gorev_padin_ustune_iner(self):
        pad = (3.0, 4.0)
        world, out, args, csvp = fly(pad, rec_xy=(3.4, 4.5))          # kayit ~0.6 m hatali
        self.assertIsNotNone(world.landed_xy, out[-1500:])
        err = math.hypot(world.landed_xy[0] - pad[0], world.landed_xy[1] - pad[1])
        self.assertLess(err, 0.4, out[-2500:])
        self.assertIn("DRON INDI VE DISARM OLDU", out)
        self.assertIn("INIS SONUCU", out)
        self.assertIn("GERCEK pad merkezine uzaklik", out)
        self.assertIn("Pad gorsel olarak bulundu", out)
        self.assertGreater(world.setpoints, 50)
        rows = csvp.read_text().strip().splitlines()
        self.assertGreater(len(rows), 20)                              # CSV yazildi

    def test_gorsel_yoksa_kor_inis_uyarisi(self):
        pad = (3.0, 4.0)
        world, out, args, csvp = fly(pad, rec_xy=(3.3, 4.2), visible=False)
        self.assertIsNotNone(world.landed_xy, out[-1500:])
        self.assertIn("PAD GORULMEDI", out)
        err = math.hypot(world.landed_xy[0] - 3.3, world.landed_xy[1] - 4.2)
        self.assertLess(err, 0.6)                                      # kayitli konuma indi

    def test_kayit_yoksa_havalanmaz(self):
        mod = load_lander()
        d = tempfile.mkdtemp()
        p = Path(d) / "bos.json"
        p.write_text("[]", encoding="utf-8")
        args = mod.parse(["--record", str(p), "--no-mavros"])
        world = World(FakeClock(), (0, 0))
        with mock.patch.dict(sys.modules, make_stubs(world)), mock.patch.object(mod, "time", world.clock):
            with self.assertRaises(ValueError):
                mod.run(args)
        self.assertFalse(world.armed)


class HelperTests(unittest.TestCase):
    def test_parse_ve_config(self):
        mod = load_lander()
        a = mod.parse(["--record", "k.json", "--start-alt", "6", "--commit-height", "1.0"])
        cfg = mod.controller_config(a)
        self.assertEqual((cfg.start_alt_m, cfg.commit_height_m), (6.0, 1.0))

    def test_rapor(self):
        mod = load_lander()
        s = mod.landing_report((3.1, 4.0), (3.0, 4.0), (3.0, 4.0), (3.1, 4.0), True, ["6.0s APPROACH -> DESCEND"])
        self.assertIn("0.10 m", s)
        s2 = mod.landing_report((3.1, 4.0), (3.0, 4.0), None, None, False, [])
        self.assertIn("mezur", s2)
        self.assertIn("PAD GORULMEDI", s2)


if __name__ == "__main__":
    unittest.main()
