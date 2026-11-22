#!/usr/bin/env python3
"""TEK DRONE: kayitli pad'e HASSAS INIS gorevi (yaw_v3.py'nin durum makinesi + gorsel inis denetleyicisi).

Akis:  WAIT_CONNECT -> STREAMS -> SYNC -> GUIDED -> ARM -> TAKEOFF -> CLIMB -> MISSION -> LAND -> DONE
  MISSION = pad_vision.landing.LandingController: kayitli konuma git, pad'i kendi kamerasiyla gor,
            ortala, irtifaya gore konik hata sinirinda alcal, commit yuksekliginde LAND.

Hazirlik: once pad_recorder ile kayit.json uret (pad'in ustunden gec), drone'u indir. Sonra:

  # simulasyon (gercek pad konumunu verirsen sonunda inis hatasi yazilir)
  python3 tools/pad_lander.py --record kayit.json --start-alt 5 --pad-east 0 --pad-north 5 --log inis.csv

  # sahada (A3, Pi Cam V2): gercek konum yok, hatayi mezurayla olcersin
  python3 tools/pad_lander.py --record kayit.json --fcu-url /dev/ttyS0:115200 --no-mavros ... \\
      --topic /camera/image_raw --hfov 62.2 --start-alt 5 --commit-height 1.0 --sync-max 240

Bu betik MAVROS'u kendisi baslatir (yaw_v3.py gibi); zaten calisiyorsa --no-mavros.
Yukseklik = ARM noktasina gore rel_alt; pad'in zeminle ayni seviyede oldugu varsayilir.
Ctrl+C: LAND moduna gecer.
"""
import argparse
import atexit
import csv
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from live_locate import (PoseSnapshot, SpeedEstimator, build_watcher,  # noqa: E402
                         quaternion_tilt_deg)
from pad_recorder import make_observation  # noqa: E402
from adapters.ros_image import subscribe_image  # noqa: E402
from pad_vision import (LandingConfig, LandingController, TrackerConfig,  # noqa: E402
                        VisionTracker, choose_target, load_record)

# yaw_v3.py ile ayni sabitler
ALT_TOLERANCE = 0.3
CLIMB_TIMEOUT = 40.0
STREAM_RATE_HZ = 10
ALT_DATA_TIMEOUT = 15.0
SYNC_MIN_SECONDS = 15.0
SYNC_STABLE_SECONDS = 8.0
STABLE_GAP = 1.0
STALE_WARN_SECONDS = 2.0
STALE_LAND_SECONDS = 10.0
GROUND_ALT = 0.3
GROUND_WAIT_SECONDS = 5.0

CSV_FIELDS = ["t", "phase", "drone_x", "drone_y", "height", "est_e", "est_n", "target_x", "target_y",
              "target_h", "pad_seen", "note"]


# ------------------------------------------------------------------ saf yardimcilar (ROS'suz, test edilir)
def controller_config(args) -> LandingConfig:
    return LandingConfig(start_alt_m=args.start_alt, arrive_tol_m=args.arrive_tol,
                         descent_rate_mps=args.descent_rate, lat_k=args.lat_k, lat_min_m=args.lat_min,
                         commit_height_m=args.commit_height, blind_below_m=args.blind_below,
                         mission_timeout_s=args.mission_timeout)


def landing_report(final_xy: Tuple[float, float], target_xy: Tuple[float, float],
                   truth: Optional[Tuple[float, float]], commit_xy: Optional[Tuple[float, float]],
                   pad_seen: bool, log) -> str:
    lines = ["", "=" * 70, "INIS SONUCU"]
    lines.append(f"Son konum (yerel): ({final_xy[0]:+.2f}, {final_xy[1]:+.2f})")
    d_t = math.hypot(final_xy[0] - target_xy[0], final_xy[1] - target_xy[1])
    lines.append(f"Kayitli hedefe uzaklik: {d_t:.2f} m")
    if commit_xy is not None:
        lines.append(f"Commit anindaki konum: ({commit_xy[0]:+.2f}, {commit_xy[1]:+.2f})")
    if truth is not None:
        err = math.hypot(final_xy[0] - truth[0], final_xy[1] - truth[1])
        lines.append(f"GERCEK pad merkezine uzaklik: {err:.2f} m")
    else:
        lines.append("Gercek pad konumu verilmedi: mezurayla pad merkezi - drone arasi mesafeyi olc.")
    lines.append("Pad gorsel olarak bulundu" if pad_seen else
                 "UYARI: PAD GORULMEDI, kor inis yapildi (sonuc yalnizca kayit dogruluguna bagli)")
    lines.append("Faz gecmisi:")
    lines.extend("  " + l for l in log)
    return "\n".join(lines)


def format_status(phase: str, pos, height: float, est, cmd) -> str:
    e = f"({est[0]:+.2f}, {est[1]:+.2f})" if est is not None else "yok"
    n = f" | {cmd.note}" if cmd is not None and cmd.note else ""
    return (f"[INIS] {phase:8s} konum=({pos[0]:+.2f}, {pos[1]:+.2f}) h={height:.2f} m  gorsel pad={e}{n}")


def yaw_to_quaternion(yaw_deg: float):
    r = math.radians(yaw_deg)
    return 0.0, 0.0, math.sin(r / 2.0), math.cos(r / 2.0)


def quaternion_to_yaw(x, y, z, w) -> float:
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# ------------------------------------------------------------------ MAVROS
mavros_proc = None


def cleanup():
    global mavros_proc
    if mavros_proc and mavros_proc.poll() is None:
        print("\n[CLEANUP] MAVROS kapatiliyor...")
        try:
            os.killpg(os.getpgid(mavros_proc.pid), signal.SIGTERM)
            mavros_proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(mavros_proc.pid), signal.SIGKILL)
            except Exception:
                pass


def start_mavros(fcu_url: str):
    global mavros_proc
    print("[BASLAT] MAVROS baslatiliyor...")
    mavros_proc = subprocess.Popen(["ros2", "launch", "mavros", "apm.launch", f"fcu_url:={fcu_url}"],
                                   preexec_fn=os.setsid)
    atexit.register(cleanup)
    for i in range(5, 0, -1):
        print(f"\r[BASLAT] {i} saniye bekleniyor...", end="", flush=True)
        time.sleep(1)
    print("\n[BASLAT] Servisler bekleniyor...\n")


# ------------------------------------------------------------------ ROS tarafi
def run(args):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, StreamRate
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                           qos_profile_sensor_data)
    from sensor_msgs.msg import NavSatFix

    # kaydi ARM'dan ONCE oku: yoksa havalanmadan hata ver
    rec = load_record(args.record, args.color)
    print(f"[KAYIT] {args.record}: n={rec['n']} yerel=({rec.get('east_m')}, {rec.get('north_m')}) "
          f"enlem={rec.get('lat_deg')} boylam={rec.get('lon_deg')}")
    truth = (args.pad_east, args.pad_north) if args.pad_east is not None and args.pad_north is not None else None
    lcfg = controller_config(args)

    watcher = build_watcher(args, log=(lambda s: print(s, flush=True)) if args.verbose else (lambda s: None))
    tracker = VisionTracker(TrackerConfig(max_tilt_deg=args.max_tilt, max_speed_mps=args.max_speed))
    speed_est = SpeedEstimator()

    csv_file = csv_writer = None
    if args.log:
        csv_file = open(args.log, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    if not args.no_mavros:
        start_mavros(args.fcu_url)

    class Lander(Node):
        def __init__(self):
            super().__init__("pad_lander")
            state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                   history=HistoryPolicy.KEEP_LAST, depth=10)
            self.state = None
            self.pose: Optional[PoseSnapshot] = None
            self.pose_t = 0.0                 # monotonic
            self.gap_t = 0.0
            self.alt_ref = 0.0
            self.yaw = 0.0
            self.hold_yaw = 0.0
            self.gps = None
            self.gps_t = 0.0
            self.t0 = time.monotonic()

            self.ctrl: Optional[LandingController] = None
            self.target_xy = (0.0, 0.0)
            self.cmd = None
            self.est = None
            self.commit_xy = None
            self.pad_seen = False
            self.last_air_xy = (0.0, 0.0)
            self.navigating = False
            self.sp = (0.0, 0.0, 0.0)

            self.phase = "WAIT_CONNECT"
            self.finished = False
            self.streams_start = self.sync_start = 0.0
            self.pending = None
            self.pending_tag = None
            self.last_req = 0.0
            self.last_warn = 0.0
            self.climb_start = 0.0
            self.ground_since = 0.0
            self.last_vis_skip = {}

            self.create_subscription(State, "/mavros/state", self.state_cb, state_qos)
            self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb,
                                     qos_profile_sensor_data)
            self.create_subscription(NavSatFix, "/mavros/global_position/global", self.gps_cb,
                                     qos_profile_sensor_data)
            subscribe_image(self, args.topic, watcher.on_image)
            self.setpoint_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
            self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
            self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
            self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
            self.stream_client = self.create_client(StreamRate, "/mavros/set_stream_rate")
            for name, cli in (("arming", self.arm_client), ("takeoff", self.takeoff_client),
                              ("set_mode", self.mode_client)):
                while not cli.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f"{name} servisi bekleniyor...")
            self.create_timer(0.2, self.tick)
            self.create_timer(1.0, self.print_status)
            self.create_timer(0.1, self.publish_setpoint)

        # ---------- callback'ler
        def state_cb(self, m):
            self.state = m

        def pose_cb(self, m):
            now = time.monotonic()
            if self.pose_t > 0.0 and now - self.pose_t > STABLE_GAP:
                self.gap_t = now
            self.pose_t = now
            p, o = m.pose.position, m.pose.orientation
            self.pose = PoseSnapshot(p.x, p.y, p.z, (o.x, o.y, o.z, o.w))
            self.yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
            speed_est.add(now, p.x, p.y, p.z)
            if self.phase == "LAND":
                self.last_air_xy = (p.x, p.y)

        def gps_cb(self, m):
            if m.status.status >= 0 and not (math.isnan(m.latitude) or math.isnan(m.longitude)):
                self.gps = (m.latitude, m.longitude)
                self.gps_t = time.monotonic()

        # ---------- yardimcilar
        def age(self):
            return math.inf if self.pose_t <= 0.0 else time.monotonic() - self.pose_t

        def rel_alt(self):
            return (self.pose.z - self.alt_ref) if self.pose else 0.0

        def pos(self):
            return (self.pose.x, self.pose.y) if self.pose else (0.0, 0.0)

        def can_send(self, interval):
            return self.pending is None and (time.monotonic() - self.last_req) > interval

        def send(self, client, req, tag):
            self.pending = client.call_async(req)
            self.pending_tag = tag
            self.last_req = time.monotonic()

        def set_mode(self, mode):
            req = SetMode.Request()
            req.custom_mode = mode
            self.send(self.mode_client, req, f"mode:{mode}")

        def set_phase(self, p):
            print(f"\n[FAZ] {self.phase} -> {p}")
            self.phase = p

        def start_land(self, why=""):
            self.navigating = False
            self.ground_since = 0.0
            if why:
                print(f"[INIS] LAND: {why}")
            self.last_air_xy = self.pos()
            self.set_phase("LAND")

        def handle_result(self, tag, res):
            if res is None:
                print(f"[HATA] {tag}: cevap alinamadi")
            elif tag == "takeoff":
                if res.success:
                    print(f"[SONUC] Kalkis komutu kabul edildi ({args.start_alt} m)")
                    self.climb_start = time.monotonic()
                    self.set_phase("CLIMB")
                else:
                    print("[SONUC] Kalkis reddedildi, tekrar denenecek")
            elif tag == "stream":
                print("[SONUC] Veri akisi istegi iletildi")
            elif tag == "arm":
                print("[SONUC] Arm kabul edildi" if res.success
                      else "[SONUC] Arm reddedildi (pre-arm: GPS/EKF hazir mi?), tekrar denenecek")
            elif tag == "disarm":
                print("[SONUC] Disarm kabul edildi" if res.success else "[SONUC] Disarm reddedildi")

        # ---------- setpoint
        def publish_setpoint(self):
            if not self.navigating:
                return
            m = PoseStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.pose.position.x, m.pose.position.y, m.pose.position.z = self.sp
            qx, qy, qz, qw = yaw_to_quaternion(self.hold_yaw)
            m.pose.orientation.x, m.pose.orientation.y = qx, qy
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            self.setpoint_pub.publish(m)

        # ---------- gorsel tahmin
        def vision_step(self):
            pose = self.pose
            now = time.monotonic()
            rep = self._last_rep_check()
            if rep is None or pose is None:
                return
            t = now - self.t0
            obs, why = make_observation(rep, pose, t=t, pose_age_s=now - self.pose_t, speed=speed_est.speed(),
                                        hfov_deg=args.hfov, color=args.color, mount_yaw_deg=args.mount_yaw,
                                        ground_z=self.alt_ref, min_height_m=0.3)
            if obs is None:
                return
            res = tracker.add(obs, t)
            if res != "ok":
                k = now - self.last_vis_skip.get(res, -1e9)
                if k > 3.0:
                    self.last_vis_skip[res] = now
                    print(f"[GORSEL] gozlem elendi: {res}")

        def _last_rep_check(self):
            before = watcher.last_report
            rep = watcher.update(self.rel_alt() if self.pose else None)
            return rep if (rep is not None and rep is not before) else None

        # ---------- durum makinesi
        def tick(self):
            if self.pending is not None:
                if not self.pending.done():
                    return
                try:
                    res = self.pending.result()
                except Exception as e:
                    print(f"[HATA] {self.pending_tag}: {e}")
                    res = None
                tag = self.pending_tag
                self.pending = None
                self.handle_result(tag, res)

            if self.state is None or self.finished:
                return
            s = self.state
            age = self.age()

            if self.phase in ("CLIMB", "MISSION"):
                if s.mode != "GUIDED":
                    print(f"[UYARI] Mod {s.mode} oldu (GUIDED degil). Kontrol pilota birakildi.")
                    self.navigating = False
                    self.set_phase("DONE")
                    self.finished = True
                    return
                if age > STALE_LAND_SECONDS:
                    self.start_land(f"konum verisi {age:.0f} sn kesildi")
                    return
                if age > STALE_WARN_SECONDS:
                    if time.monotonic() - self.last_warn > 2.0:
                        print(f"[UYARI] Veri {age:.1f} sn gelmedi, bekleniyor")
                        self.last_warn = time.monotonic()
                    return

            if self.phase == "WAIT_CONNECT":
                if s.connected:
                    self.streams_start = time.monotonic()
                    self.set_phase("STREAMS")

            elif self.phase == "STREAMS":
                if age < STALE_WARN_SECONDS:
                    print(f"[BILGI] Konum verisi geliyor ({self.pose.z:.2f} m)")
                    self.sync_start = time.monotonic()
                    self.gap_t = time.monotonic()
                    self.set_phase("SYNC")
                elif time.monotonic() - self.streams_start > ALT_DATA_TIMEOUT:
                    print("[HATA] Konum verisi gelmiyor! ARM EDILMEDI, cikiliyor.")
                    self.finished = True
                elif self.can_send(3.0):
                    if self.stream_client.service_is_ready():
                        req = StreamRate.Request()
                        req.stream_id = 0
                        req.message_rate = STREAM_RATE_HZ
                        req.on_off = True
                        print(f"[KOMUT] Veri akisi isteniyor ({STREAM_RATE_HZ} Hz)...")
                        self.send(self.stream_client, req, "stream")
                    else:
                        self.last_req = time.monotonic()

            elif self.phase == "SYNC":
                now = time.monotonic()
                elapsed = now - self.sync_start
                stable = age < STABLE_GAP and (now - self.gap_t) >= SYNC_STABLE_SECONDS
                if elapsed >= SYNC_MIN_SECONDS and stable:
                    print(f"[BILGI] Veri akisi kararli ({elapsed:.0f} sn), ARM asamasina geciliyor")
                    self.set_phase("GUIDED")
                elif elapsed > args.sync_max:
                    print("[HATA] Veri akisi kararli hale gelmedi! ARM EDILMEDI, cikiliyor.")
                    self.finished = True

            elif self.phase == "GUIDED":
                if s.mode == "GUIDED":
                    self.set_phase("ARM")
                elif self.can_send(2.0):
                    print("[KOMUT] GUIDED moduna geciliyor...")
                    self.set_mode("GUIDED")

            elif self.phase == "ARM":
                if s.armed:
                    self.alt_ref = self.pose.z
                    self.hold_yaw = self.yaw
                    print(f"[BILGI] Irtifa sifir noktasi: z_ref={self.alt_ref:.2f}")
                    self.set_phase("TAKEOFF")
                elif self.can_send(3.0):
                    print("[KOMUT] Arm istegi gonderiliyor...")
                    req = CommandBool.Request()
                    req.value = True
                    self.send(self.arm_client, req, "arm")

            elif self.phase == "TAKEOFF":
                if not s.armed:
                    print("[UYARI] Dron disarm oldu, arm asamasina donuluyor")
                    self.set_phase("ARM")
                elif self.can_send(1.0):
                    print(f"[KOMUT] Kalkis: {args.start_alt} m")
                    req = CommandTOL.Request()
                    req.altitude = float(args.start_alt)
                    self.send(self.takeoff_client, req, "takeoff")

            elif self.phase == "CLIMB":
                self.vision_step()
                if self.rel_alt() >= args.start_alt - ALT_TOLERANCE:
                    self.begin_mission()
                elif time.monotonic() - self.climb_start > CLIMB_TIMEOUT:
                    self.start_land("irtifaya cikma zaman asimi")

            elif self.phase == "MISSION":
                self.mission_step()

            elif self.phase == "LAND":
                if not s.armed:
                    print("[SONUC] DRON INDI VE DISARM OLDU!")
                    self.report()
                    self.set_phase("DONE")
                    self.finished = True
                elif s.mode != "LAND":
                    if self.can_send(2.0):
                        print("[KOMUT] LAND moduna geciliyor...")
                        self.set_mode("LAND")
                else:
                    on_ground = age < STALE_WARN_SECONDS and self.rel_alt() < GROUND_ALT
                    if not on_ground:
                        self.ground_since = 0.0
                    elif self.ground_since == 0.0:
                        self.ground_since = time.monotonic()
                    elif time.monotonic() - self.ground_since > GROUND_WAIT_SECONDS and self.can_send(3.0):
                        print("[KOMUT] Yerde ama hala armed, disarm gonderiliyor...")
                        req = CommandBool.Request()
                        req.value = False
                        self.send(self.arm_client, req, "disarm")

        # ---------- gorev
        def begin_mission(self):
            now = time.monotonic()
            gps = self.gps if (self.gps is not None and now - self.gps_t < 2.0) else None
            try:
                ch = choose_target(rec, self.pos(), gps, args.target_mode)
            except ValueError as e:
                self.start_land(f"hedef belirlenemedi: {e}")
                return
            self.target_xy = (ch.x, ch.y)
            print(f"[HEDEF] kaynak={ch.source} yerel=({ch.x:+.2f}, {ch.y:+.2f})"
                  + (f" (yerel/enlem-boylam farki {ch.diff_m:.2f} m)" if ch.diff_m is not None else ""))
            if ch.note:
                print(f"[HEDEF] UYARI: {ch.note}")
            self.ctrl = LandingController(lcfg, self.target_xy, self.pos(), self.rel_alt(), now - self.t0)
            self.sp = (self.pos()[0], self.pos()[1], self.alt_ref + self.rel_alt())
            self.navigating = True
            self.set_phase("MISSION")

        def mission_step(self):
            self.vision_step()
            now = time.monotonic() - self.t0
            height = self.rel_alt()
            self.est = tracker.estimate(now)
            cmd = self.ctrl.update(now, self.pos(), height, self.est)
            self.cmd = cmd
            self.pad_seen = cmd.pad_seen
            self.sp = (cmd.x, cmd.y, self.alt_ref + cmd.height_m)
            for line in self.ctrl.log[self._logged:]:
                print(f"[GECIS] {line}")
            self._logged = len(self.ctrl.log)
            if csv_writer:
                e = self.est if self.est is not None else ("", "")
                csv_writer.writerow({"t": f"{now:.2f}", "phase": cmd.phase, "drone_x": f"{self.pos()[0]:.3f}",
                                     "drone_y": f"{self.pos()[1]:.3f}", "height": f"{height:.3f}",
                                     "est_e": e[0] if e[0] == "" else f"{e[0]:.3f}",
                                     "est_n": e[1] if e[1] == "" else f"{e[1]:.3f}",
                                     "target_x": f"{cmd.x:.3f}", "target_y": f"{cmd.y:.3f}",
                                     "target_h": f"{cmd.height_m:.3f}", "pad_seen": int(cmd.pad_seen),
                                     "note": cmd.note})
                csv_file.flush()
            if cmd.land:
                self.commit_xy = self.pos()
                print(f"[INIS] Commit: konum=({self.commit_xy[0]:+.2f}, {self.commit_xy[1]:+.2f}) "
                      f"h={height:.2f} m -> LAND (hedef xy kilitli)")
                self.start_land()

        _logged = 0

        def report(self):
            print(landing_report(self.last_air_xy, self.target_xy, truth, self.commit_xy, self.pad_seen,
                                 self.ctrl.log if self.ctrl else []), flush=True)

        def emergency_land(self):
            self.navigating = False
            if self.state and self.state.armed:
                print("\n[ACIL] Ctrl+C: LAND moduna geciliyor...")
                req = SetMode.Request()
                req.custom_mode = "LAND"
                fut = self.mode_client.call_async(req)
                end = time.monotonic() + 3.0
                while not fut.done() and time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.1)

        def print_status(self):
            if self.state is None:
                print("Baglanti: BEKLENIYOR...")
                return
            if self.phase == "MISSION":
                print(format_status(self.ctrl.phase, self.pos(), self.rel_alt(), self.est, self.cmd), flush=True)
            else:
                print(f"[DURUM] faz={self.phase} mod={self.state.mode} armed={self.state.armed} "
                      f"h={self.rel_alt():.2f} m konum=({self.pos()[0]:+.2f}, {self.pos()[1]:+.2f})", flush=True)

    rclpy.init()
    node = Lander()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.emergency_land()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if csv_file:
            csv_file.close()


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--record", required=True, help="pad_recorder'in urettigi kayit.json")
    ap.add_argument("--color", choices=("kirmizi", "mavi"), default="kirmizi")
    ap.add_argument("--fcu-url", default="udp://127.0.0.1:14550@14555", help="gercek drone: /dev/ttyS0:115200")
    ap.add_argument("--no-mavros", action="store_true", help="MAVROS zaten calisiyor, baslatma")
    ap.add_argument("--sync-max", type=float, default=90.0, help="SYNC azami sure (sn); gercek seri hatta 180-240")
    ap.add_argument("--target-mode", choices=("auto", "local", "latlon"), default="auto")
    ap.add_argument("--pad-east", type=float, default=None, help="(istege bagli) GERCEK pad Dogu, hata raporu icin")
    ap.add_argument("--pad-north", type=float, default=None)
    # inis parametreleri
    ap.add_argument("--start-alt", type=float, default=5.0, help="yaklasma irtifasi (m); A3 testi icin ~5")
    ap.add_argument("--commit-height", type=float, default=1.5, help="bu yukseklikte son merkeze kilitlen + LAND")
    ap.add_argument("--blind-below", type=float, default=2.5, help="bunun altinda pad kaybolursa yukselme")
    ap.add_argument("--descent-rate", type=float, default=0.4)
    ap.add_argument("--lat-k", type=float, default=0.15, help="izinli yanal hata = max(lat-min, lat-k * yukseklik)")
    ap.add_argument("--lat-min", type=float, default=0.25)
    ap.add_argument("--arrive-tol", type=float, default=0.6)
    ap.add_argument("--mission-timeout", type=float, default=150.0)
    ap.add_argument("--max-tilt", type=float, default=6.0, help="gorsel gozlem kapisi: en fazla egim (derece)")
    ap.add_argument("--max-speed", type=float, default=1.0, help="gorsel gozlem kapisi: en fazla hiz (m/s)")
    # kamera / dedektor
    ap.add_argument("--topic", default="/drone/down_camera/image_raw")
    ap.add_argument("--hfov", type=float, default=80.0, help="simulasyon 80, Pi Cam V2 62.2")
    ap.add_argument("--mount-yaw", type=float, default=0.0)
    ap.add_argument("--work-width", type=int, default=320)
    ap.add_argument("--every", type=float, default=0.2)
    ap.add_argument("--use-hint", action="store_true")
    ap.add_argument("--swap-rb", action="store_true")
    ap.add_argument("--log", default=None, help="her adimi CSV'ye yaz")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
