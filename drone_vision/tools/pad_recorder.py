#!/usr/bin/env python3
"""CANLI pad KAYDEDICI (ayri surec, ucusa dokunmaz, komut gondermez).

Kamera + MAVROS konumu ile pad'i bulur, her gozlemi PadMap'e verir (kalite kapilari + medyan),
kaydedilen konumu yerel x,y VE enlem-boylam olarak yazar. Istersen gercek konumla karsilastirir.

  # simulasyon: drone (0,0)'da, pad 5 m Kuzeyde; yaw_v3.py baska terminalde ucuyor
  python3 tools/pad_recorder.py --pad-east 0 --pad-north 5 --out kayit.json

  # gercek konumu bilmiyorsan (sahada) --pad-east/--pad-north verme: yalniz kaydeder
  python3 tools/pad_recorder.py --hfov 62.2 --topic /camera/image_raw --out kayit.json

  # MAVI pad icin ya da ikisini birden kaydetmek icin
  python3 tools/pad_recorder.py --color mavi --pad-east 0 --pad-north 5 --out kayit_mavi.json
  python3 tools/pad_recorder.py --color hepsi --out kayit.json      # ikisi de ayni dosyaya

Cikti: [GOZ] satiri = her kapidan gecen gozlem, [RED] = elenen gozlem (nedeniyle, 3 sn'de bir),
[KAYIT] = PadMap'in guncel kaydi. Ctrl+C'de OZET basar ve --out JSON dosyasina yazar.
--publish: kayitlari std_msgs/String (JSON) olarak /pad_vision/pad_records konusuna da yayinlar
(inis modulu bunu dinleyecek).
Yukseklik = pose.z - --ground-z (sim: zemin z=0). Gercek dronda ARM'a gore rel_alt/lidar kullanilacak.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from adapters.pad_watcher import PadWatcher, WatchReport  # noqa: E402
from adapters.ros_image import subscribe_image  # noqa: E402
from live_locate import (PoseSnapshot, SpeedEstimator, build_watcher,  # noqa: E402
                         quaternion_tilt_deg, quaternion_yaw_deg)
from pad_vision import (CameraModel, GeometryError, Observation, PadMap,  # noqa: E402
                        PadMapConfig, PadRecord, locate_pad)


# ------------------------------------------------------------------ saf mantik (ROS'suz, test edilir)
def make_observation(rep: WatchReport, pose: PoseSnapshot, *, t: float, pose_age_s: float,
                     speed: Optional[float], hfov_deg: float, color: str = "kirmizi",
                     mount_yaw_deg: float = 0.0, ground_z: float = 0.0,
                     latlon: Optional[Tuple[float, float]] = None,
                     min_height_m: float = 1.0) -> Tuple[Optional[Observation], str]:
    """Bir tespit raporundan PadMap gozlemi uret. (Observation, "") ya da (None, atlama_nedeni).
    Kenara degen tespit BURADA atilmaz: PadMap kapisi 'kenarda' diye reddeder (nedeni gorunur olsun)."""
    height = pose.z - ground_z
    if height < min_height_m:
        return None, f"yukseklik {height:.2f} m < {min_height_m:.1f} m"
    dets = [d for d in rep.detections if d.color == color]
    if not dets:
        return None, f"{color} pad gorunmuyor"
    best = dets[0]
    w, h = rep.frame_size
    cam = CameraModel.from_hfov(w, h, hfov_deg)
    try:
        loc = locate_pad(best, cam, height, pose.q, drone_local_xy=(pose.x, pose.y),
                         drone_latlon=latlon, mount_yaw_deg=mount_yaw_deg)
    except GeometryError as e:
        return None, f"geometri: {e}"
    return Observation.from_location(loc, t, tilt_deg=quaternion_tilt_deg(pose.q),
                                     speed_mps=speed, pose_age_s=pose_age_s), ""


def colors_for(choice: str) -> List[str]:
    """--color degerinden taranacak renk listesi ('hepsi' = ikisi)."""
    return ["kirmizi", "mavi"] if choice == "hepsi" else [choice]


def format_obs(t: float, o: Observation) -> str:
    ll = f" enlem={o.lat_deg:.7f} boylam={o.lon_deg:.7f}" if o.has_latlon else ""
    sp = f"{o.speed_mps:.2f}" if o.speed_mps is not None else "?"
    return (f"[GOZ] t={t:6.1f}s {o.color.upper()} yerel=({o.east_m:+.2f}, {o.north_m:+.2f}){ll} "
            f"h={o.height_m:.1f} m hiz={sp} egim={o.tilt_deg:.1f}")


def format_record(r: PadRecord, truth: Optional[Tuple[float, float]] = None) -> str:
    err = ""
    if truth is not None and r.east_m is not None:
        e = math.hypot(r.east_m - truth[0], r.north_m - truth[1])
        err = f" HATA={e:.2f} m"
    ll = f" enlem={r.lat_deg:.7f} boylam={r.lon_deg:.7f}" if r.lat_deg is not None else ""
    loc = f"yerel=({r.east_m:+.2f}, {r.north_m:+.2f})" if r.east_m is not None else "yerel=?"
    state = "KAYDEDILDI" if r.recorded else "toplaniyor"
    return f"[KAYIT] {r.color.upper()} {state} n={r.n} {loc}{ll} dagilim={r.spread_m:.2f} m{err}"


def summary(pm: PadMap, truth: Optional[Tuple[float, float]], rejects: dict,
            truth_color: Optional[str] = None) -> str:
    lines = ["", "=" * 70, "OZET"]
    recs = pm.records()
    if not recs:
        lines.append("Hic pad kaydedilmedi (pad gorunmedi ya da tum gozlemler elendi).")
    for r in recs:
        lines.append(format_record(r, truth if truth_color in (None, r.color) else None))
    if rejects:
        lines.append("Elenen gozlemler: " + ", ".join(f"{k}={v}" for k, v in sorted(rejects.items())))
        lines.append("  (cok 'hiz'/'egim' varsa pad'in ustunden yavas gec ya da PadMapConfig esiklerini gevset;"
                     " cok 'kenarda' varsa irtifayi artir)")
    return "\n".join(lines)


def save_json(pm: PadMap, path: str) -> None:
    Path(path).write_text(pm.to_json() + "\n", encoding="utf-8")


# ------------------------------------------------------------------ ROS tarafi
def run(args):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import NavSatFix
    from std_msgs.msg import String

    watcher = build_watcher(args, log=(lambda s: print(s, flush=True)) if args.verbose else (lambda s: None))
    pm = PadMap(PadMapConfig(max_tilt_deg=args.max_tilt, max_speed_mps=args.max_speed,
                             min_observations=args.min_obs))
    speed_est = SpeedEstimator()
    truth = (args.pad_east, args.pad_north) if args.pad_east is not None and args.pad_north is not None else None
    rejects: dict = {}
    colors = colors_for(args.color)
    truth_color = args.truth_color or (args.color if args.color != "hepsi" else "kirmizi")

    class Recorder(Node):
        def __init__(self):
            super().__init__("pad_recorder")
            self.pose: Optional[PoseSnapshot] = None
            self.pose_t = 0.0
            self.gps: Optional[Tuple[float, float]] = None
            self.gps_t = 0.0
            self.t0 = time.monotonic()
            self.last_msg = {}
            self.pub = (self.create_publisher(String, "/pad_vision/pad_records", 10) if args.publish else None)
            subscribe_image(self, args.topic, watcher.on_image)
            self.create_subscription(PoseStamped, args.pose_topic, self.pose_cb, qos_profile_sensor_data)
            self.create_subscription(NavSatFix, args.gps_topic, self.gps_cb, qos_profile_sensor_data)
            self.create_timer(0.1, self.tick)
            self.get_logger().info(
                f"{args.topic} + {args.pose_topic} + {args.gps_topic} dinleniyor; hfov={args.hfov} "
                f"gercek={'yok' if truth is None else truth}")

        def pose_cb(self, m):
            p, o = m.pose.position, m.pose.orientation
            self.pose = PoseSnapshot(p.x, p.y, p.z, (o.x, o.y, o.z, o.w))
            self.pose_t = time.monotonic()
            speed_est.add(self.pose_t, p.x, p.y, p.z)

        def gps_cb(self, m):
            if m.status.status >= 0 and not (math.isnan(m.latitude) or math.isnan(m.longitude)):
                self.gps = (m.latitude, m.longitude)
                self.gps_t = time.monotonic()

        def say(self, key, text, every=3.0):
            now = time.monotonic()
            if now - self.last_msg.get(key, -1e9) >= every:
                self.last_msg[key] = now
                print(text, flush=True)

        def tick(self):
            before = watcher.last_report
            pose = self.pose
            alt = (pose.z - args.ground_z) if pose is not None else None
            rep = watcher.update(alt)
            if rep is None or rep is before:
                return
            now = time.monotonic()
            if pose is None:
                self.say("pose", "[RED] konum verisi yok (/mavros/local_position/pose)")
                return
            latlon = self.gps if (self.gps is not None and now - self.gps_t < 2.0) else None
            for color in colors:
                obs, reason = make_observation(rep, pose, t=now - self.t0, pose_age_s=now - self.pose_t,
                                               speed=speed_est.speed(), hfov_deg=args.hfov, color=color,
                                               mount_yaw_deg=args.mount_yaw, ground_z=args.ground_z,
                                               latlon=latlon, min_height_m=args.min_height)
                if obs is None:
                    self.say(reason, f"[SKIP] {reason}")
                    continue
                res = pm.add(obs)
                if not res.accepted:
                    rejects[res.reason] = rejects.get(res.reason, 0) + 1
                    self.say(f"red_{color}_{res.reason}", f"[RED] {color} gozlem elendi: {res.reason}")
                    continue
                print(format_obs(now - self.t0, obs), flush=True)
                print(format_record(res.record, truth if color == truth_color else None), flush=True)
                if self.pub is not None:
                    self.pub.publish(String(data=pm.to_json()))
                if args.out:
                    save_json(pm, args.out)

    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if args.out:
            save_json(pm, args.out)
            print(f"Kayit yazildi: {args.out}", flush=True)
        print(summary(pm, truth, rejects, truth_color), flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pad-east", type=float, default=None, help="(istege bagli) GERCEK pad Dogu (m), karsilastirma icin")
    ap.add_argument("--pad-north", type=float, default=None, help="(istege bagli) GERCEK pad Kuzey (m)")
    ap.add_argument("--topic", default="/drone/down_camera/image_raw")
    ap.add_argument("--pose-topic", default="/mavros/local_position/pose")
    ap.add_argument("--gps-topic", default="/mavros/global_position/global")
    ap.add_argument("--hfov", type=float, default=80.0, help="yatay gorus acisi; simulasyon 80, Pi Cam V2 62.2")
    ap.add_argument("--mount-yaw", type=float, default=0.0)
    ap.add_argument("--ground-z", type=float, default=0.0)
    ap.add_argument("--color", choices=("kirmizi", "mavi", "hepsi"), default="kirmizi",
                    help="kaydedilecek pad rengi; 'hepsi' = kirmizi ve mavi birlikte")
    ap.add_argument("--truth-color", choices=("kirmizi", "mavi"), default=None,
                    help="--pad-east/--pad-north hangi rengin gercek konumu (varsayilan: --color, hepsi ise kirmizi)")
    ap.add_argument("--min-height", type=float, default=1.0)
    ap.add_argument("--max-tilt", type=float, default=4.0, help="kapi: en fazla egim (derece)")
    ap.add_argument("--max-speed", type=float, default=1.5, help="kapi: en fazla hiz (m/s)")
    ap.add_argument("--min-obs", type=int, default=3, help="'kaydedildi' icin en az gozlem")
    ap.add_argument("--work-width", type=int, default=320)
    ap.add_argument("--every", type=float, default=0.2, help="islem periyodu (sn); kayit icin sik ornek lazim")
    ap.add_argument("--use-hint", action="store_true")
    ap.add_argument("--swap-rb", action="store_true")
    ap.add_argument("--out", default=None, help="kaydi bu JSON dosyasina yaz")
    ap.add_argument("--publish", action="store_true", help="/pad_vision/pad_records (JSON) yayinla")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
