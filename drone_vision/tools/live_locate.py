#!/usr/bin/env python3
"""CANLI konum tahmini + GERCEK pad konumuyla karsilastirma (ayri surec, ucusa dokunmaz).

Kamera + MAVROS konumu/durusu ile pad'in yerel (Dogu, Kuzey) konumunu tahmin eder, senin verdigin
gercek konumla karsilastirir, hatayi metre olarak yazar. Hicbir karar vermez, komut gondermez.
Ctrl+C'de ozet ve teshis basar: kamera yonu (mount yaw) hatasi mi, yukseklik/olcek hatasi mi?

  # simulasyon: drone (0,0)'da dogdu, pad 5 m Kuzeyde (MAVROS yerel ENU, drone'un baslangicina gore)
  python3 tools/live_locate.py --pad-east 0 --pad-north 5

  # drone ucarken (ornegin yaw_v3.py baska terminalde) ayni komut, hareketli ornekler ayri sayilir
  python3 tools/live_locate.py --pad-east 0 --pad-north 5 --csv sonuc.csv

Pad konumu: MAVROS yerel ENU (x=Dogu, y=Kuzey), EKF baslangicina (genelde kalkis noktasi) gore.
README'ye gore Gazebo +x = Kuzey = MAVROS +y; pad'i Gazebo'da x ekseninde verdiysen --pad-north ile ver,
Dogu isaretini canli ciktiya bakarak dogrula.

Yukseklik = pose.z - --ground-z (simulasyonda zemin z=0). Gercek dronda lidar/rel_alt kullanilacak.
Pose, karenin islendigi andaki en son konumdur (kare zamani ile eslenmez): hizli giderken hata biraz buyur.
"""
import argparse
import csv
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Deque, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.pad_watcher import PadWatcher, WatchReport  # noqa: E402
from adapters.ros_image import subscribe_image  # noqa: E402
from pad_vision import (CameraModel, DetectorConfig, GeometryError, HsvPadDetector,  # noqa: E402
                        altitude_from_radius_px, locate_pad)


# ------------------------------------------------------------------ saf mantik (ROS'suz, test edilir)
@dataclass(frozen=True)
class PoseSnapshot:
    x: float                                  # yerel Dogu (m)
    y: float                                  # yerel Kuzey (m)
    z: float                                  # yerel Yukari (m)
    q: Tuple[float, float, float, float]      # (x, y, z, w), govde -> ENU


@dataclass(frozen=True)
class Fix:
    color: str
    est_east: float            # tahmin edilen pad, yerel ENU
    est_north: float
    true_east: float           # senin verdigin gercek pad konumu
    true_north: float
    err_east: float            # tahmin - gercek
    err_north: float
    err_m: float
    est_off: Tuple[float, float]     # drone'a gore tahmin (Dogu, Kuzey)
    true_off: Tuple[float, float]    # drone'a gore gercek
    height_m: float                  # hesapta kullanilan yukseklik
    height_from_pad_m: Optional[float]   # pad yaricapindan bulunan yukseklik (capraz kontrol)
    radius_px: float
    confidence: float


def quaternion_yaw_deg(q: Sequence[float]) -> float:
    x, y, z, w = q
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quaternion_tilt_deg(q: Sequence[float]) -> float:
    """Govde 'asagi' ekseninin dikeyden sapmasi (derece)."""
    x, y, z, w = q
    c = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def evaluate_report(rep: WatchReport, pose: PoseSnapshot, *,
                    pad_truth: Tuple[float, float], hfov_deg: float,
                    color: str = "kirmizi", mount_yaw_deg: float = 0.0,
                    ground_z: float = 0.0, min_height_m: float = 1.0,
                    allow_border: bool = False) -> Tuple[Optional[Fix], str]:
    """Bir tespit raporunu gercek konumla karsilastir. (Fix, "") ya da (None, atlama_nedeni)."""
    height = pose.z - ground_z
    if height < min_height_m:
        return None, f"yukseklik {height:.2f} m < {min_height_m:.1f} m"
    dets = [d for d in rep.detections if d.color == color]
    if not dets:
        return None, f"{color} pad gorunmuyor"
    best = dets[0]                                  # guvene gore sirali
    if best.touches_border and not allow_border:
        return None, "pad kadraj kenarinda (merkez onyargili), atlandi"

    w, h = rep.frame_size
    cam = CameraModel.from_hfov(w, h, hfov_deg)
    try:
        loc = locate_pad(best, cam, height, pose.q, drone_local_xy=(pose.x, pose.y),
                         mount_yaw_deg=mount_yaw_deg)
    except GeometryError as e:
        return None, f"geometri: {e}"

    try:
        h_pad: Optional[float] = altitude_from_radius_px(best.radius_px, cam.fx)
    except GeometryError:
        h_pad = None
    te, tn = pad_truth
    ee, en = loc.east_m - te, loc.north_m - tn
    return Fix(color, loc.east_m, loc.north_m, te, tn, ee, en, math.hypot(ee, en),
               (loc.offset_east_m, loc.offset_north_m), (te - pose.x, tn - pose.y),
               height, h_pad, best.radius_px, best.confidence), ""


class SpeedEstimator:
    """Son ~1 sn'lik konum degisiminden yatay+dikey hiz (m/s). Yetersiz veri -> None."""

    def __init__(self, window_s: float = 1.0, min_span_s: float = 0.2):
        self._win, self._min = window_s, min_span_s
        self._pts: Deque[Tuple[float, float, float, float]] = deque()

    def add(self, t: float, x: float, y: float, z: float) -> None:
        self._pts.append((t, x, y, z))
        while len(self._pts) > 2 and t - self._pts[0][0] > self._win:
            self._pts.popleft()

    def speed(self) -> Optional[float]:
        if len(self._pts) < 2:
            return None
        t0, x0, y0, z0 = self._pts[0]
        t1, x1, y1, z1 = self._pts[-1]
        dt = t1 - t0
        if dt < self._min:
            return None
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) / dt


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class ErrorStats:
    """Orneklerden hata istatistigi ve teshis (kamera yonu, olcek)."""

    def __init__(self, stationary_speed: float = 0.3, min_offset_m: float = 1.0):
        self._stat_speed = stationary_speed
        self._min_off = min_offset_m
        self.rows: List[Tuple[Fix, Optional[float]]] = []

    def add(self, fix: Fix, speed: Optional[float]) -> None:
        self.rows.append((fix, speed))

    def _is_stationary(self, speed: Optional[float]) -> bool:
        return speed is not None and speed < self._stat_speed

    def _usable(self) -> List[Fix]:
        out = []
        for f, _ in self.rows:
            if math.hypot(*f.true_off) >= self._min_off and math.hypot(*f.est_off) >= self._min_off:
                out.append(f)
        return out

    def suggest_mount_yaw(self, current_deg: float = 0.0) -> Optional[Tuple[float, float]]:
        """(onerilen --mount-yaw, tutarlilik 0..1). Tahmin ofseti ile gercek ofset arasindaki ortalama
        donme acisindan bulunur; ornek yoksa None. Tutarlilik dusukse (< 0.8) donme sabit degildir."""
        fx = self._usable()
        if not fx:
            return None
        sins = coss = 0.0
        for f in fx:
            d = math.atan2(f.true_off[1], f.true_off[0]) - math.atan2(f.est_off[1], f.est_off[0])
            sins += math.sin(d)
            coss += math.cos(d)
        n = len(fx)
        mean = math.degrees(math.atan2(sins / n, coss / n))
        return _wrap_deg(current_deg + mean), math.hypot(sins / n, coss / n)

    def scale_ratio(self) -> Optional[float]:
        """Tahmin edilen ofset boyu / gercek ofset boyu (ortalama). 1.0 = uyumlu."""
        fx = self._usable()
        if not fx:
            return None
        return sum(math.hypot(*f.est_off) / math.hypot(*f.true_off) for f in fx) / len(fx)

    @staticmethod
    def _describe(fixes: List[Fix]) -> str:
        errs = [f.err_m for f in fixes]
        be = sum(f.err_east for f in fixes) / len(fixes)
        bn = sum(f.err_north for f in fixes) / len(fixes)
        return (f"ortalama hata {sum(errs) / len(errs):.2f} m, medyan {median(errs):.2f} m, "
                f"en buyuk {max(errs):.2f} m | sapma (bias): Dogu {be:+.2f}, Kuzey {bn:+.2f} m")

    def summary(self, current_mount_yaw: float = 0.0) -> str:
        lines = ["", "=" * 70, "OZET"]
        if not self.rows:
            lines.append("Hic gecerli ornek yok (pad gorunmedi, irtifa dusuk ya da konum gelmedi).")
            return "\n".join(lines)
        still = [f for f, s in self.rows if self._is_stationary(s)]
        lines.append(f"Ornek: {len(self.rows)} (durgun: {len(still)}, hareketli/bilinmeyen: {len(self.rows) - len(still)})")
        if still:
            lines.append("[durgun]  " + self._describe(still))
        lines.append("[hepsi]   " + self._describe([f for f, _ in self.rows]))

        lines.append("Teshis:")
        sug = self.suggest_mount_yaw(current_mount_yaw)
        ratio = self.scale_ratio()
        if sug is None:
            lines.append(f"  (kamera yonu/olcek icin pad'in drone'dan en az {self._min_off:.0f} m uzakta "
                         f"gorundugu ornek gerek)")
        else:
            new_yaw, consistency = sug
            diff = abs(_wrap_deg(new_yaw - current_mount_yaw))
            if consistency < 0.8:
                lines.append(f"  - Kamera yonu: donme tutarsiz (tutarlilik {consistency:.2f}); yon hatasi degil, "
                             f"gurultu/yukseklik/yanlis pad konumu olabilir")
            elif diff < 5.0:
                lines.append(f"  - Kamera yonu: uyumlu (sapma {diff:.1f} derece, --mount-yaw {current_mount_yaw:g})")
            else:
                lines.append(f"  - Kamera yonu: tahmin gercekten ~{diff:.0f} derece donuk -> "
                             f"--mount-yaw {new_yaw:.0f} ile dene (once pad koordinatini ve eksenleri de dogrula)")
        if ratio is not None:
            if abs(ratio - 1.0) < 0.1:
                lines.append(f"  - Olcek: uyumlu (tahmin/gercek mesafe = {ratio:.2f})")
            else:
                lines.append(f"  - Olcek: tahmin edilen mesafe gercegin %{ratio * 100:.0f}'i -> yukseklik kaynagi "
                             f"(--ground-z) ya da odak uzakligi (--hfov) hatali olabilir")
        return "\n".join(lines)


def format_fix(t: float, fix: Fix, speed: Optional[float], pose: PoseSnapshot) -> str:
    sp = f"{speed:.2f} m/s" if speed is not None else "?"
    hp = f"{fix.height_from_pad_m:.1f}" if fix.height_from_pad_m is not None else "?"
    return (f"[LOK] t={t:6.1f}s {fix.color.upper()} tahmin=({fix.est_east:+.2f}, {fix.est_north:+.2f}) "
            f"gercek=({fix.true_east:+.2f}, {fix.true_north:+.2f}) "
            f"hata=({fix.err_east:+.2f}, {fix.err_north:+.2f}) |{fix.err_m:.2f}| m  "
            f"h={fix.height_m:.1f} m (pad'dan {hp})  hiz={sp}  "
            f"yaw={quaternion_yaw_deg(pose.q):+.0f} egim={quaternion_tilt_deg(pose.q):.1f}  "
            f"r={fix.radius_px:.0f}px guven={fix.confidence:.2f}")


CSV_FIELDS = ["t", "color", "est_east", "est_north", "true_east", "true_north", "err_east", "err_north",
              "err_m", "height_m", "height_from_pad_m", "radius_px", "confidence", "speed",
              "yaw_deg", "tilt_deg", "drone_x", "drone_y", "drone_z"]


def csv_row(t: float, fix: Fix, speed: Optional[float], pose: PoseSnapshot) -> dict:
    return {"t": f"{t:.2f}", "color": fix.color, "est_east": f"{fix.est_east:.3f}",
            "est_north": f"{fix.est_north:.3f}", "true_east": f"{fix.true_east:.3f}",
            "true_north": f"{fix.true_north:.3f}", "err_east": f"{fix.err_east:.3f}",
            "err_north": f"{fix.err_north:.3f}", "err_m": f"{fix.err_m:.3f}",
            "height_m": f"{fix.height_m:.3f}",
            "height_from_pad_m": "" if fix.height_from_pad_m is None else f"{fix.height_from_pad_m:.3f}",
            "radius_px": f"{fix.radius_px:.1f}", "confidence": f"{fix.confidence:.3f}",
            "speed": "" if speed is None else f"{speed:.3f}",
            "yaw_deg": f"{quaternion_yaw_deg(pose.q):.1f}", "tilt_deg": f"{quaternion_tilt_deg(pose.q):.2f}",
            "drone_x": f"{pose.x:.3f}", "drone_y": f"{pose.y:.3f}", "drone_z": f"{pose.z:.3f}"}


def build_watcher(args, log=lambda s: None) -> PadWatcher:
    """ROS'suz: argumanlardan PadWatcher kurar (test edilebilir)."""
    cfg = DetectorConfig(work_width=args.work_width)
    return PadWatcher(HsvPadDetector(config=cfg), None, hfov_deg=args.hfov, use_hint=args.use_hint,
                      every_s=args.every, swap_rb=args.swap_rb, log=log)


# ------------------------------------------------------------------ ROS tarafi
def run(args):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    watcher = build_watcher(args, log=(lambda s: print(s, flush=True)) if args.verbose else (lambda s: None))
    stats = ErrorStats(args.stationary_speed)
    speed_est = SpeedEstimator()
    truth = (args.pad_east, args.pad_north)

    csv_file = csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    class Locate(Node):
        def __init__(self):
            super().__init__("pad_locate")
            self.pose: Optional[PoseSnapshot] = None
            self.pose_t = 0.0
            self.t0 = time.monotonic()
            self.last_skip = {}
            subscribe_image(self, args.topic, watcher.on_image)
            self.create_subscription(PoseStamped, args.pose_topic, self.pose_cb, qos_profile_sensor_data)
            self.create_timer(0.1, self.tick)
            self.get_logger().info(
                f"{args.topic} + {args.pose_topic} dinleniyor; gercek pad=({truth[0]:+.2f} Dogu, "
                f"{truth[1]:+.2f} Kuzey), hfov={args.hfov}, mount-yaw={args.mount_yaw}, ground-z={args.ground_z}")

        def pose_cb(self, m):
            p, o = m.pose.position, m.pose.orientation
            self.pose = PoseSnapshot(p.x, p.y, p.z, (o.x, o.y, o.z, o.w))
            self.pose_t = time.monotonic()
            speed_est.add(self.pose_t, p.x, p.y, p.z)

        def skip(self, reason):
            now = time.monotonic()
            if now - self.last_skip.get(reason, -1e9) >= 3.0:
                self.last_skip[reason] = now
                print(f"[LOK] atlandi: {reason}", flush=True)

        def tick(self):
            before = watcher.last_report
            pose = self.pose
            alt = (pose.z - args.ground_z) if pose is not None else None
            rep = watcher.update(alt)
            if rep is None or rep is before:
                return
            if pose is None or time.monotonic() - self.pose_t > 1.0:
                self.skip("konum verisi yok/eski (/mavros/local_position/pose)")
                return
            fix, reason = evaluate_report(rep, pose, pad_truth=truth, hfov_deg=args.hfov, color=args.color,
                                          mount_yaw_deg=args.mount_yaw, ground_z=args.ground_z,
                                          min_height_m=args.min_height, allow_border=args.allow_border)
            if fix is None:
                self.skip(reason)
                return
            spd = speed_est.speed()
            t = time.monotonic() - self.t0
            stats.add(fix, spd)
            print(format_fix(t, fix, spd, pose), flush=True)
            if csv_writer:
                csv_writer.writerow(csv_row(t, fix, spd, pose))
                csv_file.flush()

    rclpy.init()
    node = Locate()
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
        if csv_file:
            csv_file.close()
        print(stats.summary(args.mount_yaw), flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pad-east", type=float, required=True,
                    help="GERCEK pad konumu, yerel Dogu (m), EKF baslangicina gore")
    ap.add_argument("--pad-north", type=float, required=True,
                    help="GERCEK pad konumu, yerel Kuzey (m), EKF baslangicina gore")
    ap.add_argument("--topic", default="/drone/down_camera/image_raw")
    ap.add_argument("--pose-topic", default="/mavros/local_position/pose")
    ap.add_argument("--hfov", type=float, default=80.0, help="yatay gorus acisi (derece); simulasyon 80, Pi Cam V2 62.2")
    ap.add_argument("--mount-yaw", type=float, default=0.0,
                    help="kamera goruntu-ustu burundan kac derece saat yonu tersi donuk (0 = ust burun yonu)")
    ap.add_argument("--ground-z", type=float, default=0.0, help="pad/zemin duzleminin yerel z'si (yukseklik = z - bu)")
    ap.add_argument("--color", choices=("kirmizi", "mavi"), default="kirmizi")
    ap.add_argument("--min-height", type=float, default=1.0, help="bunun altinda tahmin yapma (m)")
    ap.add_argument("--allow-border", action="store_true", help="kadraj kenarindaki (kesik) pad'i de kullan")
    ap.add_argument("--work-width", type=int, default=320, help="tespit icin kucultme genisligi")
    ap.add_argument("--every", type=float, default=0.5, help="islem periyodu (sn)")
    ap.add_argument("--use-hint", action="store_true", help="yukseklikten beklenen yaricap ipucu")
    ap.add_argument("--swap-rb", action="store_true", help="kirmizi/mavi ters gorunuyorsa kanal sirasini cevir")
    ap.add_argument("--stationary-speed", type=float, default=0.3, help="bunun altindaki hiz 'durgun' sayilir (m/s)")
    ap.add_argument("--csv", default=None, help="her ornegi bu CSV dosyasina yaz")
    ap.add_argument("--verbose", action="store_true", help="PadWatcher'in gorundu/kayboldu satirlarini da yaz")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
