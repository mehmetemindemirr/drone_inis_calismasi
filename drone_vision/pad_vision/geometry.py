"""Geometri katmani: piksel -> yer ofseti -> yerel x,y -> enlem-boylam.

Dedektorun buldugu pad merkezini (piksel) gercek dunya konumuna cevirir. Karar VERMEZ,
ucusa karismaz; ROS'u, MAVROS'u, saati bilmez. Girdileri disaridan alir:

  * piksel (u, v)            : PadDetection.center_px (ORIJINAL cozunurluk, piksel merkezi = tam sayi)
  * kamera modeli            : CameraModel (fx, fy, cx, cy); --hfov'dan kurulabilir
  * yukseklik (m)            : drone'un pad DUZLEMINE DIKEY yuksekligi (rel_alt / lidar)
  * durus (quaternion)       : /mavros/local_position/pose.pose.orientation (x, y, z, w)
  * drone yerel konumu       : /mavros/local_position/pose.pose.position (x=Dogu, y=Kuzey)
  * drone enlem-boylami      : /mavros/global_position/global (istege bagli)

Koordinat sozlesmeleri (ROS REP-103, yaw.py ile ayni):
  * Dunya: ENU (x=Dogu, y=Kuzey, z=Yukari). Yaw 0 = Dogu, 90 = Kuzey, saat yonu tersi pozitif.
  * Govde: FLU (x=Ileri/burun, y=Sol, z=Yukari).
  * Kamera: asagi bakar; goruntunun USTU burun yonune, SAGI drone'un sagina denk gelir
    (baska takildiysa mount_yaw_deg ver). u saga, v asagi artar.
  * Euler: roll x (ileri) etrafinda, pitch y (sol) etrafinda, yaw z etrafinda (ZYX).
    DIKKAT: bu sozlesmede pozitif pitch = burun ASAGI, pozitif roll = sag kanat ASAGI.

Neden egim (roll/pitch) duzeltmesi? Drone 10 derece yatikken yerdeki bir nokta 10 m'de ~1.8 m
kayar. Tam kuaterniyonu vererek isin gercekten yere indigi nokta bulunur.

Sinirlar (bilerek yapilmadi):
  * Lens bozulmasi (distorsiyon) yok sayilir; gercek kamerada kalibrasyonla eklenir.
  * Zemin duz kabul edilir; pad ile drone'un altindaki zemin ayni duzlemde.
  * Kamera gecikmesi/zaman eslemesi yok: konum ve durus karenin cekildigi ana yakin olmali.
  * enlem-boylam GPS dogrulugunda (1-3 m) kalir. INIS icin yerel koordinat kullan: pad ile
    drone ayni yerel sistemde oldugundan GPS hatasi iptal olur. Enlem-boylam kayit/paylasim icindir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .models import PadDetection

_WGS84_A = 6378137.0              # yari buyuk eksen (m)
_WGS84_E2 = 6.69437999014e-3      # birinci dis merkezkacliklik karesi

# Optik eksenle dikey arasindaki aci bunu asarsa isin zemine "sıyırarak" iner: kucuk
# aci hatasi buyuk konum hatasi uretir, sonuc guvenilmez sayilir.
MAX_RAY_TILT_DEG = 75.0

_LEVEL = (0.0, 0.0, 0.0, 1.0)     # duz durus (x, y, z, w), yaw = 0


class GeometryError(ValueError):
    """Gecersiz ya da guvenilmez geometri (irtifa <= 0, isin zemine inmiyor, ...)."""


# ------------------------------------------------------------------ kamera
@dataclass(frozen=True)
class CameraModel:
    """Pinhole kamera. Piksel merkezi = tam sayi sozlesmesi (dedektorla ayni)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraModel":
        """Yatay gorus acisindan kur (kare piksel varsayimi: fy = fx)."""
        if not (0.0 < hfov_deg < 180.0):
            raise ValueError("hfov_deg 0 ile 180 arasinda olmali")
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx, fx, (width - 1) / 2.0, (height - 1) / 2.0, width, height)

    def scaled(self, new_width: int) -> "CameraModel":
        """Ayni sensor, baska cozunurluk (ornegin 640 -> 320)."""
        s = new_width / self.width
        return CameraModel(self.fx * s, self.fy * s,
                           (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5,
                           new_width, int(round(self.height * s)))


def footprint_m(camera: CameraModel, height_m: float) -> Tuple[float, float]:
    """Dik bakista kadrajin yerde kapladigi alan (genislik, yukseklik) metre.
    Tarama cizgisi aralığını secerken kullan."""
    if height_m <= 0.0:
        raise GeometryError("height_m > 0 olmali")
    return camera.width * height_m / camera.fx, camera.height * height_m / camera.fy


# ------------------------------------------------------------------ durus
def quaternion_from_yaw(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Sadece yaw (ENU, derece) -> (x, y, z, w)."""
    half = math.radians(yaw_deg) / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_from_euler_deg(roll_deg: float = 0.0, pitch_deg: float = 0.0,
                              yaw_deg: float = 0.0) -> Tuple[float, float, float, float]:
    """ZYX Euler (derece) -> (x, y, z, w). Pozitif pitch = burun ASAGI (ROS sozlesmesi)."""
    cr, sr = math.cos(math.radians(roll_deg) / 2.0), math.sin(math.radians(roll_deg) / 2.0)
    cp, sp = math.cos(math.radians(pitch_deg) / 2.0), math.sin(math.radians(pitch_deg) / 2.0)
    cy, sy = math.cos(math.radians(yaw_deg) / 2.0), math.sin(math.radians(yaw_deg) / 2.0)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quaternion_to_matrix(q: Sequence[float]) -> np.ndarray:
    """(x, y, z, w) -> govde(FLU) -> dunya(ENU) donusum matrisi (3x3)."""
    x, y, z, w = (float(c) for c in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise GeometryError("sifir uzunluklu quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def vertical_height_from_range(range_m: float, q_enu: Sequence[float] = _LEVEL) -> float:
    """Yere bakan lidar MESAFESINI (isin boyu) dikey yukseklige cevir: h = range * cos(egim).
    Isinin govdenin tam altina baktigi varsayilir (PITCH_270)."""
    h = range_m * float(quaternion_to_matrix(q_enu)[2, 2])
    if h <= 0.0:
        raise GeometryError("gecersiz lidar mesafesi/durus")
    return h


def altitude_from_radius_px(radius_px: float, fx_px: float, pad_radius_m: float = 1.0) -> float:
    """Pad'in goruntudeki yaricapindan yukseklik: h = fx * R / r (models.expected_radius_px'in tersi).
    Baro/EKF irtifasini capraz dogrulamak icin; kadraj kenarindaki pad'lerde kullanma."""
    if radius_px <= 0.0:
        raise GeometryError("radius_px > 0 olmali")
    return fx_px * pad_radius_m / radius_px


# ------------------------------------------------------------------ piksel <-> yer
def _mount_rotate(f: float, l: float, mount_yaw_deg: float) -> Tuple[float, float]:
    """Kamera-hizali (ileri, sol) -> govde (ileri, sol); goruntu-ustu burundan mount_yaw kadar saat yonu tersi."""
    c, s = math.cos(math.radians(mount_yaw_deg)), math.sin(math.radians(mount_yaw_deg))
    return f * c - l * s, f * s + l * c


def pixel_to_ground_offset(u: float, v: float, camera: CameraModel, height_m: float,
                           q_enu: Sequence[float] = _LEVEL,
                           mount_yaw_deg: float = 0.0,
                           lever_body_m: Sequence[float] = (0.0, 0.0, 0.0)) -> Tuple[float, float]:
    """Pikselin gordugu yer noktasinin drone'a gore (Dogu, Kuzey) ofseti, metre.

    height_m     : drone (base_link) govdesinin pad duzlemine DIKEY yuksekligi
    q_enu        : govdenin dunyaya gore durusu (x, y, z, w); verilmezse duz + yaw 0
    mount_yaw_deg: kamera goruntu-ustu burundan kac derece saat yonu tersi donuk (0 = burun yukarida)
    lever_body_m : kameranin govde merkezine gore konumu (ileri, sol, yukari), metre
    Sonuc, drone'un yatay konumuna eklenince pad'in yerel konumu olur.
    """
    R = quaternion_to_matrix(q_enu)
    lever_w = R @ np.asarray(lever_body_m, dtype=float)
    cam_h = height_m + float(lever_w[2])
    if not cam_h > 0.0:
        raise GeometryError(f"kamera yuksekligi > 0 olmali (height_m={height_m})")

    x_img = (u - camera.cx) / camera.fx          # saga
    y_img = (v - camera.cy) / camera.fy          # asagi
    f_c, l_c, up = -y_img, -x_img, -1.0          # kamera-hizali FLU isin yonu
    f, l = _mount_rotate(f_c, l_c, mount_yaw_deg)
    d = R @ np.array([f, l, up])

    cos_tilt = -float(d[2]) / float(np.linalg.norm(d))
    if cos_tilt < math.cos(math.radians(MAX_RAY_TILT_DEG)):
        raise GeometryError(
            f"isin diklikten {math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt)))):.0f} derece "
            f"sapmis (> {MAX_RAY_TILT_DEG:.0f}): sonuc guvenilmez")

    t = cam_h / -float(d[2])
    return float(lever_w[0] + t * d[0]), float(lever_w[1] + t * d[1])


def ground_offset_to_pixel(d_east: float, d_north: float, camera: CameraModel, height_m: float,
                           q_enu: Sequence[float] = _LEVEL,
                           mount_yaw_deg: float = 0.0,
                           lever_body_m: Sequence[float] = (0.0, 0.0, 0.0)) -> Tuple[float, float]:
    """pixel_to_ground_offset'in tersi: drone'a gore (d_east, d_north) yer noktasi hangi piksele duser?

    Simulasyonda gercek pad konumunu bilirsin: bunu goruntuye izdusurup dedektorun buldugu
    pikselle karsilastirarak geometriyi dogrularsin."""
    R = quaternion_to_matrix(q_enu)
    lever_w = R @ np.asarray(lever_body_m, dtype=float)
    p_cam = np.array([d_east, d_north, -height_m]) - lever_w
    pb = R.T @ p_cam
    depth = -float(pb[2])
    if depth <= 1e-6:
        raise GeometryError("nokta kameranin arkasinda ya da hizasinda")
    c, s = math.cos(math.radians(mount_yaw_deg)), math.sin(math.radians(mount_yaw_deg))
    f_c = c * pb[0] + s * pb[1]
    l_c = -s * pb[0] + c * pb[1]
    return (camera.cx + camera.fx * (-l_c / depth),
            camera.cy + camera.fy * (-f_c / depth))


# ------------------------------------------------------------------ enlem-boylam
def _radii(lat_deg: float) -> Tuple[float, float]:
    if abs(lat_deg) > 89.9:
        raise GeometryError("kutuplara cok yakin enlem")
    s = math.sin(math.radians(lat_deg))
    w = math.sqrt(1.0 - _WGS84_E2 * s * s)
    meridional = _WGS84_A * (1.0 - _WGS84_E2) / (w ** 3)
    prime_vertical = _WGS84_A / w
    return meridional, prime_vertical


def offset_to_latlon(lat_deg: float, lon_deg: float,
                     d_east_m: float, d_north_m: float) -> Tuple[float, float]:
    """(lat, lon) noktasindan (Dogu, Kuzey) metre ofsetteki noktanin enlem-boylami (WGS84, kucuk mesafe)."""
    rn, re = _radii(lat_deg)
    lat = lat_deg + math.degrees(d_north_m / rn)
    lon = lon_deg + math.degrees(d_east_m / (re * math.cos(math.radians(lat_deg))))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def latlon_to_offset(lat0_deg: float, lon0_deg: float,
                     lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """offset_to_latlon'un tersi: (lat0, lon0)'dan (lat, lon)'a (Dogu, Kuzey) metre."""
    rn, re = _radii(lat0_deg)
    dlon = (lon_deg - lon0_deg + 180.0) % 360.0 - 180.0
    return (math.radians(dlon) * re * math.cos(math.radians(lat0_deg)),
            math.radians(lat_deg - lat0_deg) * rn)


# ------------------------------------------------------------------ PadDetection -> konum
@dataclass(frozen=True)
class PadLocation:
    color: str
    offset_east_m: float                 # drone'a gore Dogu
    offset_north_m: float                # drone'a gore Kuzey
    distance_m: float                    # yatay mesafe
    height_m: float                      # hesapta kullanilan yukseklik
    east_m: Optional[float]              # yerel ENU x (drone konumu verildiyse)
    north_m: Optional[float]             # yerel ENU y
    lat_deg: Optional[float]             # enlem (drone enlem-boylami verildiyse)
    lon_deg: Optional[float]
    confidence: float                    # dedektorun SIRALAMA puani (olasilik degil)
    touches_border: bool                 # True ise pad kesik: merkez ONYARGILI, guvenme


def locate_pad(det: PadDetection, camera: CameraModel, height_m: float,
               q_enu: Sequence[float] = _LEVEL,
               drone_local_xy: Optional[Tuple[float, float]] = None,
               drone_latlon: Optional[Tuple[float, float]] = None,
               mount_yaw_deg: float = 0.0,
               lever_body_m: Sequence[float] = (0.0, 0.0, 0.0)) -> PadLocation:
    """Tek tespiti gercek dunya konumuna cevir.

    camera ORIJINAL goruntu cozunurlugune gore kurulmus olmali (tespit merkezi orada).
    drone_local_xy: (x=Dogu, y=Kuzey) yerel konum -> east_m/north_m dolar (INIS icin bunu kullan)
    drone_latlon  : (enlem, boylam)               -> lat_deg/lon_deg dolar (kayit/paylasim icin)
    """
    de, dn = pixel_to_ground_offset(det.center_px[0], det.center_px[1], camera, height_m,
                                    q_enu, mount_yaw_deg, lever_body_m)
    east = north = lat = lon = None
    if drone_local_xy is not None:
        east, north = drone_local_xy[0] + de, drone_local_xy[1] + dn
    if drone_latlon is not None:
        lat, lon = offset_to_latlon(drone_latlon[0], drone_latlon[1], de, dn)
    return PadLocation(det.color, de, dn, math.hypot(de, dn), height_m, east, north, lat, lon,
                       det.confidence, det.touches_border)
