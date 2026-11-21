"""PadMap: pad gozlemlerini toplar, kalite kapilarindan gecirir, medyanla tek konuma indirir.

ROS'suz, saf Python. Karar vermez; "su renkte pad su konumda, su kadar gozleme dayaniyor" der.

Akis:  PadLocation (geometry.locate_pad) + o anki egim/hiz/poz yasi  ->  Observation  ->  PadMap.add()
Kapilar (varsayilan): kadraj kenarina degen, egimli (>4 derece), hizli (>1.5 m/s), eski konumlu (>0.5 s),
yuksekligi uygunsuz, dusuk guvenli gozlemler ELENIR (konum hatasi hizla/egimle buyuyor; olcum:
3.7 m/s ve 7.6 derecede 1.14 m hata, yavasta 0.05-0.3 m).
Kume: ayni renkte birbirine cluster_radius_m icindeki gozlemler tek pad sayilir; konum = eksen basina MEDYAN
(gurultuyu ve aykiri degeri eler; kamera/yukseklik SAPMASINI (bias) gidermez).
Konum iki cercevede tutulur: yerel ENU (kendi drone'un EKF'si, inis icin) ve enlem-boylam (paylasim icin).
Ikisi KARISTIRILMAZ: bir gozlem yalnizca ayni cerceveyi tasiyan kumelerle karsilastirilir.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .geometry import latlon_to_offset


@dataclass(frozen=True)
class PadMapConfig:
    max_tilt_deg: float = 4.0         # roll/pitch bilesik egimi
    max_speed_mps: float = 1.5        # yatay hiz
    max_pose_age_s: float = 0.5       # konum verisinin yasi
    min_height_m: float = 1.5         # pad'e dikey yukseklik (alt)
    max_height_m: float = 60.0        # (ust)
    min_confidence: float = 0.5       # dedektor SIRALAMA puani
    cluster_radius_m: float = 2.0     # ayni pad sayilma yaricapi
    min_observations: int = 3         # "kaydedildi" icin en az gozlem
    max_per_cluster: int = 300        # bellek siniri (en eskiler duser)


@dataclass(frozen=True)
class Observation:
    color: str
    t: float                              # gozlem zamani (s), saat tutarli olmali
    east_m: Optional[float] = None        # yerel ENU (pad konumu)
    north_m: Optional[float] = None
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    height_m: Optional[float] = None
    confidence: float = 1.0
    touches_border: bool = False
    tilt_deg: Optional[float] = None      # None = bilinmiyor, kapidan gecer
    speed_mps: Optional[float] = None
    pose_age_s: Optional[float] = None

    @property
    def has_local(self) -> bool:
        return self.east_m is not None and self.north_m is not None

    @property
    def has_latlon(self) -> bool:
        return self.lat_deg is not None and self.lon_deg is not None

    @classmethod
    def from_location(cls, loc, t: float, tilt_deg: Optional[float] = None,
                      speed_mps: Optional[float] = None,
                      pose_age_s: Optional[float] = None) -> "Observation":
        """geometry.PadLocation'dan gozlem uret."""
        return cls(loc.color, t, loc.east_m, loc.north_m, loc.lat_deg, loc.lon_deg, loc.height_m,
                   loc.confidence, loc.touches_border, tilt_deg, speed_mps, pose_age_s)


@dataclass(frozen=True)
class PadRecord:
    color: str
    east_m: Optional[float]
    north_m: Optional[float]
    lat_deg: Optional[float]
    lon_deg: Optional[float]
    n: int
    spread_m: float                       # gozlemlerin medyan merkezden medyan uzakligi
    t_first: float
    t_last: float
    recorded: bool                        # n >= min_observations

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class AddResult:
    accepted: bool
    reason: str                           # kabul: "ok"; ret: kapi adi
    record: Optional[PadRecord] = None    # kabulde, gozlemin girdigi kumenin guncel kaydi


@dataclass
class _Cluster:
    obs: Deque[Observation]
    _center: Optional[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]] = field(
        default=None, repr=False)

    def centers(self):
        """((east, north) | None, (lat, lon) | None) -- eksen basina medyan."""
        if self._center is None:
            loc = [o for o in self.obs if o.has_local]
            ll = [o for o in self.obs if o.has_latlon]
            c_loc = ((statistics.median(o.east_m for o in loc),
                      statistics.median(o.north_m for o in loc)) if loc else None)
            c_ll = ((statistics.median(o.lat_deg for o in ll),
                     statistics.median(o.lon_deg for o in ll)) if ll else None)
            self._center = (c_loc, c_ll)
        return self._center


def _dist_to_cluster(o: Observation, cl: _Cluster) -> float:
    """Ayni cercevede mesafe; ortak cerceve yoksa sonsuz (karistirma)."""
    c_loc, c_ll = cl.centers()
    if o.has_latlon and c_ll is not None:
        de, dn = latlon_to_offset(c_ll[0], c_ll[1], o.lat_deg, o.lon_deg)
        return math.hypot(de, dn)
    if o.has_local and c_loc is not None:
        return math.hypot(o.east_m - c_loc[0], o.north_m - c_loc[1])
    return math.inf


def _spread(cl: _Cluster) -> float:
    c_loc, c_ll = cl.centers()
    ds: List[float] = []
    if c_ll is not None:
        for o in cl.obs:
            if o.has_latlon:
                ds.append(math.hypot(*latlon_to_offset(c_ll[0], c_ll[1], o.lat_deg, o.lon_deg)))
    elif c_loc is not None:
        for o in cl.obs:
            if o.has_local:
                ds.append(math.hypot(o.east_m - c_loc[0], o.north_m - c_loc[1]))
    return statistics.median(ds) if ds else 0.0


class PadMap:
    def __init__(self, config: Optional[PadMapConfig] = None):
        self.cfg = config or PadMapConfig()
        self._clusters: Dict[str, List[_Cluster]] = {}

    # ---- kapilar -------------------------------------------------------------------------------
    def _gate(self, o: Observation) -> Optional[str]:
        c = self.cfg
        if not (o.has_local or o.has_latlon):
            return "konum_yok"
        if o.touches_border:
            return "kenarda"
        if o.confidence < c.min_confidence:
            return "guven"
        if o.height_m is not None and not (c.min_height_m <= o.height_m <= c.max_height_m):
            return "yukseklik"
        if o.tilt_deg is not None and o.tilt_deg > c.max_tilt_deg:
            return "egim"
        if o.speed_mps is not None and o.speed_mps > c.max_speed_mps:
            return "hiz"
        if o.pose_age_s is not None and o.pose_age_s > c.max_pose_age_s:
            return "poz_eski"
        return None

    # ---- ekleme --------------------------------------------------------------------------------
    def add(self, o: Observation) -> AddResult:
        reason = self._gate(o)
        if reason is not None:
            return AddResult(False, reason)
        clusters = self._clusters.setdefault(o.color, [])
        best, best_d = None, math.inf
        for cl in clusters:
            d = _dist_to_cluster(o, cl)
            if d < best_d:
                best, best_d = cl, d
        if best is None or best_d > self.cfg.cluster_radius_m:
            best = _Cluster(deque(maxlen=self.cfg.max_per_cluster))
            clusters.append(best)
        best.obs.append(o)
        best._center = None
        return AddResult(True, "ok", self._record(o.color, best))

    # ---- okuma ---------------------------------------------------------------------------------
    def _record(self, color: str, cl: _Cluster) -> PadRecord:
        c_loc, c_ll = cl.centers()
        ts = [o.t for o in cl.obs]
        n = len(cl.obs)
        return PadRecord(color,
                         c_loc[0] if c_loc else None, c_loc[1] if c_loc else None,
                         c_ll[0] if c_ll else None, c_ll[1] if c_ll else None,
                         n, _spread(cl), min(ts), max(ts), n >= self.cfg.min_observations)

    def records(self, color: Optional[str] = None) -> List[PadRecord]:
        out = [self._record(c, cl) for c, cls in self._clusters.items()
               if color is None or c == color for cl in cls]
        return sorted(out, key=lambda r: (-r.n, r.spread_m))

    def best(self, color: str, recorded_only: bool = True) -> Optional[PadRecord]:
        """Renkteki en cok gozleme dayanan (esitlikte en az dagilimli) pad. Yoksa None."""
        for r in self.records(color):
            if r.recorded or not recorded_only:
                return r
        return None

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.records()], ensure_ascii=False)

    def clear(self, color: Optional[str] = None) -> None:
        if color is None:
            self._clusters.clear()
        else:
            self._clusters.pop(color, None)
