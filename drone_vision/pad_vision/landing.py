"""Hassas inis mantigi: ROS'suz, saf Python. Komut URETIR, gondermez.

Parcalar:
  load_record / choose_target : kayit.json'dan hedef konum (yerel x,y ya da enlem-boylam)
  VisionTracker               : kameradan gelen pad gozlemlerinin kisa pencere medyani (kalite kapili)
  LandingController           : durum makinesi; her adimda (hedef x, hedef y, hedef yukseklik) verir

Akis:  APPROACH -> DESCEND -> LAND      (pad gorunmezse: SEARCH -> BLIND -> LAND)
  APPROACH : kayitli konuma git (pad gorunur gorunmez gorsel tahmine gec)
  DESCEND  : pad'e ortalanmis kal, yukseklige gore izinli yanal hata KONISI icinde alcal;
             hata konisi disindaysa alcalmayi DURDUR, once ortala
  SEARCH   : pad gorulmuyor -> yavasca yuksel (kadraj buyusun), bulursan DESCEND
  BLIND    : arama basarisiz -> son bilinen konuma kor inis (sonuc "pad gorulmedi" diye isaretlenir)
  LAND     : commit yuksekliginde son merkeze kilitlen, LAND modu (pad artik kadraja sigmaz)
Yukseklik = pad duzlemine dikey mesafe (ARM'a gore rel_alt, pad'in zeminle ayni seviyede oldugu varsayilir).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .geometry import latlon_to_offset
from .padmap import Observation


# ------------------------------------------------------------------ kayit -> hedef
def load_record(path: str, color: str = "kirmizi") -> Dict:
    """kayit.json'daki `recorded` olan, en cok gozlemli kaydi dondur. Yoksa ValueError."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cands = [r for r in data if r.get("color") == color and r.get("recorded")]
    if not cands:
        raise ValueError(f"{path}: '{color}' icin KAYDEDILMIS pad yok")
    return max(cands, key=lambda r: (r.get("n", 0), -r.get("spread_m", 0.0)))


@dataclass(frozen=True)
class TargetChoice:
    x: float
    y: float
    source: str                      # "local" | "latlon"
    diff_m: Optional[float]          # iki yontemin hedefi arasindaki fark (ikisi de varsa)
    note: str


def choose_target(rec: Dict, cur_local: Tuple[float, float],
                  cur_latlon: Optional[Tuple[float, float]], mode: str = "auto",
                  agree_m: float = 2.0) -> TargetChoice:
    """Kayitli pad'in SIMDIKI yerel koordinattaki konumu.
    local : kaydin yerel x,y'si (ayni EKF oturumu icin; re-arm orijini degistirmediyse dogru)
    latlon: kaydin enlem-boylami -> simdiki GPS'e gore ofset (EKF orijininden bagimsiz, GPS hatasi var)
    auto  : ikisi de varsa ve farklari agree_m'den kucukse local, degilse latlon (not dusulur)."""
    loc = None
    if rec.get("east_m") is not None and rec.get("north_m") is not None:
        loc = (rec["east_m"], rec["north_m"])
    ll = None
    if (rec.get("lat_deg") is not None and rec.get("lon_deg") is not None and cur_latlon is not None):
        de, dn = latlon_to_offset(cur_latlon[0], cur_latlon[1], rec["lat_deg"], rec["lon_deg"])
        ll = (cur_local[0] + de, cur_local[1] + dn)
    diff = math.hypot(loc[0] - ll[0], loc[1] - ll[1]) if (loc and ll) else None

    if mode == "local":
        if loc is None:
            raise ValueError("kayitta yerel konum yok")
        return TargetChoice(loc[0], loc[1], "local", diff, "")
    if mode == "latlon":
        if ll is None:
            raise ValueError("enlem-boylam hedefi icin kayitta lat/lon ve anlik GPS gerekli")
        return TargetChoice(ll[0], ll[1], "latlon", diff, "")
    if loc is None and ll is None:
        raise ValueError("kayitta kullanilabilir konum yok")
    if loc is None:
        return TargetChoice(ll[0], ll[1], "latlon", None, "yerel konum yok, enlem-boylam kullanildi")
    if ll is None:
        return TargetChoice(loc[0], loc[1], "local", None, "GPS/enlem-boylam yok, yerel konum kullanildi")
    if diff <= agree_m:
        return TargetChoice(loc[0], loc[1], "local", diff, "")
    return TargetChoice(ll[0], ll[1], "latlon", diff,
                        f"yerel ve enlem-boylam hedefi {diff:.1f} m farkli (EKF orijini degismis olabilir), "
                        f"enlem-boylam kullanildi")


# ------------------------------------------------------------------ gorsel tahmin
@dataclass(frozen=True)
class TrackerConfig:
    window_s: float = 2.0
    min_samples: int = 2
    max_tilt_deg: float = 6.0
    max_speed_mps: float = 1.0
    max_pose_age_s: float = 0.5
    min_confidence: float = 0.5
    min_height_m: float = 0.3
    max_height_m: float = 60.0
    max_jump_m: float = 3.0           # mevcut tahminden bu kadar uzaktaki gozlem aykiri sayilir
    reset_after_rejects: int = 6      # ust uste bu kadar aykiri -> pencereyi sifirla (pad gercekten oradadir)
    allow_border: bool = False


class VisionTracker:
    """Son window_s saniyedeki gecerli gozlemlerin ekseni basina medyani."""

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self._s: Deque[Tuple[float, float, float]] = deque()
        self._jump_rejects = 0

    def _gate(self, o: Observation) -> Optional[str]:
        c = self.cfg
        if not o.has_local:
            return "konum_yok"
        if o.touches_border and not c.allow_border:
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

    def _prune(self, now: float) -> None:
        while self._s and now - self._s[0][0] > self.cfg.window_s:
            self._s.popleft()

    def add(self, o: Observation, now: float) -> str:
        """"ok" ya da elenme nedeni."""
        why = self._gate(o)
        if why is not None:
            return why
        self._prune(now)
        est = self.estimate(now)
        if est is not None and len(self._s) >= 3 and math.hypot(o.east_m - est[0], o.north_m - est[1]) > self.cfg.max_jump_m:
            self._jump_rejects += 1
            if self._jump_rejects >= self.cfg.reset_after_rejects:
                self._s.clear()
                self._jump_rejects = 0
                self._s.append((o.t, o.east_m, o.north_m))
                return "ok"
            return "ziplama"
        self._jump_rejects = 0
        self._s.append((o.t, o.east_m, o.north_m))
        return "ok"

    def estimate(self, now: float) -> Optional[Tuple[float, float]]:
        self._prune(now)
        if len(self._s) < self.cfg.min_samples:
            return None
        es = sorted(s[1] for s in self._s)
        ns = sorted(s[2] for s in self._s)
        return (_median(es), _median(ns))

    def clear(self) -> None:
        self._s.clear()
        self._jump_rejects = 0


def _median(v: List[float]) -> float:
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


# ------------------------------------------------------------------ denetleyici
@dataclass(frozen=True)
class LandingConfig:
    start_alt_m: float = 5.0          # yaklasma irtifasi
    arrive_tol_m: float = 0.6         # hedef noktaya "vardi" sayilan yatay mesafe
    acquire_s: float = 6.0            # vardiktan sonra pad'i gormek icin bekleme
    descent_rate_mps: float = 0.4     # alcalma hizi
    descent_lead_m: float = 1.0       # hedef yukseklik gercek yukseklikten en fazla bu kadar asagida
    lat_k: float = 0.15               # izinli yanal hata = max(lat_min, lat_k * yukseklik)  (konik)
    lat_min_m: float = 0.25
    commit_height_m: float = 1.5      # bu yukseklikte son merkeze kilitlen, LAND ver
    blind_below_m: float = 2.5        # bunun altinda pad kaybolursa YUKSELME, son merkezle inmeye devam et
    max_xy_speed_mps: float = 1.0     # hedef noktanin kayma hizi (ani sicrama olmasin)
    lost_s: float = 3.0               # DESCEND'de bu kadar gorsel tahmin yoksa SEARCH
    search_step_m: float = 1.0
    search_hold_s: float = 3.0
    search_max_extra_m: float = 3.0   # start_alt + bu kadar yukariya cik
    search_timeout_s: float = 12.0    # SEARCH bu kadar surerse BLIND
    mission_timeout_s: float = 150.0


@dataclass(frozen=True)
class Command:
    phase: str
    x: float                          # hedef yerel Dogu
    y: float                          # hedef yerel Kuzey
    height_m: float                   # hedef yukseklik (pad duzlemine gore)
    land: bool = False                # True: LAND moduna gec
    note: str = ""
    pad_seen: bool = True             # inis gorsel tahminle mi yapildi (False: kor)


class LandingController:
    def __init__(self, cfg: LandingConfig, goal_xy: Tuple[float, float],
                 drone_xy: Tuple[float, float], height_m: float, now: float):
        self.cfg = cfg
        self.phase = "APPROACH"
        self.goal = goal_xy                       # kayitli konum
        self.last_known = goal_xy                 # son bilinen en iyi pad konumu
        self.tx, self.ty = drone_xy               # kayan hedef noktasi
        self.h_target = height_m
        self._t = now
        self._t0 = now
        self._last_est_t = -math.inf
        self._wait_since: Optional[float] = None
        self._search_t0 = 0.0
        self._search_step_t = 0.0
        self._seen_pad = False                    # gorsel tahmin hic kullanildi mi
        self.log: List[str] = []

    # -- yardimcilar
    def _slew(self, desired: Tuple[float, float], dt: float) -> None:
        dx, dy = desired[0] - self.tx, desired[1] - self.ty
        d = math.hypot(dx, dy)
        step = self.cfg.max_xy_speed_mps * dt
        if d <= step or d == 0.0:
            self.tx, self.ty = desired
        else:
            self.tx += dx / d * step
            self.ty += dy / d * step

    def _go(self, phase: str, now: float, note: str = "") -> None:
        self.log.append(f"{now - self._t0:6.1f}s {self.phase} -> {phase} {note}".rstrip())
        self.phase = phase
        if phase == "SEARCH":
            self._search_t0 = self._search_step_t = now

    def _cmd(self, note: str = "", land: bool = False) -> Command:
        return Command(self.phase, self.tx, self.ty, self.h_target, land, note, self._seen_pad)

    def _descend(self, dt: float, height: float) -> None:
        c = self.cfg
        self.h_target = max(c.commit_height_m, self.h_target - c.descent_rate_mps * dt)
        self.h_target = max(self.h_target, height - c.descent_lead_m)

    # -- ana adim
    def update(self, now: float, drone_xy: Tuple[float, float], height: float,
               est: Optional[Tuple[float, float]]) -> Command:
        c = self.cfg
        dt = min(max(now - self._t, 0.0), 0.5)
        self._t = now
        if est is not None:
            self.last_known = est
            self._last_est_t = now
            self._seen_pad = True

        if self.phase in ("LAND",):
            return self._cmd(land=True)

        if now - self._t0 > c.mission_timeout_s:
            self._go("LAND", now, "gorev zaman asimi")
            return self._cmd("gorev zaman asimi", land=True)

        desired = est if est is not None else (self.goal if not self._seen_pad else self.last_known)
        dist = math.hypot(drone_xy[0] - desired[0], drone_xy[1] - desired[1])

        if self.phase == "APPROACH":
            self.h_target = max(self.h_target, c.start_alt_m) if height < c.start_alt_m - 0.5 else c.start_alt_m
            if dist <= c.arrive_tol_m:
                if est is not None:
                    self._go("DESCEND", now, f"pad gorundu, hata {dist:.2f} m")
                else:
                    if self._wait_since is None:
                        self._wait_since = now
                    elif now - self._wait_since > c.acquire_s:
                        self._go("SEARCH", now, "kayitli konuma varildi ama pad gorulmuyor")
            else:
                self._wait_since = None
            self._slew(desired, dt)
            return self._cmd()

        if self.phase == "DESCEND":
            note = ""
            lost = now - self._last_est_t
            if est is None and height > c.blind_below_m and lost > c.lost_s:
                self._go("SEARCH", now, f"{lost:.1f} s gorsel tahmin yok")
                self._slew(desired, dt)
                return self._cmd()
            allowed = max(c.lat_min_m, c.lat_k * height)
            if est is None:
                note = "pad kadrajda degil, son merkezle iniliyor"
                self._descend(dt, height)             # alt irtifada kor devam
            elif dist <= allowed:
                self._descend(dt, height)
            else:
                note = f"ortalaniyor (hata {dist:.2f} > izinli {allowed:.2f} m), alcalma durdu"
            self._slew(desired, dt)
            if height <= c.commit_height_m + 0.15:
                self._go("LAND", now, f"commit {height:.2f} m, son merkeze kilitlendi")
                return self._cmd("commit", land=True)
            return self._cmd(note)

        if self.phase == "SEARCH":
            if est is not None:
                self._go("DESCEND", now, "pad bulundu")
                self._slew(desired, dt)
                return self._cmd()
            ceiling = c.start_alt_m + c.search_max_extra_m
            if now - self._search_step_t >= c.search_hold_s and self.h_target < ceiling:
                self.h_target = min(ceiling, max(self.h_target, height) + c.search_step_m)
                self._search_step_t = now
            if now - self._search_t0 > c.search_timeout_s:
                self._go("BLIND", now, "pad bulunamadi, son bilinen konuma kor inis")
            self._slew(desired, dt)
            return self._cmd("pad araniyor")

        if self.phase == "BLIND":
            self._descend(dt, height)
            self._slew(desired, dt)
            if height <= c.commit_height_m + 0.15:
                self._go("LAND", now, "kor inis commit")
                return self._cmd("PAD GORULMEDI: kor inis", land=True)
            return self._cmd("kor inis")

        return self._cmd()
