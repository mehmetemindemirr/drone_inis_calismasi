"""geometry.py testleri: isaret yonleri, yaw, egim duzeltmesi, ileri-geri tutarlilik, enlem-boylam."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pad_vision import PadDetection, expected_radius_px  # noqa: E402
from pad_vision.geometry import (CameraModel, GeometryError, altitude_from_radius_px,  # noqa: E402
                                 footprint_m, ground_offset_to_pixel, latlon_to_offset,
                                 locate_pad, offset_to_latlon, pixel_to_ground_offset,
                                 quaternion_from_euler_deg, quaternion_from_yaw,
                                 quaternion_to_matrix, vertical_height_from_range)

CAM = CameraModel.from_hfov(640, 480, 62.2)
H = 10.0


def q(yaw=0.0, pitch=0.0, roll=0.0):
    return quaternion_from_euler_deg(roll, pitch, yaw)


def det_at(u, v, touches=False):
    return PadDetection(color="kirmizi", center_px=(u, v), radius_px=40.0, area_px=5000.0,
                        confidence=0.9, circularity=0.95, touches_border=touches)


class CameraTests(unittest.TestCase):
    def test_from_hfov(self):
        self.assertAlmostEqual(CAM.fx, 320.0 / math.tan(math.radians(31.1)), places=6)
        self.assertEqual(CAM.fx, CAM.fy)
        self.assertEqual((CAM.cx, CAM.cy), (319.5, 239.5))

    def test_from_hfov_rejects_bad_angle(self):
        with self.assertRaises(ValueError):
            CameraModel.from_hfov(640, 480, 0.0)
        with self.assertRaises(ValueError):
            CameraModel.from_hfov(640, 480, 180.0)

    def test_scaled_matches_direct(self):
        a = CAM.scaled(320)
        b = CameraModel.from_hfov(320, 240, 62.2)
        for x, y in ((a.fx, b.fx), (a.fy, b.fy), (a.cx, b.cx), (a.cy, b.cy)):
            self.assertAlmostEqual(x, y, places=9)
        self.assertEqual((a.width, a.height), (320, 240))

    def test_footprint(self):
        cam = CameraModel.from_hfov(640, 480, 90.0)          # fx = w/2
        w, h = footprint_m(cam, 10.0)
        self.assertAlmostEqual(w, 20.0, places=9)
        self.assertAlmostEqual(h, 15.0, places=9)
        with self.assertRaises(GeometryError):
            footprint_m(cam, 0.0)


class QuaternionTests(unittest.TestCase):
    def test_yaw_matches_euler(self):
        for yaw in (-170.0, -30.0, 0.0, 45.0, 90.0, 200.0):
            a = np.array(quaternion_from_yaw(yaw))
            b = np.array(quaternion_from_euler_deg(0, 0, yaw))
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_yaw90_points_nose_north(self):
        nose = quaternion_to_matrix(quaternion_from_yaw(90.0)) @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(nose, [0.0, 1.0, 0.0], atol=1e-12)

    def test_positive_pitch_is_nose_down(self):
        nose = quaternion_to_matrix(q(pitch=10.0)) @ np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(nose[2], -math.sin(math.radians(10.0)), places=12)

    def test_positive_roll_is_right_wing_down(self):
        right = quaternion_to_matrix(q(roll=10.0)) @ np.array([0.0, -1.0, 0.0])   # govde sagi = -y
        self.assertAlmostEqual(right[2], -math.sin(math.radians(10.0)), places=12)

    def test_unnormalised_ok_and_zero_rejected(self):
        a = quaternion_to_matrix((0.0, 0.0, 2.0 * math.sin(0.3), 2.0 * math.cos(0.3)))
        b = quaternion_to_matrix((0.0, 0.0, math.sin(0.3), math.cos(0.3)))
        np.testing.assert_allclose(a, b, atol=1e-12)
        with self.assertRaises(GeometryError):
            quaternion_to_matrix((0, 0, 0, 0))


class PixelToGroundTests(unittest.TestCase):
    def test_center_pixel_is_directly_below(self):
        for yaw in (0.0, 37.0, 90.0, -140.0):
            e, n = pixel_to_ground_offset(CAM.cx, CAM.cy, CAM, H, q(yaw))
            self.assertAlmostEqual(e, 0.0, places=9)
            self.assertAlmostEqual(n, 0.0, places=9)

    def test_nose_north_up_is_north_right_is_east(self):
        up = pixel_to_ground_offset(CAM.cx, CAM.cy - CAM.fy * 0.5, CAM, H, q(yaw=90.0))
        self.assertAlmostEqual(up[0], 0.0, places=9)
        self.assertAlmostEqual(up[1], 5.0, places=9)             # goruntunun ustu = ileri = Kuzey
        right = pixel_to_ground_offset(CAM.cx + CAM.fx * 0.5, CAM.cy, CAM, H, q(yaw=90.0))
        self.assertAlmostEqual(right[0], 5.0, places=9)          # sag = Dogu
        self.assertAlmostEqual(right[1], 0.0, places=9)

    def test_nose_east(self):
        up = pixel_to_ground_offset(CAM.cx, CAM.cy - CAM.fy * 0.5, CAM, H, q(yaw=0.0))
        self.assertAlmostEqual(up[0], 5.0, places=9)             # ileri = Dogu
        right = pixel_to_ground_offset(CAM.cx + CAM.fx * 0.5, CAM.cy, CAM, H, q(yaw=0.0))
        self.assertAlmostEqual(right[1], -5.0, places=9)         # sag = Guney

    def test_nose_west(self):
        up = pixel_to_ground_offset(CAM.cx, CAM.cy - CAM.fy * 0.5, CAM, H, q(yaw=180.0))
        self.assertAlmostEqual(up[0], -5.0, places=9)
        self.assertAlmostEqual(up[1], 0.0, places=9)

    def test_scales_linearly_with_height(self):
        a = pixel_to_ground_offset(500.0, 100.0, CAM, 5.0, q(yaw=20.0))
        b = pixel_to_ground_offset(500.0, 100.0, CAM, 15.0, q(yaw=20.0))
        self.assertAlmostEqual(b[0], 3.0 * a[0], places=9)
        self.assertAlmostEqual(b[1], 3.0 * a[1], places=9)

    def test_mount_yaw_rotates_image_up(self):
        # goruntu-ustu burnun soluna bakiyor (90 derece CCW); burun Kuzey -> goruntu-ustu Bati
        e, n = pixel_to_ground_offset(CAM.cx, CAM.cy - CAM.fy * 0.5, CAM, H, q(yaw=90.0),
                                      mount_yaw_deg=90.0)
        self.assertAlmostEqual(e, -5.0, places=9)
        self.assertAlmostEqual(n, 0.0, places=9)

    def test_lever_arm_center_pixel(self):
        # kamera govdenin 0.1 m onunde ve 0.05 m altinda; burun Kuzey
        e, n = pixel_to_ground_offset(CAM.cx, CAM.cy, CAM, H, q(yaw=90.0),
                                      lever_body_m=(0.1, 0.0, -0.05))
        self.assertAlmostEqual(e, 0.0, places=9)
        self.assertAlmostEqual(n, 0.1, places=9)

    def test_errors(self):
        with self.assertRaises(GeometryError):
            pixel_to_ground_offset(CAM.cx, CAM.cy, CAM, 0.0)
        with self.assertRaises(GeometryError):
            pixel_to_ground_offset(CAM.cx, CAM.cy, CAM, -2.0)
        with self.assertRaises(GeometryError):                    # 80 derece yatik: isin siyiriyor
            pixel_to_ground_offset(CAM.cx, CAM.cy, CAM, H, q(roll=80.0))


class TiltTests(unittest.TestCase):
    """Egimin yonu ve duzeltmenin gercekten isledigi (fiziksel dogrulama)."""

    def test_nose_up_pad_below_appears_toward_tail(self):
        # burun yukari 10 derece: kamera ileri-asagi bakar, tam altimizdaki pad goruntunun ALT tarafina duser
        u, v = ground_offset_to_pixel(0.0, 0.0, CAM, H, q(pitch=-10.0))
        self.assertAlmostEqual(u, CAM.cx, places=6)
        self.assertGreater(v, CAM.cy)
        self.assertAlmostEqual(v, CAM.cy + CAM.fy * math.tan(math.radians(10.0)), places=6)

    def test_pitch_compensation_recovers_zero(self):
        att = q(yaw=0.0, pitch=-10.0)
        u, v = ground_offset_to_pixel(0.0, 0.0, CAM, H, att)
        e, n = pixel_to_ground_offset(u, v, CAM, H, att)
        self.assertAlmostEqual(e, 0.0, places=9)
        self.assertAlmostEqual(n, 0.0, places=9)
        # egimi yok sayarsak (duz varsayim) pad'i kuyruga dogru H*tan(10) kayik sanardik
        e0, n0 = pixel_to_ground_offset(u, v, CAM, H, quaternion_from_yaw(0.0))
        self.assertAlmostEqual(e0, -H * math.tan(math.radians(10.0)), places=9)
        self.assertAlmostEqual(n0, 0.0, places=9)

    def test_roll_compensation(self):
        att = q(yaw=0.0, roll=10.0)                               # sag kanat asagi
        u, v = ground_offset_to_pixel(0.0, 0.0, CAM, H, att)
        self.assertGreater(u, CAM.cx)                             # altimizdaki pad sagda gorunur
        e, n = pixel_to_ground_offset(u, v, CAM, H, att)
        self.assertAlmostEqual(e, 0.0, places=9)
        self.assertAlmostEqual(n, 0.0, places=9)
        e0, n0 = pixel_to_ground_offset(u, v, CAM, H, quaternion_from_yaw(0.0))
        self.assertAlmostEqual(n0, -H * math.tan(math.radians(10.0)), places=9)   # yaw 0: sag = Guney


class RoundTripTests(unittest.TestCase):
    def test_pixel_ground_pixel(self):
        poses = [(0, 0, 0), (5, -8, 37), (-12, 9, -120), (3, 3, 200)]      # (roll, pitch, yaw)
        mounts = [0.0, 90.0, -30.0]
        levers = [(0.0, 0.0, 0.0), (0.1, -0.05, -0.08)]
        offsets = [(0.0, 0.0), (3.0, -2.0), (-6.0, 4.5)]
        for roll, pitch, yaw in poses:
            for m in mounts:
                for lv in levers:
                    for de, dn in offsets:
                        with self.subTest(pose=(roll, pitch, yaw), mount=m, lever=lv, off=(de, dn)):
                            att = q(yaw, pitch, roll)
                            u, v = ground_offset_to_pixel(de, dn, CAM, 8.0, att, m, lv)
                            e, n = pixel_to_ground_offset(u, v, CAM, 8.0, att, m, lv)
                            self.assertAlmostEqual(e, de, places=8)
                            self.assertAlmostEqual(n, dn, places=8)

    def test_behind_camera_rejected(self):
        with self.assertRaises(GeometryError):
            ground_offset_to_pixel(1.0, 1.0, CAM, -1.0)


class LatLonTests(unittest.TestCase):
    def test_equator_one_degree(self):
        lat, lon = offset_to_latlon(0.0, 0.0, 111319.49, 0.0)
        self.assertAlmostEqual(lon, 1.0, places=5)
        self.assertAlmostEqual(lat, 0.0, places=9)
        lat, lon = offset_to_latlon(0.0, 0.0, 0.0, 110574.3)
        self.assertAlmostEqual(lat, 1.0, places=3)

    def test_known_scale_at_40_north(self):
        lat, lon = offset_to_latlon(40.0, 39.0, 100.0, 100.0)
        self.assertAlmostEqual(lat - 40.0, 100.0 / 111040.0, delta=2e-6)     # ~0.000900 derece
        self.assertAlmostEqual(lon - 39.0, 100.0 / 85394.0, delta=2e-6)      # ~0.001171 derece

    def test_directions(self):
        lat, lon = offset_to_latlon(40.0, 39.0, 0.0, 50.0)
        self.assertGreater(lat, 40.0)
        self.assertAlmostEqual(lon, 39.0, places=12)
        lat, lon = offset_to_latlon(40.0, 39.0, -50.0, 0.0)
        self.assertLess(lon, 39.0)
        self.assertAlmostEqual(lat, 40.0, places=12)

    def test_roundtrip(self):
        for lat0, lon0 in ((40.44, 39.52), (-35.36, 149.17), (0.0, 0.0), (60.0, -120.0)):
            lat, lon = offset_to_latlon(lat0, lon0, 37.5, -12.25)
            e, n = latlon_to_offset(lat0, lon0, lat, lon)
            self.assertAlmostEqual(e, 37.5, places=6)
            self.assertAlmostEqual(n, -12.25, places=6)

    def test_longitude_wraps(self):
        _, lon = offset_to_latlon(0.0, 179.99999, 5.0, 0.0)
        self.assertLess(lon, 0.0)
        self.assertGreaterEqual(lon, -180.0)

    def test_pole_rejected(self):
        with self.assertRaises(GeometryError):
            offset_to_latlon(89.95, 0.0, 1.0, 1.0)


class LocatePadTests(unittest.TestCase):
    def test_full_chain(self):
        # drone yerel (10, 20), yaw 37, hafif egik, 8 m, GPS (40.0, 39.0); pad drone'a gore (3, -2)
        att = q(yaw=37.0, pitch=2.0, roll=-1.5)
        u, v = ground_offset_to_pixel(3.0, -2.0, CAM, 8.0, att)
        loc = locate_pad(det_at(u, v), CAM, 8.0, att, drone_local_xy=(10.0, 20.0),
                         drone_latlon=(40.0, 39.0))
        self.assertAlmostEqual(loc.offset_east_m, 3.0, places=8)
        self.assertAlmostEqual(loc.offset_north_m, -2.0, places=8)
        self.assertAlmostEqual(loc.distance_m, math.hypot(3.0, 2.0), places=8)
        self.assertAlmostEqual(loc.east_m, 13.0, places=8)
        self.assertAlmostEqual(loc.north_m, 18.0, places=8)
        exp_lat, exp_lon = offset_to_latlon(40.0, 39.0, 3.0, -2.0)
        self.assertAlmostEqual(loc.lat_deg, exp_lat, places=12)
        self.assertAlmostEqual(loc.lon_deg, exp_lon, places=12)
        self.assertEqual(loc.color, "kirmizi")
        self.assertFalse(loc.touches_border)

    def test_optional_outputs_are_none(self):
        loc = locate_pad(det_at(CAM.cx + 50, CAM.cy - 20, touches=True), CAM, 6.0)
        self.assertIsNone(loc.east_m)
        self.assertIsNone(loc.north_m)
        self.assertIsNone(loc.lat_deg)
        self.assertIsNone(loc.lon_deg)
        self.assertTrue(loc.touches_border)


class HeightHelperTests(unittest.TestCase):
    def test_altitude_from_radius_inverts_expected_radius(self):
        for h in (3.0, 7.5, 20.0):
            r = expected_radius_px(h, CAM.fx, 1.0)
            self.assertAlmostEqual(altitude_from_radius_px(r, CAM.fx, 1.0), h, places=9)
        with self.assertRaises(GeometryError):
            altitude_from_radius_px(0.0, CAM.fx)

    def test_vertical_height_from_range(self):
        self.assertAlmostEqual(vertical_height_from_range(10.0), 10.0, places=12)
        self.assertAlmostEqual(vertical_height_from_range(10.0, q(pitch=20.0)),
                               10.0 * math.cos(math.radians(20.0)), places=12)
        with self.assertRaises(GeometryError):
            vertical_height_from_range(10.0, q(roll=100.0))


if __name__ == "__main__":
    unittest.main()
