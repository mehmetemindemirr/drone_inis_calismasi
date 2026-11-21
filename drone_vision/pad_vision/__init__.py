"""pad_vision: renkli iniş pad'i tespiti. ROS'tan bağımsız, yalnızca OpenCV + NumPy."""
from .coverage import ColorCoverage
from .detector import HsvPadDetector, PadDetector
from .models import (BLUE, RED, ColorProfile, DetectionHint, DetectorConfig,
                     PadDetection, expected_radius_px)
from .geometry import (CameraModel, GeometryError, PadLocation, altitude_from_radius_px,
                       footprint_m, ground_offset_to_pixel, latlon_to_offset, locate_pad,
                       offset_to_latlon, pixel_to_ground_offset, quaternion_from_euler_deg,
                       quaternion_from_yaw, quaternion_to_matrix, vertical_height_from_range)
from .viz import draw_detections

__all__ = [
    "HsvPadDetector", "PadDetector", "ColorCoverage",
    "ColorProfile", "RED", "BLUE",
    "DetectorConfig", "DetectionHint", "PadDetection",
    "expected_radius_px", "draw_detections",
    "CameraModel", "GeometryError", "PadLocation", "locate_pad",
    "pixel_to_ground_offset", "ground_offset_to_pixel",
    "offset_to_latlon", "latlon_to_offset",
    "altitude_from_radius_px", "vertical_height_from_range", "footprint_m",
    "quaternion_from_yaw", "quaternion_from_euler_deg", "quaternion_to_matrix",
]
