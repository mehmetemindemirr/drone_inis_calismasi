"""Goruntu yardimcilari: girdi dogrulama ve kucultme. Sadece OpenCV + NumPy."""
from __future__ import annotations

import cv2
import numpy as np

#ensure_bgr bgr den emin olmak demektir. 
"""
ros kamera sürücüleri rgb (rbg9) şeklinde yayın yaparlar. eğer adaptör bunu çevirmezse kırmızı mavi mavi kırmızı görülebilir. 
burada ensure_bgr görüntünün istenilen formatta mı geldiğini doğrulacaktır. 
görüntünün ndarray olup olmadığını 
ndim boyut sayısını
shape kanal sayısını 
uint8 ise 0 - 255 arası değer alabileceğini
ifade eder. 
"""


def ensure_bgr(image) -> None:
    """Girdi (H, W, 3) uint8 BGR dizisi degilse ValueError firlatir."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8):
        raise ValueError("image_bgr: (H, W, 3) uint8 BGR dizisi olmali")
"""
INTER_AREA sayesinde komşu piksellerin ortalaması alınarak kenar tırtılanmasını ve gürültüyü azaltır. 
"""

def downscale_to_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Genislik max_width'ten buyukse oranini koruyarak kucultur; kucukse dokunmaz."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    nw = int(max_width)
    nh = max(1, int(round(h * nw / w)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
