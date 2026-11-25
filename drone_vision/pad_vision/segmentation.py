"""Renk bolutleme: HSV goruntu + renk profili -> ikili maske. Tek sorumluluk."""
from __future__ import annotations

import cv2
import numpy as np

from .models import ColorProfile

"""
ilk başta boş bir  zerolardan oluşan bir matris oluşturur. 
sonra for ile 2 kırmızı aralığını döner. 


"""

class ColorSegmenter:
    def __init__(self, kernel_frac: float = 0.01):
        self._kernel_frac = kernel_frac

    def mask(self, hsv: np.ndarray, profile: ColorProfile) -> np.ndarray:
        h, w = hsv.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for lo, hi in profile.ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        k = max(3, int(round(self._kernel_frac * w)))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # tek tuk benekleri sil
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # ic delikleri kapat
        return mask

"""
1. cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) (Açma İşlemi):
	• Önce aşındırma (erosion), sonra genişletme (dilation) yapar.
	• Etkisi: Pistin dışındaki yerlerde bulunan 9×9 pikselden küçük kırmızı tozları, ışık parlamalarını ve tek tük gürültü beneklerini tamamen siler.
2. cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) (Kapama İşlemi):
	• Önce genişletme (dilation), sonra aşındırma (erosion) yapar.
	• Etkisi: Kırmızı iniş pistinin üzerindeki gölge, yansıma veya pist içindeki yazılar/çizgiler yüzünden oluşmuş küçük siyah delikleri (kopuklukları) doldurup pisti tek parça,
    dolu bir beyaz daire haline getirir.
"""


"""
Evet, tam olarak öyle! İster dairenin sol tarafı olsun, ister sağ tarafı, ister üzerindeki gölgeli bir nokta olsun; 
iki farklı aralıktan yakalanan tüm kırmızı pikselleri eksiksiz birleştirip tek parça bütün bir daire haline getirir.

1. Aralık Maskesi (Sadece sol tarafı yakaladı):
[ 255,   0,   0 ]
[ 255,   0,   0 ]
[ 255,   0,   0 ]

2. Aralık Maskesi (Sadece sağ tarafı yakaladı):

[   0,   0, 255 ]
[   0,   0, 255 ]
[   0,   0, 255 ]

3. mask1 | mask2 (bitwise_or) Sonucu (İkisini birleştirdi):

[ 255,   0, 255 ]
[ 255,   0, 255 ]
[ 255,   0, 255 ]

amaç kırmızı alanı tam görebilmektir. 
neden toplama değilde or yaptık nedeni taşma olurdu uint8 kullandık. 

"""

