# drone_vision: kırmızı/mavi iniş pedi tespiti

Aşağıya bakan kameradan 1 m yarıçaplı kırmızı/mavi daire pad'i bulur. Karar vermez, uçuşa karışmaz:
sadece görür ve bildirir. ROS'a bağımlı olan tek yer `adapters/` ve `tools/`; `pad_vision/` saf Python + OpenCV + NumPy.

Durum: simülasyonda (Gazebo Classic + ArduPilot + MAVROS) doğrulandı, ölçülen fx ≈ 385 px (kuramsal 381). Gerçek kamera henüz denenmedi.

## Yapı
| Yol | Görev |
|---|---|
| `pad_vision/` | Tespit çekirdeği: HSV maske → şekil analizi → güven skoru; `coverage.py` kadrajın kaç %'i kırmızı/mavi |
| `pad_vision/geometry.py` | Piksel → drone'a göre (Doğu, Kuzey) ofset → yerel x,y → enlem-boylam; eğim (roll/pitch) düzeltmeli, ROS'suz (kullanım aşağıda) |
| `adapters/ros_image.py` | ROS görüntü mesajı → OpenCV karesi (cv_bridge yok) |
| `adapters/pad_watcher.py` | Gözlemci: en son kareyi alır, tespit eder, konsola yazar, fx'i ölçer, hata olsa uçuşu bozmaz |
| `tools/live_detect.py` | Canlı izleme (ayrı süreç) |
| `tools/live_locate.py` | Canlı konum tahmini; senin verdiğin GERÇEK pad konumuyla karşılaştırır, hatayı yazar, Ctrl+C'de teşhis basar (ayrı süreç) |
| `tools/pad_recorder.py` | Canlı pad kaydedici: gözlem → `PadMap` (kapı + medyan) → yerel x,y + enlem-boylam, JSON/topic çıktısı (ayrı süreç) |
| `pad_vision/landing.py` | İniş mantığı (ROS'suz): `load_record`/`choose_target`, `VisionTracker`, `LandingController` (APPROACH→DESCEND→LAND; SEARCH/BLIND yedekleri) |
| `tools/pad_lander.py` | TEK drone hassas iniş görevi: `yaw_v3` durum makinesi + `LandingController`; MAVROS'u kendisi başlatır |
| `tools/save_frames.py` | Kamera + konumu kaydeder (PNG + CSV) |
| `tools/detect_images.py` | Kayıtlı karelerde çevrimdışı tespit (ROS gerekmez) |
| `pad_vision/padmap.py` | `PadMap`: gözlemleri kalite kapılarından geçirir, kümeler, medyanla tek pad konumu (yerel x,y + enlem-boylam) + JSON kaydı; ROS'suz |
| `tests/` | 180 test, ROS gerekmez |

## Kurulum
Gerekenler: `python3`, `python3-opencv`, `python3-numpy`; canlı araçlar için ROS 2 (Humble: simülasyon, Jazzy: Pi).
Testler: `python3 -m unittest discover -s tests` (hepsi geçmeli).

## Simülasyonda çalıştırma (4 terminal, her birinde `source /opt/ros/humble/setup.bash`)
1. Gazebo → 2. `sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map -I0` → 3. `python3 yaw.py` (MAVROS'u kendisi başlatır)
4. `python3 tools/live_detect.py --topic /drone/down_camera/image_raw --hfov 80 --use-hint --publish`
(Görüntü: `rqt_image_view` → `/pad_vision/debug_image`)

## Raspberry Pi + gerçek kamera (drone yok)
Kamera ROS topic'i olarak yayınlanmalı (`camera_ros`, öneri 640x480; 1280x960 Pi'yi yorar, 640x480 sensör kırpması yapabilir — `kamera-kurulum-notlari.md`'ye bak).
1. `source /opt/ros/jazzy/setup.bash`
2. `python3 tools/live_detect.py --topic /camera/image_raw --hfov 62.2` (**`--use-hint` verme**: drone/irtifa yok)
3. Ekranda tam kırmızı bir daire göster (siyah/gri zemin, parlaklık düşük, ekrana dik, yarıçap ≥ ~20–30 px).
4. Bulamazsa: `python3 tools/save_frames.py --out kareler --every 1 --max 20 --image-topic /camera/image_raw`
   sonra `python3 tools/detect_images.py kareler --out sonuc` ve `sonuc/` içindeki işaretli kareye bak.
5. Renkler ters çıkıyorsa `--swap-rb`. Kırmızı hiç bulunmuyorsa HSV aralığı (aşağıda) ayarlanır.
Pi Camera V2 **NoIR**'da kızılötesi filtre yok: renkler kayar, HSV yeniden ayarlanması beklenir.
Başsız (ekransız) Pi: `--publish` + başka bilgisayarda `rqt_image_view`, veya kareleri kaydedip bilgisayarda incele.

## Log nasıl okunur
- `KIRMIZI gorundu/goruluyor/kayboldu`: merkez (px), yarıçap (px), `guven`, irtifa.
- `guven` doğruluk yüzdesi değil, aday **sıralama skoru**dur.
- `KADRAJ KENARINDA`: pad kesik, merkezi güvenilmez; kaydetme.
- `KADRAJIN %X'I KIRMIZI`: pad kadraja sığmıyor, tam üstündeyiz (tam kırmızı karede daire aranmaz, bu bilerek).
- `olculen fx~…`: gerçek kameranın odak uzaklığı (fx = r_px·irtifa/R); `--hfov` değerini doğrulamak için bak.

## Ana göreve (yaw.py) ekleme: en son, canlı araç sorunsuz çalıştıktan sonra
```python
import sys; sys.path.insert(0, '/YOL/drone_vision')
from adapters.pad_watcher import PadWatcher
from adapters.ros_image import subscribe_image
from pad_vision import HsvPadDetector, ColorCoverage, DetectorConfig
# __init__:
self.pad = PadWatcher(HsvPadDetector(config=DetectorConfig(work_width=320)), ColorCoverage(),
                      hfov_deg=80.0, use_hint=True, every_s=1.0)
subscribe_image(self, '/drone/down_camera/image_raw', self.pad.on_image)
# tick() başında:
self.pad.update(self.rel_alt())
```
Pi'de görüntüyü ana süreçte işlemek `spin_once` döngüsünü yavaşlatabilir; önce ayrı süreç olarak kullan.

## Pad'in gerçek konumu (`pad_vision/geometry.py`)
Tespit edilen pikseli drone'a göre metreye, oradan yerel x,y'ye ve enlem-boylama çevirir. Karar vermez.
```python
from pad_vision import CameraModel, locate_pad
cam = CameraModel.from_hfov(640, 480, 62.2)          # KULLANILAN görüntü çözünürlüğüne göre
q = pose.pose.orientation                            # /mavros/local_position/pose
loc = locate_pad(det, cam, height_m=rel_alt,         # pad düzlemine DİKEY yükseklik
                 q_enu=(q.x, q.y, q.z, q.w),
                 drone_local_xy=(pose.pose.position.x, pose.pose.position.y),
                 drone_latlon=(gps.latitude, gps.longitude))   # /mavros/global_position/global
# loc.east_m, loc.north_m  -> yerel ENU (İNİŞ için bunu kullan)
# loc.lat_deg, loc.lon_deg -> kayıt/paylaşım için (GPS doğruluğunda)
```
- Sözleşmeler: ENU, yaw 0° = Doğu, 90° = Kuzey. Kamera aşağı bakar, görüntünün üstü burun. Başka takıldıysa `mount_yaw_deg`.
- **Euler'de pozitif pitch = burun AŞAĞI, pozitif roll = sağ kanat AŞAĞI** (ROS). Çoğu zaman quaternion'u olduğu gibi ver.
- Yükseklik hatası konuma doğrudan yansır (5 m'de 0.5 m hata ≈ %10). Baro yerine ARM'a göre `rel_alt` ya da lidar
  (`vertical_height_from_range` ile eğim düzeltmeli) kullan; `altitude_from_radius_px` ile pad yarıçapından çapraz doğrula.
- `touches_border=True` olan tespitte merkez önyargılı: konum hedefi yapma.
- Simülasyonda gerçek pad konumunu bilirsin: `ground_offset_to_pixel` ile görüntüye izdüşür, dedektörün pikseliyle karşılaştır.
- Lens bozulması yok sayılır, zemin düz varsayılır, kare zamanı ile konum zamanı eşlenmez (gerçek kamerada sırayla eklenecek).

### Konum doğruluğu testi (simülasyon): `tools/live_locate.py`
Drone (0,0)'da doğdu, pad 5 m Kuzeyde ise (4. terminal, `live_detect.py` yerine):
```bash
python3 tools/live_locate.py --pad-east 0 --pad-north 5 --csv sonuc.csv
```
Pad konumu MAVROS yerel ENU'dadır (x=Doğu, y=Kuzey), drone'un başlangıcına göre. Gazebo +x = Kuzey olduğundan pad'i
Gazebo'da x ekseninde verdiysen `--pad-north` ile ver; Doğu işaretini ilk çıktıya bakarak doğrula.
1. **Görünürlük:** hfov 80°'de 5 m irtifada kadraj yaklaşık 8.4 x 6.2 m; pad 5 m uzakta olduğundan en az ~10 m'ye çık
   (ya da drone'u pad'e yaklaştır). Pad kadraj kenarına değerse tespit "kesik" sayılıp atlanır.
2. Önce drone durağan (pad'e uzak, 10 m), sonra 5 m ve farklı yaw'larda tekrarla; sonra `yaw.py`/`yaw_v3.py` ile uçarken.
3. Ctrl+C'de **TEŞHİS** satırlarını oku: `Kamera yönü ... --mount-yaw X` (kamera burna göre dönük takılı) ve
   `Ölçek ... %N` (yükseklik/hfov hatası). Önerileri uygulayıp tekrar dene. Hedef: durgun örneklerde 5-10 m'de ~0.3-0.5 m altı hata.
4. Hata sabit bir yöne kayıyorsa (bias) kamera/yükseklik; rastgele dağılıyorsa tespit gürültüsü; hız arttıkça büyüyorsa
   kare-konum zaman uyumsuzluğu (bu araç kareyi işlendiği andaki en son konumla eşler).

## Kayıt → iniş (tek drone): `pad_recorder.py` + `pad_lander.py`
```bash
python3 tools/pad_recorder.py --pad-east 0 --pad-north 5 --out kayit.json     # pad'in üstünden geç, drone'u indir
python3 tools/pad_lander.py --record kayit.json --pad-east 0 --pad-north 5 --log inis.csv   # MAVROS'u kendisi başlatır
```
Mavi pad: iki araçta da `--color mavi` (recorder'da `--color hepsi` ikisini aynı `kayit.json`'a yazar; `--truth-color` gerçek konumun hangi renge ait olduğunu söyler).
Sahada gerçek pad konumu verme; inişten sonra pad merkezi ile drone arasını mezurayla ölç. Gerçek drone: `--fcu-url /dev/ttyS0:115200 --sync-max 240 --topic ... --hfov 62.2`.
`tests/test_pad_lander.py` düğümü SAHTE rclpy + basit fizikle baştan sona koşturur (duman testi, gerçek ROS/Gazebo değildir).

## Ayarlanacak eşikler (`DetectorConfig`, `models.py`)
| Alan | Ne zaman değiştir |
|---|---|
| `RED` / `BLUE` HSV aralıkları | gerçek kamerada ilk iş: pad kaçırılıyorsa S/V alt sınırını düşür |
| `min_radius_px` | yüksekten (30 m+) tespit gerekiyorsa düşür |
| `min_circularity`, `min_ellipse_iou`, `max_rect_fill` | gerçek pad eleniyorsa gevşet, yanlış pozitif varsa sıkılaştır |
| `size_tolerance` | irtifa ölçümü gürültülüyse artır |
| `min_confidence` | son çare |

## Bilinen sınırlar
- Eşikler sentetik sahne + Gazebo ile ayarlı; gerçek kamerada yeniden ayar beklenir.
- Pad kadrajdan fazla taşarsa reddedilir; son metreler son bilinen merkeze göre inilmeli.
- Düzgün altıgen/sekizgen daireden ayırt edilemez: ardışık kare onayı (sonraki katman) gerekir.
- Görüntü ve konum zamanları farklı saatlerden gelir (Gazebo sim zamanı / MAVROS duvar saati); konum kaydında `poz_yasi_s` filtrelenmeli.
- Eksenler: MAVROS yerel ENU (yaw 0° = Doğu). Gazebo +x = Kuzey = MAVROS +y.

## Sonraki adımlar
Ardışık kare onayı → piksel→yerel x,y→enlem-boylam + konum kaydı → hassas iniş → sürü paylaşımı.
