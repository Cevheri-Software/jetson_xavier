from ultralytics import YOLO
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import math


results = {}

mot_tracker = Sort()

# Araçlar için tracker zaten var: mot_tracker = Sort()
# Silahlar için ayrı bir tracker başlat
mot_gun_tracker = Sort()

# load models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
gun_detector = YOLO('gun_deteckter.pt')

coco_model.to(device)
license_plate_detector.to(device)
gun_detector.to(device)

# load video
cap = cv2.VideoCapture(0)

# Kamera parametrelerini ayarla
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# GStreamer pipeline için video writer oluştur
# Hedef IP adresini değiştirin (Ubuntu bilgisayarın IP'si)
TARGET_IP = "192.168.1.100"  # Ubuntu bilgisayarın IP adresini buraya yazın
TARGET_PORT = 5000

# UDP streaming için GStreamer pipeline
gstreamer_pipeline = (
    "appsrc ! "
    "videoconvert ! "
    "video/x-raw,format=I420 ! "
    "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    f"udpsink host={TARGET_IP} port={TARGET_PORT}"
)

# Alternative RTMP streaming pipeline (uncomment if you prefer RTMP)
# gstreamer_pipeline = (
#     "appsrc ! "
#     "videoconvert ! "
#     "video/x-raw,format=I420 ! "
#     "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
#     "flvmux ! "
#     f"rtmpsink location=rtmp://{TARGET_IP}:1935/live/stream"
# )

# VideoWriter oluştur
fourcc = cv2.VideoWriter_fourcc(*'X264')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(gstreamer_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))

if not out.isOpened():
    print("GStreamer pipeline açılamadı! Lütfen hedef IP ve port'u kontrol edin.")
    exit(1)

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
guns_results = []  # Silah tespitleri için liste

# Silah tracking için ID ve sayım tutmak için dict
tracked_guns = {}  # gun_id: {'count': int, 'last_frame': int, 'bbox': list, 'score': float, 'class_id': int}
GUN_TRACK_MIN_FRAMES = 15

# Okunan plakaları tutmak için bir küme
detected_plates = set()

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

gun_buffer = []  # Her silah tespiti için takip listesi
confirmed_guns = []  # Doğrulanan silahlar (CSV'ye yazılacaklar)
BUFFER_FRAMES = 5
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.6

MIN_GUN_AREA = 800  # Piksel cinsinden, ihtiyaca göre ayarla
MAX_GUN_AREA = 50000  # Piksel cinsinden, ihtiyaca göre ayarla

print(f"Streaming başlatılıyor... Hedef: {TARGET_IP}:{TARGET_PORT}")
print("Çıkmak için 'q' tuşuna basın")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera bağlantısı kesildi!")
            break
            
        frame_nmr += 1
        results[frame_nmr] = {}
        
        # detect vehicles
        detections = coco_model(frame, device=device)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = mot_tracker.update(np.empty((0, 5)))

        # detect license plates
        license_plates = license_plate_detector(frame, device=device)[0]
        drawn_car_ids = set()
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1 and car_id not in drawn_car_ids:
                # Araç kutusu çiz (sadece plaka tespit edilen araca)
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                drawn_car_ids.add(car_id)

            # Plaka kutusu çiz
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            # Plaka büyütülüp ekranda gösterilecek ve üstüne yazı yazılacak
            if license_plate_crop is not None and license_plate_crop.size > 0:
                try:
                    # Plaka görüntüsünü büyüt
                    scale = 3
                    plate_h, plate_w = license_plate_crop.shape[:2]
                    resized_plate = cv2.resize(license_plate_crop, (plate_w*scale, plate_h*scale))
                    # Plakayı, orijinal plakanın hemen üstüne yerleştir
                    y_top = max(0, int(y1) - resized_plate.shape[0] - 10)
                    x_left = max(0, int(x1))
                    y_bottom = y_top + resized_plate.shape[0]
                    x_right = x_left + resized_plate.shape[1]
                    # Frame sınırlarını aşmamak için kontrol
                    if y_bottom <= frame.shape[0] and x_right <= frame.shape[1]:
                        frame[y_top:y_bottom, x_left:x_right] = resized_plate
                        # Plaka metnini büyütülmüş plakanın hemen üstüne yaz
                        if license_plate_text is not None:
                            text = license_plate_text
                            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
                            text_x = x_left + max(0, (resized_plate.shape[1] - text_width) // 2)
                            text_y = max(0, y_top - 10)
                            # Arka planı beyaz yap
                            cv2.rectangle(frame, (text_x, text_y - text_height - 10), (text_x + text_width, text_y), (255,255,255), -1)
                            cv2.putText(frame, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 6, cv2.LINE_AA)
                except Exception as e:
                    print('Plaka büyütme/gösterme hatası:', e)

            if license_plate_text is not None:
                # Plaka daha önce kaydedilmediyse plakalar.txt'ye ekle
                if license_plate_text not in detected_plates:
                    with open('plakalar.txt', 'a', encoding='utf-8') as f:
                        f.write(license_plate_text + '\n')
                    detected_plates.add(license_plate_text)
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}

        # --- Silah tespiti ve tracking ---
        gun_detections = gun_detector(frame, device=device)[0]
        gun_detections_ = []
        for gun in gun_detections.boxes.data.tolist():
            gx1, gy1, gx2, gy2, gscore, gclass_id = gun
            area = (gx2 - gx1) * (gy2 - gy1)
            if gscore >= SCORE_THRESHOLD and MIN_GUN_AREA < area < MAX_GUN_AREA:
                gun_detections_.append([gx1, gy1, gx2, gy2, gscore])
                # CSV için kaydet
                guns_results.append({
                    'frame_nmr': frame_nmr,
                    'bbox': [gx1, gy1, gx2, gy2],
                    'score': gscore,
                    'class_id': gclass_id
                })
                
        # Silah kutularını track et
        if len(gun_detections_) > 0:
            gun_tracks = mot_gun_tracker.update(np.asarray(gun_detections_))
        else:
            gun_tracks = mot_gun_tracker.update(np.empty((0, 5)))
            
        # Her track için ID ve sayım güncelle
        for track in gun_tracks:
            x1, y1, x2, y2, gun_id = track
            # En yakın tespit ile eşleştir (skor ve class_id almak için)
            best_score = 0
            best_class = 0
            for gun in gun_detections.boxes.data.tolist():
                gx1, gy1, gx2, gy2, gscore, gclass_id = gun
                # IoU ile eşleşme
                iou_val = iou([x1, y1, x2, y2], [gx1, gy1, gx2, gy2])
                if iou_val > 0.5 and gscore > best_score:
                    best_score = gscore
                    best_class = gclass_id
                    
            gun_id = int(gun_id)
            if gun_id not in tracked_guns:
                tracked_guns[gun_id] = {'count': 1, 'last_frame': frame_nmr, 'bbox': [x1, y1, x2, y2], 'score': best_score, 'class_id': best_class}
            else:
                tracked_guns[gun_id]['count'] += 1
                tracked_guns[gun_id]['last_frame'] = frame_nmr
                tracked_guns[gun_id]['bbox'] = [x1, y1, x2, y2]
                tracked_guns[gun_id]['score'] = best_score
                tracked_guns[gun_id]['class_id'] = best_class
                
        # Ekranda ve CSV'de sadece yeterince takip edilenleri göster
        for gun_id, gun_info in tracked_guns.items():
            if gun_info['count'] >= GUN_TRACK_MIN_FRAMES and gun_info['last_frame'] == frame_nmr:
                gx1, gy1, gx2, gy2 = gun_info['bbox']
                cv2.rectangle(frame, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 0, 255), 3)
                cv2.putText(frame, f'Gun: {gun_info["score"]:.2f}', (int(gx1), int(gy1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Frame'i GStreamer pipeline'a gönder
        if out.isOpened():
            out.write(frame)
        
        # Lokal görüntü için küçük bir preview (isteğe bağlı)
        # frame_display = cv2.resize(frame, (640, 360))
        # cv2.imshow('Preview', frame_display)
        
        # 'q' ile çıkış
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Çıkış yapılıyor...")
            break
            
        # Her 100 frame'de bir durum bilgisi yazdır
        if frame_nmr % 100 == 0:
            print(f"Frame {frame_nmr} işlendi, streaming devam ediyor...")

except KeyboardInterrupt:
    print("\nKeyboard interrupt ile durduruldu")
except Exception as e:
    print(f"Hata oluştu: {e}")
finally:
    # Temizlik
    print("Kaynaklar temizleniyor...")
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    
    # Sonuçları kaydet
    write_csv(results, './test.csv')

    # Silah tespitlerini CSV'ye yaz
    import csv
    with open('guns.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_nmr', 'gun_id', 'bbox', 'score', 'class_id'])
        writer.writeheader()
        for gun_id, gun_info in tracked_guns.items():
            if gun_info['count'] >= GUN_TRACK_MIN_FRAMES:
                bbox = [float(x) for x in gun_info['bbox']]
                writer.writerow({'frame_nmr': gun_info['last_frame'], 'gun_id': gun_id, 'bbox': str(bbox), 'score': gun_info['score'], 'class_id': gun_info['class_id']})
    
    print("Program sonlandırıldı.")