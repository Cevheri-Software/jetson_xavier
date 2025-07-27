from ultralytics import YOLO
import cv2
import numpy as np
import threading
import queue
import time

# Load models with optimizations
car_model = YOLO("yolov8n.pt")
plate_model = YOLO("license_plate_detector.pt")

# Optimize models for inference
car_model.fuse()  # Fuse layers for faster inference
plate_model.fuse()

# Try different camera backends for better compatibility
backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
cap = None

for backend in backends_to_try:
    print(f"Trying camera with backend: {backend}")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret:
            print(f"Camera opened successfully with backend: {backend}")
            break
        else:
            cap.release()
            cap = None
    else:
        if cap:
            cap.release()
        cap = None

if cap is None or not cap.isOpened():
    print("Error: Could not open video capture with any backend")
    exit()

# Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag

# Verify actual resolution
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

# GStreamer pipeline optimized for Jetson
host_ip = "192.168.1.100"
gst_str = (
    f'appsrc ! videoconvert ! nvvidconv ! '
    f'nvv4l2h264enc bitrate=2000000 preset-level=1 ! '
    f'h264parse ! rtph264pay config-interval=1 pt=96 ! '
    f'udpsink host={host_ip} port=5000 sync=false'
)

# OpenCV VideoWriter for sending via GStreamer
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (int(actual_width), int(actual_height)))

if not out.isOpened():
    print("Error: Could not open video writer")
    exit()

# Threading setup for parallel processing
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)
detection_active = threading.Event()
detection_active.set()

def detection_worker():
    """Worker thread for running YOLO inference"""
    while detection_active.is_set():
        try:
            frame_data = frame_queue.get(timeout=0.1)
            if frame_data is None:
                continue
                
            frame, frame_id = frame_data
            
            # Resize frame for faster inference
            inference_frame = cv2.resize(frame, (416, 416))  # Smaller input size
            
            # Run car detection with optimized parameters
            car_results = car_model(inference_frame, verbose=False, conf=0.5, iou=0.4)
            
            detected_objects = []
            
            if car_results[0].boxes is not None and len(car_results[0].boxes) > 0:
                car_boxes = car_results[0].boxes.xyxy.cpu().numpy()
                
                # Scale boxes back to original frame size
                scale_x = actual_width / 416
                scale_y = actual_height / 416
                
                for box in car_boxes:
                    x1, y1, x2, y2 = box
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(int(actual_width), x1))
                    y1 = max(0, min(int(actual_height), y1))
                    x2 = max(0, min(int(actual_width), x2))
                    y2 = max(0, min(int(actual_height), y2))
                    
                    # Skip small detections
                    if x2 - x1 < 80 or y2 - y1 < 60:
                        continue
                    
                    detected_objects.append({
                        'type': 'car',
                        'bbox': (x1, y1, x2, y2),
                        'plates': []
                    })
                    
                    # Only process license plates for larger car detections
                    if x2 - x1 > 120 and y2 - y1 > 90:
                        # Crop car region from original frame
                        car_crop = frame[y1:y2, x1:x2]
                        
                        if car_crop.size > 0:
                            # Resize car crop for faster plate detection
                            car_crop_resized = cv2.resize(car_crop, (224, 224))
                            
                            # Run plate detection
                            plate_results = plate_model(car_crop_resized, verbose=False, conf=0.3)
                            
                            if plate_results[0].boxes is not None and len(plate_results[0].boxes) > 0:
                                plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy()
                                
                                # Scale plate boxes back
                                plate_scale_x = (x2 - x1) / 224
                                plate_scale_y = (y2 - y1) / 224
                                
                                for pb in plate_boxes:
                                    px1, py1, px2, py2 = pb
                                    
                                    # Convert to absolute coordinates
                                    abs_px1 = x1 + int(px1 * plate_scale_x)
                                    abs_py1 = y1 + int(py1 * plate_scale_y)
                                    abs_px2 = x1 + int(px2 * plate_scale_x)
                                    abs_py2 = y1 + int(py2 * plate_scale_y)
                                    
                                    detected_objects[-1]['plates'].append((abs_px1, abs_py1, abs_px2, abs_py2))
            
            # Send results back
            if not result_queue.full():
                result_queue.put((frame_id, detected_objects))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Detection error: {e}")
            continue

# Start detection thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

print("Streaming with optimized license plate detection... Press Ctrl+C to stop")

frame_count = 0
last_detections = []
detection_interval = 10  # Increased interval for better performance
fps_counter = 0
fps_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate and display FPS every second
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            print(f"FPS: {fps:.1f}")
            fps_counter = 0
            fps_start_time = time.time()

        # Submit frame for detection (non-blocking)
        if frame_count % detection_interval == 0:
            if not frame_queue.full():
                frame_queue.put((frame.copy(), frame_count))

        # Get latest detection results (non-blocking)
        try:
            while not result_queue.empty():
                result_frame_id, detections = result_queue.get_nowait()
                last_detections = detections
        except queue.Empty:
            pass

        # Draw detections on current frame
        display_frame = frame.copy()
        for obj in last_detections:
            if obj['type'] == 'car':
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, 'Car', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw license plates
                for plate_bbox in obj['plates']:
                    px1, py1, px2, py2 = plate_bbox
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(display_frame, 'Plate', (px1, py1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # Send frame to stream
        out.write(display_frame)

except KeyboardInterrupt:
    print("Streaming stopped")
finally:
    detection_active.clear()
    detection_thread.join(timeout=1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
