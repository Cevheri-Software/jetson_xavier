
"""
Optimized Real-Time Detection System for Jetson Xavier NX
Designed for drone applications with streaming to control laptop
Focuses on performance and reliability
"""

from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time
import threading
import queue
import re
import logging
from collections import defaultdict, deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedTracker:
    """Lightweight object tracker optimized for Jetson"""
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > 50:  # Distance threshold
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])

        # Return tracking results
        tracking_results = {}
        for object_id, centroid in self.objects.items():
            # Find the corresponding rectangle
            for i, (x1, y1, x2, y2) in enumerate(rects):
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                if abs(cx - centroid[0]) < 5 and abs(cy - centroid[1]) < 5:
                    tracking_results[object_id] = (x1, y1, x2, y2)
                    break
        
        return tracking_results

class DroneDetectionSystem:
    def __init__(self, 
                 source=0,
                 output_ip="192.168.1.100", 
                 output_port=5000,
                 model_path='yolov8n.pt',
                 input_size=640,
                 confidence_threshold=0.6):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Performance settings
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.skip_frames = 2  # Process every 3rd frame for license plates
        self.frame_count = 0
        
        # Detection classes (COCO)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load model
        self.load_model(model_path)
        
        # Initialize trackers
        self.vehicle_tracker = OptimizedTracker(max_disappeared=15)
        self.gun_tracker = OptimizedTracker(max_disappeared=5)
        
        # Results storage (in-memory only)
        self.detected_plates = set()
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Setup video input/output - get actual resolution first
        self.setup_input(source)
        self.setup_output(output_ip, output_port)
        
        # OCR setup (simplified)
        self.setup_ocr()
    
    def load_model(self, model_path):
        """Load optimized YOLO model"""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Optimize model like working code
            self.model.fuse()  # Fuse layers for faster inference
            
            # Additional optimizations for inference
            if self.device == 'cuda':
                self.model.model.half()  # Use FP16 for speed
                torch.backends.cudnn.benchmark = True
            
            logger.info(f"Model loaded and optimized: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_input(self, source):
        """Setup video input with optimization for Jetson"""
        logger.info(f"Setting up video input: {source}")
        
        if isinstance(source, str) and source.startswith('rtsp://'):
            # Optimized RTSP pipeline for Jetson
            pipeline = (
                f"rtspsrc location={source} latency=0 drop-on-latency=true ! "
                "rtph264depay ! h264parse ! nvv4l2decoder ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # Try different camera backends like your working code
            backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
            self.cap = None
            
            for backend in backends_to_try:
                logger.info(f"Trying camera with backend: {backend}")
                self.cap = cv2.VideoCapture(source, backend)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        logger.info(f"Camera opened successfully with backend: {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
            
            if self.cap is None or not self.cap.isOpened():
                raise Exception(f"Could not open video source with any backend: {source}")
            
            # Optimize camera settings for Jetson - use smaller resolution like working code
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {self.actual_width}x{self.actual_height}")
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video source: {source}")
        
        logger.info("Video input ready")
    
    def setup_output(self, ip, port):
        """Setup GStreamer output using the exact working pipeline"""
        logger.info(f"Setting up stream output: {ip}:{port}")
        
        # Use the exact same pipeline that works in your inf.py
        gst_str = (
            f'appsrc ! videoconvert ! nvvidconv ! '
            f'nvv4l2h264enc bitrate=2000000 preset-level=1 ! '
            f'h264parse ! rtph264pay config-interval=1 pt=96 ! '
            f'udpsink host={ip} port={port} sync=false'
        )
        
        try:
            # Use actual camera resolution from setup_input
            # Modified this line: Removed the '0' for fourcc to match working inf.py behavior
            self.out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 30, (self.actual_width, self.actual_height))
            
            if self.out.isOpened():
                logger.info("✅ Video output stream ready with hardware encoding")
            else:
                logger.error("❌ Hardware encoding failed")
                # Try software fallback
                gst_str_fallback = (
                    f'appsrc ! videoconvert ! '
                    f'x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! '
                    f'rtph264pay ! '
                    f'udpsink host={ip} port={port} sync=false'
                )
                # Modified this line: Removed the '0' for fourcc to match working inf.py behavior
                self.out = cv2.VideoWriter(gst_str_fallback, cv2.CAP_GSTREAMER, 30, (self.actual_width, self.actual_height))
                
                if self.out.isOpened():
                    logger.info("✅ Video output stream ready with software encoding")
                else:
                    logger.error("❌ All encoding methods failed")
                    self.out = None
                    
        except Exception as e:
            logger.error(f"Stream output setup failed: {e}")
            self.out = None
    
    def setup_ocr(self):
        """Setup lightweight OCR for license plates"""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("OCR initialized")
        except ImportError:
            logger.warning("EasyOCR not available, license plate text won't be extracted")
            self.ocr_reader = None
        except Exception as e:
            logger.error(f"OCR setup failed: {e}")
            self.ocr_reader = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        # Keep original resolution from camera - don't resize like the original code
        return frame
    
    def detect_objects(self, frame):
        """Run YOLO detection on frame with optimizations like working code"""
        try:
            # Resize frame for faster inference like in working code
            inference_frame = cv2.resize(frame, (416, 416))
            
            # Run inference with optimized parameters
            results = self.model(inference_frame, 
                               imgsz=416,  # Smaller input size like working code
                               conf=self.confidence_threshold,
                               iou=0.4,  # Add IOU threshold like working code
                               device=self.device,
                               verbose=False)[0]
            
            vehicles = []
            guns = []
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Scale boxes back to original frame size
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = int(classes[i])
                    
                    # Scale coordinates back
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Ensure coordinates are within bounds
                    x1 = max(0, min(frame.shape[1], x1))
                    y1 = max(0, min(frame.shape[0], y1))
                    x2 = max(0, min(frame.shape[1], x2))
                    y2 = max(0, min(frame.shape[0], y2))
                    
                    # Filter detections
                    if cls in self.vehicle_classes:
                        # Skip small detections like working code
                        if x2 - x1 < 80 or y2 - y1 < 60:
                            continue
                        vehicles.append((x1, y1, x2, y2, conf, cls))
                    elif cls == 0:  # Assuming gun class is 0
                        # Add area filtering for guns
                        area = (x2 - x1) * (y2 - y1)
                        if 1000 < area < 50000:
                            guns.append((x1, y1, x2, y2, conf))
            
            return vehicles, guns, []
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], [], []
    
    def extract_license_plate_text(self, plate_crop):
        """Extract text from license plate crop"""
        if self.ocr_reader is None or plate_crop.size == 0:
            return None, 0
        
        try:
            # Preprocess the crop
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            results = self.ocr_reader.readtext(thresh, detail=1)
            
            if results:
                # Get best result
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].upper()
                confidence = best_result[2]
                
                # Clean text (remove non-alphanumeric)
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                if len(text) >= 5 and confidence > 0.4:
                    return text, confidence
            
            return None, 0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None, 0
    
    def process_frame(self, frame):
        """Main frame processing function"""
        self.frame_count += 1
        
        # Preprocess
        frame = self.preprocess_frame(frame)
        
        # Detection
        vehicles, guns, license_plates = self.detect_objects(frame)
        
        # Vehicle tracking
        vehicle_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in vehicles]
        tracked_vehicles = self.vehicle_tracker.update(vehicle_rects)
        
        # Gun tracking
        gun_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf in guns]
        tracked_guns = self.gun_tracker.update(gun_rects)
        
        # Draw detections
        self.draw_detections(frame, tracked_vehicles, tracked_guns, vehicles, guns)
        
        # Process license plates (every few frames to save computation)
        if self.frame_count % self.skip_frames == 0:
            self.process_license_plates(frame, vehicles, tracked_vehicles)
        
        return frame
    
    def process_license_plates(self, frame, vehicles, tracked_vehicles):
        """Process license plates within detected vehicles"""
        for x1, y1, x2, y2, conf, cls in vehicles:
            # Expand search area slightly
            margin = 20
            search_x1 = max(0, x1 - margin)
            search_y1 = max(0, y1 - margin)
            search_x2 = min(frame.shape[1], x2 + margin)
            search_y2 = min(frame.shape[0], y2 + margin)
            
            # Extract vehicle region
            vehicle_crop = frame[search_y1:search_y2, search_x1:search_x2]
            
            if vehicle_crop.size > 0:
                # Look for rectangular regions that could be license plates
                self.detect_license_plate_regions(frame, vehicle_crop, search_x1, search_y1)
    
    def detect_license_plate_regions(self, frame, vehicle_crop, offset_x, offset_y):
        """Detect license plate regions using computer vision"""
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and size (typical license plate characteristics)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                if (2.0 < aspect_ratio < 6.0 and 
                    1000 < area < 15000 and
                    w > 60 and h > 15):
                    
                    # Extract potential license plate
                    plate_crop = vehicle_crop[y:y+h, x:x+w]
                    
                    if plate_crop.size > 0:
                        # Try to read text
                        text, confidence = self.extract_license_plate_text(plate_crop)
                        
                        if text and text not in self.detected_plates:
                            self.detected_plates.add(text)
                            
                            # Draw on frame
                            abs_x1 = offset_x + x
                            abs_y1 = offset_y + y
                            abs_x2 = abs_x1 + w
                            abs_y2 = abs_y1 + h
                            
                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                            cv2.putText(frame, text, (abs_x1, abs_y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            
                            # Log detection only
                            logger.info(f"License plate detected: {text} (confidence: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
    
    def draw_detections(self, frame, tracked_vehicles, tracked_guns, vehicles, guns):
        """Draw detection results on frame"""
        # Draw tracked vehicles
        for track_id, (x1, y1, x2, y2) in tracked_vehicles.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Vehicle {track_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw tracked guns
        for track_id, (x1, y1, x2, y2) in tracked_guns.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, 'GUN DETECTED!', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            logger.warning(f"Gun detected! Track ID: {track_id}")
        
        # Add info overlay
        cv2.putText(frame, f'Vehicles: {len(tracked_vehicles)} | Guns: {len(tracked_guns)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Plates: {len(self.detected_plates)}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def capture_thread(self):
        """Thread for capturing frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Skip frame if queue is full
                    pass
            else:
                logger.error("Failed to read frame")
                break
    
    def process_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put(processed_frame)
                
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 60)
        logger.info("🚀 DRONE DETECTION SYSTEM STARTED")
        logger.info("=" * 60)
        
        self.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        process_thread = threading.Thread(target=self.process_thread)
        
        capture_thread.start()
        process_thread.start()
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = time.time()
        
        try:
            while self.running:
                try:
                    # Get processed frame
                    processed_frame = self.result_queue.get(timeout=1)
                    frame_count += 1
                    
                    # Send to output stream
                    if self.out is not None:
                        self.out.write(processed_frame)
                    
                    # Calculate and display FPS
                    current_time = time.time()
                    if current_time - last_fps_time >= 5.0:
                        fps = frame_count / (current_time - start_time)
                        logger.info(f"FPS: {fps:.1f} | Frames: {frame_count} | "
                                  f"Vehicles: {len(self.vehicle_tracker.objects)} | "
                                  f"Plates: {len(self.detected_plates)}")
                        last_fps_time = current_time
                    
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    logger.info("Stopping system...")
                    break
        
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.running = False
            capture_thread.join()
            process_thread.join()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
        
        logger.info("System cleanup complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Drone Detection System')
    parser.add_argument('--source', default=0, help='Video source (camera ID or RTSP URL)')
    parser.add_argument('--output-ip', default='192.168.1.100', help='Output stream IP')
    parser.add_argument('--output-port', type=int, default=5000, help='Output stream port')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold')
    
    args = parser.parse_args()
    
    try:
        # Create and run detection system
        system = DroneDetectionSystem(
            source=args.source,
            output_ip=args.output_ip,
            output_port=args.output_port,
            model_path=args.model,
            input_size=args.input_size,
            confidence_threshold=args.confidence
        )
        system.run()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        logger.info("Make sure all dependencies are installed and camera is connected")

