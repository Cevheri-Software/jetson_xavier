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
        self.skip_frames = 1  # Process license plates more frequently
        self.frame_count = 0
        
        # Detection classes (COCO)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load model
        self.load_model(model_path)
        
        # Initialize tracker (removed gun tracker)
        self.vehicle_tracker = OptimizedTracker(max_disappeared=15)
        
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
                    
                    # Filter detections - only vehicles now
                    if cls in self.vehicle_classes:
                        # Skip small detections like working code
                        if x2 - x1 < 80 or y2 - y1 < 60:
                            continue
                        vehicles.append((x1, y1, x2, y2, conf, cls))
            
            return vehicles
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def enhance_license_plate_image(self, img):
        """Enhanced preprocessing for license plate images"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize if too small
        h, w = gray.shape
        if h < 40 or w < 100:
            scale_factor = max(40/h, 100/w)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Try multiple threshold methods and return the best one
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        return methods
    
    def extract_license_plate_text(self, plate_crop):
        """Improved license plate text extraction"""
        if self.ocr_reader is None or plate_crop.size == 0:
            return None, 0
        
        try:
            # Get multiple preprocessed versions
            processed_images = self.enhance_license_plate_image(plate_crop)
            
            best_text = None
            best_confidence = 0
            
            # Try OCR on each preprocessed version
            for processed_img in processed_images:
                try:
                    # OCR with different configurations
                    results = self.ocr_reader.readtext(processed_img, 
                                                     detail=1,
                                                     paragraph=False,
                                                     width_ths=0.7,
                                                     height_ths=0.7)
                    
                    if results:
                        # Process each detection
                        for result in results:
                            text = result[1].upper().strip()
                            confidence = result[2]
                            
                            # Clean text (remove non-alphanumeric except spaces and dashes)
                            text = re.sub(r'[^A-Z0-9\s\-]', '', text)
                            text = re.sub(r'\s+', '', text)  # Remove all spaces
                            
                            # License plate validation
                            if self.is_valid_license_plate(text) and confidence > best_confidence:
                                best_text = text
                                best_confidence = confidence
                
                except Exception as e:
                    continue
            
            # Final validation
            if best_text and best_confidence > 0.3:  # Lower threshold for better detection
                return best_text, best_confidence
            
            return None, 0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None, 0
    
    def is_valid_license_plate(self, text):
        """Validate if text looks like a license plate"""
        if not text or len(text) < 4:
            return False
        
        # Remove common OCR errors
        text = text.replace('O', '0').replace('I', '1').replace('S', '5')
        
        # Check for reasonable license plate patterns
        # Allow 4-8 characters, mix of letters and numbers
        if 4 <= len(text) <= 8:
            has_letter = any(c.isalpha() for c in text)
            has_number = any(c.isdigit() for c in text)
            
            # Must have both letters and numbers, or be all numbers (some regions)
            if has_letter and has_number:
                return True
            elif text.isdigit() and len(text) >= 5:  # All numeric plates
                return True
        
        return False
    
    def process_frame(self, frame):
        """Main frame processing function"""
        self.frame_count += 1
        
        # Preprocess
        frame = self.preprocess_frame(frame)
        
        # Detection (removed guns)
        vehicles = self.detect_objects(frame)
        
        # Vehicle tracking
        vehicle_rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cls in vehicles]
        tracked_vehicles = self.vehicle_tracker.update(vehicle_rects)
        
        # Draw detections (removed guns)
        self.draw_detections(frame, tracked_vehicles, vehicles)
        
        # Process license plates more frequently
        if self.frame_count % self.skip_frames == 0:
            self.process_license_plates(frame, vehicles, tracked_vehicles)
        
        return frame
    
    def process_license_plates(self, frame, vehicles, tracked_vehicles):
        """Enhanced license plate processing within detected vehicles"""
        for x1, y1, x2, y2, conf, cls in vehicles:
            # Focus on front and rear areas of vehicles where plates are typically located
            vehicle_h = y2 - y1
            vehicle_w = x2 - x1
            
            # Define search areas (front and rear of vehicle)
            search_areas = [
                # Front area (bottom 25% of vehicle)
                {
                    'x1': max(0, x1 - 10),
                    'y1': max(0, y1 + int(vehicle_h * 0.75)),
                    'x2': min(frame.shape[1], x2 + 10),
                    'y2': min(frame.shape[0], y2 + 20)
                },
                # Rear area (top 25% of vehicle)
                {
                    'x1': max(0, x1 - 10),
                    'y1': max(0, y1 - 20),
                    'x2': min(frame.shape[1], x2 + 10),
                    'y2': min(frame.shape[0], y1 + int(vehicle_h * 0.25))
                }
            ]
            
            for search_area in search_areas:
                # Extract search region
                search_crop = frame[search_area['y1']:search_area['y2'], 
                                 search_area['x1']:search_area['x2']]
                
                if search_crop.size > 0:
                    self.detect_license_plate_regions(frame, search_crop, 
                                                    search_area['x1'], search_area['y1'])
    
    def detect_license_plate_regions(self, frame, vehicle_crop, offset_x, offset_y):
        """Enhanced license plate region detection"""
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
            # Multiple edge detection approaches
            # Canny edge detection
            edges1 = cv2.Canny(gray, 30, 100, apertureSize=3)
            
            # Morphological operations to connect text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:10]:  # Check top 10 largest contours
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enhanced filtering for license plate characteristics
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # License plate criteria (more permissive)
                if (1.5 < aspect_ratio < 8.0 and  # Wider range for different plate types
                    500 < area < 20000 and        # Larger area range
                    w > 50 and h > 12 and         # Minimum size
                    w < vehicle_crop.shape[1] * 0.8 and  # Not too wide relative to search area
                    h < vehicle_crop.shape[0] * 0.6):    # Not too tall relative to search area
                    
                    # Extract potential license plate
                    plate_crop = vehicle_crop[y:y+h, x:x+w]
                    
                    if plate_crop.size > 0:
                        # Try to read text
                        text, confidence = self.extract_license_plate_text(plate_crop)
                        
                        if text and text not in self.detected_plates and confidence > 0.3:
                            self.detected_plates.add(text)
                            
                            # Draw on frame
                            abs_x1 = offset_x + x
                            abs_y1 = offset_y + y
                            abs_x2 = abs_x1 + w
                            abs_y2 = abs_y1 + h
                            
                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                            cv2.putText(frame, f'{text} ({confidence:.2f})', (abs_x1, abs_y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # Log detection
                            logger.info(f"License plate detected: {text} (confidence: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"License plate detection error: {e}")
    
    def draw_detections(self, frame, tracked_vehicles, vehicles):
        """Draw detection results on frame (removed gun functionality)"""
        # Draw tracked vehicles
        for track_id, (x1, y1, x2, y2) in tracked_vehicles.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Vehicle {track_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add info overlay (removed gun count)
        cv2.putText(frame, f'Vehicles: {len(tracked_vehicles)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'License Plates: {len(self.detected_plates)}', 
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
        logger.info("🚗 VEHICLE & LICENSE PLATE DETECTION SYSTEM STARTED")
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
                                  f"License Plates: {len(self.detected_plates)}")
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
    
    parser = argparse.ArgumentParser(description='Vehicle & License Plate Detection System')
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