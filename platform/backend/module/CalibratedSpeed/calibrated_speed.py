import cv2
import numpy as np
import base64
from ultralytics import YOLO
from queue import Queue
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CalibratedSpeedProcessor:
    """
    Calibrated speed tracking processor using homography transformation.
    Tracks vehicles and calculates their speed based on real-world distance calibration.
    """
    
    def __init__(self):
        self.model = None
        self.cap = None
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.inference_thread = None
        
        # Calibration parameters
        self.roi_points = []  # 4 points for ROI polygon
        self.roi_polygon = None  # Pre-computed polygon for faster checks
        self.width_real_meters = 0.0
        self.depth_real_meters = 0.0
        self.homography_matrix = None
        self.fps = 30.0
        self.delta_t = 1.0 / self.fps
        
        # Tracking data
        self.object_tracks = {}  # track_id -> [x, y]
        self.object_speeds = {}  # track_id -> speed_kmh
        self.frame_counter = 0  # Frame counter for statistics
        
        # Speed statistics
        self.all_speeds = []  # All recorded speeds (max 100)
        self.max_speed = 0.0
        self.avg_speed = 0.0
        
        # Performance optimization
        self.frame_skip = 0  # Process every frame by default
        self.current_frame_skip = 0  # Counter for frame skipping
        self.jpeg_quality = 70  # Lower quality for faster encoding
        self.target_width = 640  # Target frame width for faster inference
        
    def load_model(self, model_path: str) -> bool:
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def open_source(self, source_type: str, source: str, upload_dir: Path) -> bool:
        """Open video source (file, stream, or webcam)"""
        try:
            if source_type == "file":
                video_path = upload_dir / source
                self.cap = cv2.VideoCapture(str(video_path))
            elif source_type == "stream":
                self.cap = cv2.VideoCapture(source)
            elif source_type == "webcam":
                self.cap = cv2.VideoCapture(int(source))
            else:
                return False
            
            if not self.cap.isOpened():
                print(f"Failed to open source: {source}")
                return False
            
            print(f"Source opened: {source_type} - {source}")
            return True
        except Exception as e:
            print(f"Error opening source: {e}")
            return False
    
    def set_calibration(self, roi_points: List[List[int]], width_meters: float, depth_meters: float):
        """
        Set calibration parameters and calculate homography matrix.
        
        Args:
            roi_points: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                       Order: top-left, top-right, bottom-right, bottom-left
            width_meters: Real-world width of ROI in meters
            depth_meters: Real-world depth of ROI in meters
        """
        self.roi_points = roi_points
        self.width_real_meters = width_meters
        self.depth_real_meters = depth_meters
        
        # Calculate homography matrix
        src_pts = np.float32(roi_points)
        dst_pts = np.float32([
            [0, 0],
            [width_meters, 0],
            [width_meters, depth_meters],
            [0, depth_meters]
        ])
        
        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        
        # Pre-compute ROI polygon for faster is_in_roi checks
        self.roi_polygon = np.int32(roi_points).reshape((-1, 1, 2))
        
        print(f"Calibration set: {width_meters}m x {depth_meters}m")
        
        # Reset tracking data when calibration changes
        self.object_tracks = {}
        self.object_speeds = {}
        self.all_speeds = []
        self.max_speed = 0.0
        self.avg_speed = 0.0
    
    def get_group_label(self, cls_idx: int) -> str:
        """Group vehicle classes into simplified categories"""
        if self.model is None:
            return str(cls_idx)
        
        names = self.model.names
        if isinstance(names, dict):
            raw = names.get(int(cls_idx), str(cls_idx))
        else:
            raw = names[int(cls_idx)] if 0 <= int(cls_idx) < len(names) else str(cls_idx)
        
        name = str(raw).strip().lower().replace("_", " ").replace("-", " ")
        
        truck_keywords = ["truck", "axle", "trailer", "semi", "lorry"]
        if any(k in name for k in truck_keywords):
            return "Truck"
        if "bus" in name:
            return "Bus"
        if "car" in name:
            return "Car"
        if "motorcycle" in name or "moto" in name or "bike" in name:
            return "Motorcycle"
        
        return str(raw)
    
    def calculate_speed(self, track_id: int, current_x: float, current_y: float) -> Optional[float]:
        """
        Calculate speed of tracked object using homography transformation.
        
        Returns:
            Speed in km/h or None if this is the first frame for this track_id
        """
        if self.homography_matrix is None:
            return None
        
        if track_id not in self.object_tracks:
            self.object_tracks[track_id] = [current_x, current_y]
            return None
        
        prev_x, prev_y = self.object_tracks[track_id]
        self.object_tracks[track_id] = [current_x, current_y]
        
        # Transform pixel coordinates to real-world coordinates
        prev_coords_pixel = np.array([[[prev_x, prev_y]]], dtype='float32')
        prev_coords_real = cv2.perspectiveTransform(prev_coords_pixel, self.homography_matrix)[0][0]
        
        curr_coords_pixel = np.array([[[current_x, current_y]]], dtype='float32')
        curr_coords_real = cv2.perspectiveTransform(curr_coords_pixel, self.homography_matrix)[0][0]
        
        # Calculate distance in meters
        distance_real = np.sqrt(
            (curr_coords_real[0] - prev_coords_real[0]) ** 2 +
            (curr_coords_real[1] - prev_coords_real[1]) ** 2
        )
        
        # Calculate speed: distance / time
        speed_mps = distance_real / self.delta_t
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh
    
    def is_in_roi(self, x: float, y: float) -> bool:
        """Check if point is inside ROI polygon (optimized with pre-computed polygon)"""
        if self.roi_polygon is None:
            return True  # Allow all objects if ROI not set
        
        try:
            result = cv2.pointPolygonTest(self.roi_polygon, (x, y), False)
            return result >= 0
        except:
            return True  # Allow if error occurs
    
    def update_speed_stats(self, speed_kmh: float):
        """Update speed statistics"""
        self.all_speeds.append(speed_kmh)
        
        # Keep only recent speeds (last 100 for moving average)
        if len(self.all_speeds) > 100:
            self.all_speeds.pop(0)
        
        # Update max speed
        if speed_kmh > self.max_speed:
            self.max_speed = speed_kmh
        
        # Update average speed
        if self.all_speeds:
            self.avg_speed = sum(self.all_speeds) / len(self.all_speeds)
    
    def inference_loop(self):
        """Main inference loop running in separate thread (optimized)"""
        import time
        
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue
            
            # Get latest frame, skip old ones
            frame = None
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except:
                    break
            
            if frame is None:
                continue
            
            # Frame skipping for performance
            self.current_frame_skip = (self.current_frame_skip + 1) % (self.frame_skip + 1)
            if self.current_frame_skip != 0:
                continue
            
            # Draw ROI polygon if set
            if self.roi_polygon is not None:
                cv2.polylines(frame, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            
            if self.model is None:
                # Send frame with ROI drawn (no model loaded yet)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                result = {
                    "frame": frame_base64,
                    "detections": [],
                    "stats": {
                        "avg_speed": 0.0,
                        "max_speed": 0.0,
                        "active_tracks": 0
                    }
                }
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(result)
                continue
            
            # Run YOLO tracking with optimized settings
            results = self.model.track(
                frame,
                persist=True,
                conf=0.3,
                iou=0.45,
                verbose=False
            )
            
            # Process detections
            detections = []
            
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) >> 1  # Faster than // 2
                    bottom_y = y2
                    
                    # Only process objects in ROI
                    if not self.is_in_roi(center_x, bottom_y):
                        continue
                    
                    # Calculate speed
                    speed_kmh = self.calculate_speed(track_id, center_x, bottom_y)
                    
                    # Update cached speed for display consistency
                    if speed_kmh is not None:
                        self.object_speeds[track_id] = speed_kmh
                        self.update_speed_stats(speed_kmh)
                    
                    # Get display speed (current or cached)
                    display_speed = self.object_speeds.get(track_id, None)
                    
                    # Get simplified class name
                    class_name = self.get_group_label(cls_idx)
                    
                    detections.append({
                        "track_id": int(track_id),
                        "class_name": class_name,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int(center_x), int(bottom_y)],
                        "speed_kmh": float(display_speed) if display_speed is not None else None
                    })
            # Draw detections on frame
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                label = f'{det["class_name"]} ID {det["track_id"]}'
                if det["speed_kmh"] is not None:
                    label += f': {det["speed_kmh"]:.1f} km/h'
                
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Put result in queue (keep only latest)
            result = {
                "frame": frame_base64,
                "detections": detections,
                "stats": {
                    "avg_speed": round(self.avg_speed, 2),
                    "max_speed": round(self.max_speed, 2),
                    "active_tracks": len(self.object_speeds)
                }
            }
            
            # Clear old results and put latest
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break
            self.result_queue.put(result)
    
    def start_inference_thread(self):
        """Start inference thread"""
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
            self.inference_thread.start()
            print("Inference thread started")
    
    def stop(self):
        """Stop processing and release resources"""
        self.running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        print("CalibratedSpeedProcessor stopped")
