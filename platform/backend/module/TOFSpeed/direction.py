from queue import Queue
import threading
import cv2
import numpy as np
import math
import base64
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
from model.model_registry import ModelMeta


class TOFSpeedProcessor:
    def __init__(self):
        self.loadedModel = None
        self.modelMeta = None
        self.cap = None
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.inference_thread = None
        
        # Vehicle tracking variables
        self.vehicle_tracks = {}
        self.current_vehicle_state = {}
        self.recent_vehicle_buffer = {}
        self.frame_count = 0
        
        # Configuration parameters
        self.MAX_TRACK_LENGTH = 90
        self.SHORT_TRACKING_FRAMES = 10
        self.MIN_MOVEMENT_DISPLACEMENT = 3
        self.DIRECTION_WINDOW_SIZE = 20
        self.MIN_TURN_ANGLE = 5
        self.MIN_TRACKING_FRAMES = 90
        self.MAX_STATE_BUFFER_TIME = 30  # frames
        self.MAX_REAPPEAR_DISTANCE = 80
        
        self.fps = 30
        
        # Speed tracking variables
        self.vehicle_speeds = {}  # track_id -> speed (km/h)
        self.vehicle_last_time = {}  # track_id -> last update time
        self.speed_limit = 60  # km/h default
        self.pixels_per_meter = 20  # Conversion factor: 20 pixels = 1 meter (adjustable)
        
        # Speeding screenshots
        self.speeding_screenshots = []  # List of last 10 speeding incidents
        self.screenshot_dir = Path("speeding_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        self.MAX_SCREENSHOTS = 10
        
    def resolve_video_path(self, upload_dir: str, video_id: str) -> str:
        p = Path(upload_dir) / video_id
        if not p.exists():
            raise FileNotFoundError(f"video not found: {video_id}")
        return str(p)
    
    def load_model(self, modelTarget='yolo11s', custom_weights=None):
        try:
            if custom_weights and custom_weights.strip():
                self.loadedModel = YOLO(custom_weights)
            else:
                self.loadedModel = YOLO(modelTarget)
            print("TOF Speed Model loaded")
            return True
        except Exception as e:
            print("Error loading TOF model:", e)
            return False
    
    def open_source(self, source_type, source, upload_dir):
        try:
            if self.cap:
                self.cap.release()
            
            if source_type == "rtsp":
                self.cap = cv2.VideoCapture(source)
            elif source_type == "file":
                video_path = self.resolve_video_path(upload_dir, source)
                self.cap = cv2.VideoCapture(video_path)
            elif source_type == "url":
                self.cap = cv2.VideoCapture(source)
            
            if self.cap and self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 30
                print(f"Video opened successfully. FPS: {self.fps}")
                return True
            else:
                print("Failed to open video source")
                return False
        except Exception as e:
            print("Error opening source:", e)
            return False
    
    def check_direction_state(self, track_id, track_points):
        """Determine vehicle direction state"""
        current_track_length = len(track_points)
        
        # Stage 1: Insufficient data
        if current_track_length < self.SHORT_TRACKING_FRAMES:
            return 'PENDING'
        
        # Stage 2: Check if stopped
        short_vec = np.array(track_points[-1]) - np.array(track_points[current_track_length - self.SHORT_TRACKING_FRAMES])
        short_disp = np.linalg.norm(short_vec)
        
        if short_disp < self.MIN_MOVEMENT_DISPLACEMENT * 1.5:
            return 'STOPPED'
        
        # Stage 3: Length-based turn detection restriction
        if current_track_length < self.MIN_TRACKING_FRAMES:
            return 'STRAIGHT'
        
        # Stage 4: Turn detection with sufficient data
        start_idx = current_track_length - self.MIN_TRACKING_FRAMES
        mid_idx = current_track_length - self.DIRECTION_WINDOW_SIZE
        
        vector_a = np.array(track_points[mid_idx]) - np.array(track_points[start_idx])
        vector_b = np.array(track_points[-1]) - np.array(track_points[mid_idx])
        
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        if norm_a == 0 or norm_b == 0:
            return 'STRAIGHT'
        
        cos_angle = np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        cross_product_z = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
        
        if angle_deg < self.MIN_TURN_ANGLE:
            return 'STRAIGHT'
        else:
            return 'RIGHT_TURN' if cross_product_z > 0 else 'LEFT_TURN'
    
    def calculate_speed(self, track_id, current_pos):
        """Calculate speed in km/h based on pixel displacement"""
        current_time = time.time()
        
        if track_id not in self.vehicle_last_time:
            self.vehicle_last_time[track_id] = current_time
            self.vehicle_speeds[track_id] = 0
            return 0
        
        time_diff = current_time - self.vehicle_last_time[track_id]
        
        if time_diff < 0.1:  # Update every 0.1 seconds minimum
            return self.vehicle_speeds.get(track_id, 0)
        
        if track_id in self.vehicle_tracks and len(self.vehicle_tracks[track_id]) > 1:
            # Get position from 0.5 seconds ago (or available frames)
            frames_back = min(int(self.fps * 0.5), len(self.vehicle_tracks[track_id]) - 1)
            if frames_back > 0:
                old_pos = self.vehicle_tracks[track_id][-frames_back]
                pixel_distance = np.linalg.norm(np.array(current_pos) - np.array(old_pos))
                
                # Convert to meters
                meters = pixel_distance / self.pixels_per_meter
                
                # Calculate speed (m/s to km/h)
                time_span = frames_back / self.fps
                speed_ms = meters / time_span if time_span > 0 else 0
                speed_kmh = speed_ms * 3.6
                
                self.vehicle_speeds[track_id] = speed_kmh
                self.vehicle_last_time[track_id] = current_time
                
                return speed_kmh
        
        return self.vehicle_speeds.get(track_id, 0)
    
    def save_speeding_screenshot(self, frame, bbox, track_id, speed, state):
        """Save screenshot of speeding vehicle"""
        x1, y1, x2, y2 = bbox
        
        # Expand bbox slightly for better view
        margin = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Crop vehicle image
        vehicle_img = frame[y1:y2, x1:x2].copy()
        
        # Add info text
        info_text = f"ID:{track_id} | Speed:{speed:.1f}km/h | State:{state}"
        cv2.putText(vehicle_img, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speeding_{track_id}_{timestamp}.jpg"
        filepath = self.screenshot_dir / filename
        cv2.imwrite(str(filepath), vehicle_img)
        
        # Encode for transmission
        _, buffer = cv2.imencode('.jpg', vehicle_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Add to list
        screenshot_data = {
            "id": track_id,
            "speed": round(speed, 1),
            "state": state,
            "timestamp": timestamp,
            "image": img_base64,
            "filename": filename
        }
        
        self.speeding_screenshots.append(screenshot_data)
        
        # Keep only last 10
        if len(self.speeding_screenshots) > self.MAX_SCREENSHOTS:
            # Delete old file
            old_file = self.screenshot_dir / self.speeding_screenshots[0]["filename"]
            if old_file.exists():
                old_file.unlink()
            self.speeding_screenshots.pop(0)
        
        return screenshot_data
    
    def set_speed_limit(self, limit):
        """Set speed limit in km/h"""
        self.speed_limit = limit
        print(f"Speed limit set to {limit} km/h")
    
    def draw_bbox_and_path(self, frame, bbox, track_id, state, color_map, speed=0):
        """Draw bounding box, ID, speed, and tracking path"""
        """Draw bounding box, ID, and tracking path"""
        x1, y1, x2, y2 = bbox
        state_display = state.replace('_TURN', '')
        color = color_map.get(state, (150, 150, 150))
        
        # Change color to red if speeding
        if speed > self.speed_limit:
            color = (0, 0, 255)  # Red for speeding
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} ({state_display}) {speed:.1f}km/h"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw path
        if track_id in self.vehicle_tracks:
            points = self.vehicle_tracks[track_id]
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
    
    def inference_worker(self):
        """Main inference loop"""
        color_map = {
            'STRAIGHT': (0, 255, 0),
            'LEFT_TURN': (255, 0, 0),
            'RIGHT_TURN': (0, 165, 255),
            'STOPPED': (128, 0, 128),
            'PENDING': (150, 150, 150)
        }
        
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.frame_count += 1
                
                # Run tracking
                results = self.loadedModel.track(
                    frame,
                    persist=True,
                    conf=self.modelMeta.conf if self.modelMeta else 0.4,
                    tracker="bytetrack.yaml",
                    imgsz=1024
                )
                
                annotated_frame = frame.copy()
                current_track_ids = set()
                new_speeding_screenshots = []
                
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    
                    for bbox, track_id in zip(bboxes, track_ids):
                        current_track_ids.add(track_id)
                        x1, y1, x2, y2 = bbox
                        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Initialize or reappear track
                        if track_id not in self.vehicle_tracks:
                            if track_id in self.recent_vehicle_buffer:
                                last_pos, last_frame = self.recent_vehicle_buffer[track_id]
                                distance = np.linalg.norm(np.array(current_center) - np.array(last_pos))
                                if distance < self.MAX_REAPPEAR_DISTANCE:
                                    pass  # Continue with existing state
                            self.vehicle_tracks[track_id] = []
                        
                        self.vehicle_tracks[track_id].append(current_center)
                        if len(self.vehicle_tracks[track_id]) > self.MAX_TRACK_LENGTH:
                            self.vehicle_tracks[track_id].pop(0)
                        
                        # Calculate speed
                        speed = self.calculate_speed(track_id, current_center)
                        
                        # Update state
                        state_type = self.check_direction_state(track_id, self.vehicle_tracks[track_id])
                        if state_type and state_type != 'PENDING':
                            self.current_vehicle_state[track_id] = state_type
                        
                        state = self.current_vehicle_state.get(track_id, 'PENDING')
                        
                        # Check for speeding and capture screenshot
                        if speed > self.speed_limit and len(self.vehicle_tracks[track_id]) > self.SHORT_TRACKING_FRAMES:
                            # Only capture if we haven't captured this vehicle recently
                            if track_id not in [s["id"] for s in self.speeding_screenshots[-3:]]:
                                screenshot = self.save_speeding_screenshot(frame, bbox, track_id, speed, state)
                                new_speeding_screenshots.append(screenshot)
                        
                        # Draw bbox and path with speed
                        self.draw_bbox_and_path(annotated_frame, bbox, track_id, state, color_map, speed)
                
                # Handle disappeared vehicles
                for tid in list(self.vehicle_tracks.keys()):
                    if tid not in current_track_ids:
                        last_pos = self.vehicle_tracks[tid][-1] if self.vehicle_tracks[tid] else (0, 0)
                        self.recent_vehicle_buffer[tid] = (last_pos, self.frame_count)
                        del self.vehicle_tracks[tid]
                        if tid in self.current_vehicle_state:
                            del self.current_vehicle_state[tid]
                
                # Clean old buffer
                self.recent_vehicle_buffer = {
                    k: v for k, v in self.recent_vehicle_buffer.items()
                    if self.frame_count - v[1] < self.MAX_STATE_BUFFER_TIME
                }
                
                # Count vehicles by state
                state_counts = defaultdict(int)
                for state in self.current_vehicle_state.values():
                    state_counts[state] += 1
                
                # Draw counts on frame
                y_pos = 30
                for s_name, s_color in color_map.items():
                    count = state_counts.get(s_name, 0)
                    cv2.putText(annotated_frame, f"{s_name}: {count}", (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)
                    y_pos += 30
                
                # Draw speed limit
                cv2.putText(annotated_frame, f"Speed Limit: {self.speed_limit} km/h", (10, y_pos + 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                result = {
                    "frame": frame_base64,
                    "state_counts": dict(state_counts),
                    "total_tracked": len(self.current_vehicle_state),
                    "speeding_screenshots": new_speeding_screenshots,
                    "all_screenshots": self.speeding_screenshots[-10:]
                }
                
                if not self.result_queue.full():
                    self.result_queue.put(result)
    
    def start_inference_thread(self):
        """Start the inference thread"""
        if self.inference_thread and self.inference_thread.is_alive():
            return
        
        self.inference_thread = threading.Thread(
            target=self.inference_worker,
            daemon=True
        )
        self.inference_thread.start()
        print("TOF Speed inference thread started")
    
    def setModelMeta(self, model_meta: ModelMeta):
        """Set model metadata"""
        self.modelMeta = model_meta
