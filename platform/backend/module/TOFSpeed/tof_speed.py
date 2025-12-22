import cv2
import numpy as np
import base64
import time
import threading
import json
import csv
from datetime import datetime
from queue import Queue
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
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
        
        # Speed measurement settings
        self.speed_limit = 50.0  # km/h
        self.line_upper = 219
        self.line_lower = 300
        self.line_tolerance = 3
        self.pixel_distance = abs(self.line_lower - self.line_upper)
        
        # Directional distances (meters)
        self.dist_upward_m = 23.0
        self.dist_downward_m = 22.0
        
        # Pixels per meter
        self.ppm_upward = self.pixel_distance / self.dist_upward_m
        self.ppm_downward = self.pixel_distance / self.dist_downward_m
        
        # Tracker data (persistent across sessions)
        self.tracker_data = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Permanent storage setup
        self.violation_dir = Path("speed_violations")
        self.violation_dir.mkdir(exist_ok=True)
        self.violation_images_dir = self.violation_dir / "images"
        self.violation_images_dir.mkdir(exist_ok=True)
        self.violations_json = self.violation_dir / "violations.json"
        self.violations_csv = self.violation_dir / "violations.csv"
        
        # Screenshot storage (for UI + persistent)
        self.violation_screenshots = []
        self.max_screenshots = 10
        
        # Batch processing mode
        self.batch_mode = False
        self.batch_output_dir = None
        
        # Auto-cleanup settings
        self.retention_days = 30
        self.cleanup_enabled = True
        
        # Performance optimization
        self.frame_skip = 0  # 0 = process every frame, 1 = skip 1 frame, etc.
        self.optimization_enabled = False
        
        # Perspective correction
        self.perspective_correction_enabled = False
        self.perspective_matrix = None
        
        # Load existing violations from disk (after all settings are initialized)
        self.load_violations_from_disk()
        
    def resolve_video_path(self, upload_dir: Path, video_id: str) -> str:
        p = upload_dir / video_id
        if not p.exists():
            raise FileNotFoundError(f"video not found: {video_id}")
        return str(p)
    
    def load_model(self, modelTarget='yolo11s', custom_weights=None):
        try:
            if custom_weights:
                self.loadedModel = YOLO(custom_weights)
                print(f"Custom model loaded: {custom_weights}")
            else:
                self.loadedModel = YOLO(modelTarget)
                print(f"Model loaded: {modelTarget}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def open_source(self, source_type, source, upload_dir):
        try:
            if source_type == 'rtsp':
                self.cap = cv2.VideoCapture(source)
            elif source_type == 'file':
                video_path = self.resolve_video_path(upload_dir, source)
                self.cap = cv2.VideoCapture(video_path)
            elif source_type == 'http':
                self.cap = cv2.VideoCapture(source)
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video source")
            
            print(f"Video source opened: {source_type}")
        except Exception as e:
            print(f"Error opening source: {e}")
            raise
    
    def setModelMeta(self, model_meta: ModelMeta):
        self.modelMeta = model_meta
    
    def load_violations_from_disk(self):
        """Load existing violations from JSON file"""
        if self.violations_json.exists():
            try:
                with open(self.violations_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Load only recent violations (last 10) for memory
                    self.violation_screenshots = data[-10:]
                    print(f"Loaded {len(data)} violations from disk")
            except Exception as e:
                print(f"Error loading violations: {e}")
        
        # Run auto-cleanup if enabled
        if self.cleanup_enabled:
            self.cleanup_old_violations()
    
    def cleanup_old_violations(self):
        """Delete violations older than retention_days"""
        try:
            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            
            if self.violations_json.exists():
                with open(self.violations_json, 'r', encoding='utf-8') as f:
                    violations = json.load(f)
                
                # Filter out old violations
                filtered = [v for v in violations if v.get('timestamp', 0) > cutoff_time]
                deleted_count = len(violations) - len(filtered)
                
                # Delete old image files
                for v in violations:
                    if v.get('timestamp', 0) <= cutoff_time:
                        img_path = Path(v.get('image_path', ''))
                        if img_path.exists():
                            img_path.unlink()
                
                # Save filtered violations
                if deleted_count > 0:
                    with open(self.violations_json, 'w', encoding='utf-8') as f:
                        json.dump(filtered, f, indent=2, ensure_ascii=False)
                    print(f"Cleaned up {deleted_count} violations older than {self.retention_days} days")
                    
                    # Update CSV too
                    if self.violations_csv.exists():
                        with open(self.violations_csv, 'w', newline='', encoding='utf-8') as f:
                            if filtered:
                                fieldnames = filtered[0].keys()
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(filtered)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def save_violation_to_disk(self, violation_data):
        """Save single violation to disk (JSON + image)"""
        try:
            # Save image to disk
            timestamp = violation_data['timestamp']
            track_id = violation_data['track_id']
            image_filename = f"{self.session_id}_{track_id}_{int(timestamp)}.jpg"
            image_path = self.violation_images_dir / image_filename
            
            # Decode base64 and save
            import base64
            img_data = base64.b64decode(violation_data['image'])
            with open(image_path, 'wb') as f:
                f.write(img_data)
            
            # Update violation data with file path
            violation_record = {
                'session_id': self.session_id,
                'track_id': track_id,
                'speed': violation_data['speed'],
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                'image_path': str(image_path),
                'speed_limit': self.speed_limit,
                'line_upper': self.line_upper,
                'line_lower': self.line_lower
            }
            
            # Append to JSON file
            violations = []
            if self.violations_json.exists():
                with open(self.violations_json, 'r', encoding='utf-8') as f:
                    violations = json.load(f)
            
            violations.append(violation_record)
            
            with open(self.violations_json, 'w', encoding='utf-8') as f:
                json.dump(violations, f, indent=2, ensure_ascii=False)
            
            # Append to CSV file
            csv_exists = self.violations_csv.exists()
            with open(self.violations_csv, 'a', newline='', encoding='utf-8') as f:
                fieldnames = ['session_id', 'track_id', 'speed', 'timestamp', 'datetime', 'image_path', 'speed_limit', 'line_upper', 'line_lower']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not csv_exists:
                    writer.writeheader()
                
                writer.writerow(violation_record)
            
            print(f"Violation saved: ID {track_id}, Speed {violation_data['speed']} km/h")
            
        except Exception as e:
            print(f"Error saving violation: {e}")
    
    def export_violations_csv(self, output_path=None):
        """Export all violations to CSV file"""
        if output_path is None:
            output_path = self.violation_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            if self.violations_json.exists():
                with open(self.violations_json, 'r', encoding='utf-8') as f:
                    violations = json.load(f)
                
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    if violations:
                        fieldnames = violations[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(violations)
                
                print(f"Exported {len(violations)} violations to {output_path}")
                return str(output_path)
            else:
                print("No violations to export")
                return None
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return None
    
    def update_settings(self, settings: dict):
        """Update speed measurement settings"""
        if 'speed_limit' in settings:
            self.speed_limit = float(settings['speed_limit'])
        if 'line_upper' in settings:
            self.line_upper = int(settings['line_upper'])
        if 'line_lower' in settings:
            self.line_lower = int(settings['line_lower'])
        if 'dist_upward_m' in settings:
            self.dist_upward_m = float(settings['dist_upward_m'])
        if 'dist_downward_m' in settings:
            self.dist_downward_m = float(settings['dist_downward_m'])
        if 'retention_days' in settings:
            self.retention_days = int(settings['retention_days'])
        if 'cleanup_enabled' in settings:
            self.cleanup_enabled = bool(settings['cleanup_enabled'])
        if 'frame_skip' in settings:
            self.frame_skip = int(settings['frame_skip'])
        if 'optimization_enabled' in settings:
            self.optimization_enabled = bool(settings['optimization_enabled'])
        if 'perspective_correction_enabled' in settings:
            self.perspective_correction_enabled = bool(settings['perspective_correction_enabled'])
        
        # Recalculate derived values
        self.pixel_distance = abs(self.line_lower - self.line_upper)
        self.ppm_upward = self.pixel_distance / self.dist_upward_m
        self.ppm_downward = self.pixel_distance / self.dist_downward_m
    
    def calculate_speed_directional(self, time_in, time_out, direction_m):
        """Calculate speed based on time of flight between two lines"""
        time_diff = time_out - time_in
        
        if time_diff > 0.05 and time_diff < 10.0:
            speed_mps = direction_m / time_diff
            speed_kmh = speed_mps * 3.6
            
            if speed_kmh < 200:  # Sanity check - increased to 200 km/h
                return speed_kmh, time_diff, speed_mps
        
        return 0, time_diff, 0.0
    
    def capture_violation(self, frame, track_id, speed, bbox):
        """Capture screenshot of speeding vehicle"""
        x1, y1, x2, y2 = bbox
        
        # Crop vehicle region
        vehicle_crop = frame[y1:y2, x1:x2].copy()
        
        # Add info to frame
        info_frame = frame.copy()
        cv2.rectangle(info_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(info_frame, f"SPEEDING: {speed:.1f} km/h", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', info_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        violation_data = {
            'track_id': track_id,
            'speed': round(speed, 1),
            'timestamp': time.time(),
            'image': img_base64
        }
        
        # Store (keep only last N screenshots in memory)
        self.violation_screenshots.append(violation_data)
        if len(self.violation_screenshots) > self.max_screenshots:
            self.violation_screenshots.pop(0)
        
        # Save to disk permanently
        self.save_violation_to_disk(violation_data)
        
        return violation_data
    
    def inference_worker(self):
        """Main inference loop with speed measurement"""
        if not self.loadedModel or not self.modelMeta:
            print("Model not loaded")
            return
        
        class_keys = list(self.modelMeta.classes.keys())
        frame_counter = 0
        
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = self.frame_queue.get()
            
            # Frame skip optimization
            if self.optimization_enabled and self.frame_skip > 0:
                frame_counter += 1
                if frame_counter % (self.frame_skip + 1) != 0:
                    continue
            current_time = time.time()
            frame_height, frame_width = frame.shape[:2]
            
            # Run YOLO tracking
            results = self.loadedModel.track(
                frame,
                conf=self.modelMeta.conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=[int(k) for k in class_keys]
            )
            
            # Process detections
            detections = []
            violations = []
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                
                for detection in boxes:
                    if detection.size < 7:
                        continue
                    
                    x1, y1, x2, y2 = detection[:4].astype(int)
                    track_id = int(detection[4])
                    conf = float(detection[5])
                    cls_id = int(detection[6])
                    
                    # Get class name
                    cls_name = self.modelMeta.classes.get(str(cls_id), "unknown")
                    
                    # Tracking point (bottom center of bbox)
                    tracking_point_y = y2
                    
                    # Initialize tracker data
                    if track_id not in self.tracker_data:
                        self.tracker_data[track_id] = {
                            'time_in': 0.0,
                            'time_out': 0.0,
                            'line_in': None,
                            'speed': 0.0,
                            'is_speeding': False,
                            'direction': None
                        }
                    
                    # Check line crossing
                    is_at_upper = (tracking_point_y >= self.line_upper - self.line_tolerance and 
                                   tracking_point_y <= self.line_upper + self.line_tolerance)
                    is_at_lower = (tracking_point_y >= self.line_lower - self.line_tolerance and 
                                   tracking_point_y <= self.line_lower + self.line_tolerance)
                    
                    # Line In detection
                    if self.tracker_data[track_id]['time_in'] == 0.0:
                        if is_at_upper:
                            self.tracker_data[track_id]['time_in'] = current_time
                            self.tracker_data[track_id]['line_in'] = 'upper'
                        elif is_at_lower:
                            self.tracker_data[track_id]['time_in'] = current_time
                            self.tracker_data[track_id]['line_in'] = 'lower'
                    
                    # Line Out detection and speed calculation
                    if self.tracker_data[track_id]['time_in'] != 0.0 and self.tracker_data[track_id]['time_out'] == 0.0:
                        line_in = self.tracker_data[track_id]['line_in']
                        
                        if line_in == 'upper' and is_at_lower:
                            # Downward movement
                            self.tracker_data[track_id]['time_out'] = current_time
                            self.tracker_data[track_id]['direction'] = 'downward'
                            
                            speed, time_diff, speed_mps = self.calculate_speed_directional(
                                self.tracker_data[track_id]['time_in'],
                                self.tracker_data[track_id]['time_out'],
                                self.dist_downward_m
                            )
                            
                            self.tracker_data[track_id]['speed'] = speed
                            
                            if speed > self.speed_limit:
                                self.tracker_data[track_id]['is_speeding'] = True
                                violation = self.capture_violation(frame, track_id, speed, (x1, y1, x2, y2))
                                violations.append(violation)
                        
                        elif line_in == 'lower' and is_at_upper:
                            # Upward movement
                            self.tracker_data[track_id]['time_out'] = current_time
                            self.tracker_data[track_id]['direction'] = 'upward'
                            
                            speed, time_diff, speed_mps = self.calculate_speed_directional(
                                self.tracker_data[track_id]['time_in'],
                                self.tracker_data[track_id]['time_out'],
                                self.dist_upward_m
                            )
                            
                            self.tracker_data[track_id]['speed'] = speed
                            
                            if speed > self.speed_limit:
                                self.tracker_data[track_id]['is_speeding'] = True
                                violation = self.capture_violation(frame, track_id, speed, (x1, y1, x2, y2))
                                violations.append(violation)
                    
                    # Prepare detection data
                    speed = self.tracker_data[track_id]['speed']
                    is_speeding = self.tracker_data[track_id]['is_speeding']
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'track_id': int(track_id),
                        'class': cls_name,
                        'conf': float(conf),
                        'speed': float(speed),
                        'is_speeding': bool(is_speeding)
                    })
            
            # Draw visualization on frame
            annotated_frame = frame.copy()
            
            # Draw measurement lines
            cv2.line(annotated_frame, (0, self.line_upper), (frame_width, self.line_upper), 
                     (0, 255, 255), 2)
            cv2.line(annotated_frame, (0, self.line_lower), (frame_width, self.line_lower), 
                     (255, 0, 0), 2)
            
            # Draw info text
            info_text = f"UP: {self.ppm_upward:.2f} ({self.dist_upward_m}m) | DOWN: {self.ppm_downward:.2f} ({self.dist_downward_m}m) | Limit: {self.speed_limit}km/h"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 0, 255) if det['is_speeding'] else (0, 255, 0)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID {det['track_id']}: {det['class']}"
                if det['speed'] > 0:
                    label += f" {det['speed']:.1f}km/h"
                
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare result
            result = {
                'frame': frame_base64,
                'detections': detections,
                'violations': violations,
                'settings': {
                    'speed_limit': self.speed_limit,
                    'line_upper': self.line_upper,
                    'line_lower': self.line_lower,
                    'ppm_upward': round(self.ppm_upward, 2),
                    'ppm_downward': round(self.ppm_downward, 2)
                }
            }
            
            # Put result in queue (drop old if full)
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass
            self.result_queue.put(result)
    
    def start_inference_thread(self):
        """Start the inference worker thread"""
        if self.inference_thread and self.inference_thread.is_alive():
            print("Inference thread already running")
            return
        
        self.inference_thread = threading.Thread(
            target=self.inference_worker,
            daemon=True
        )
        self.inference_thread.start()
        print("TOF Speed inference thread started")
    
    def set_batch_mode(self, enabled=True, output_dir=None):
        """Enable/disable batch processing mode"""
        self.batch_mode = enabled
        if output_dir:
            self.batch_output_dir = Path(output_dir)
            self.batch_output_dir.mkdir(exist_ok=True)
        print(f"Batch mode: {enabled}")
    
    def process_video_batch(self, video_path, output_dir=None):
        """Process a single video file in batch mode"""
        self.set_batch_mode(True, output_dir)
        
        try:
            # Open video source
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            self.running = True
            self.start_inference_thread()
            
            # Process all frames
            frame_count = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Put frame in queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
            
            print(f"Batch processing complete: {frame_count} frames")
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        # Don't clear tracker_data for cross-session persistence
        print("TOF Speed processor stopped")
