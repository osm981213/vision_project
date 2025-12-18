import base64
from queue import Queue
import threading
import time
import cv2
from ultralytics import YOLO
from domain.detection import Detection
from collections import defaultdict

from platform.backend.model.model_registry import ModelMeta

from .vehiclecounter import VehicleCounter



class CCTVProcessor:
    def resolve_video_path(self, upload_dir: str, video_id: str) -> str:
        p = upload_dir / video_id
        if not p.exists():
            raise FileNotFoundError(f"video not found: {video_id}")
        return str(p)
    
    def __init__(self):
        self.model = None
        self.cap = None
        self.regions = []
        self.running = False
        self.mode = "track"  # "detect" or "track"
        self.modelclasses = [2,3,5,7]  # car, motorcycle, bus, truck
        
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

        self.inference_thread = None

# --------------------------
# Video Source and Model Management
# model_size: 's', 'm', 'l', 'x' default 's'
# --------------------------
    def load_model(self, model_size='s', custom_weights=None):
        try:
            if custom_weights and custom_weights.strip():
                self.model = YOLO(custom_weights)
            else:
                self.model = YOLO(f"model/yolo11{model_size}.pt")
            print("Model loaded")
            return True
        except Exception as e:
            print("Error loading model:", e)
            return False

    def open_source(self, source_type, source, upload_dir):
        try:
            if self.cap:
                self.cap.release()

            if source_type == "rtsp":
                self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif source_type == "file":
                # source는 video_id
                path = self.resolve_video_path(upload_dir, source)
                self.cap = cv2.VideoCapture(path)

            elif source_type == "url":
                self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                print("Failed to open video source")
                return False

            print("Video source opened")
            return True
        except Exception as e:
            print("Error opening source:", e)
            return False

    def point_in_region(self, x, y, region):
        return (region['x'] <= x <= region['x'] + region['w'] and
                region['y'] <= y <= region['y'] + region['h'])

    # --------------------------
    # Inference Worker Thread
    # --------------------------
    def inference_worker(self, counter: VehicleCounter):
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        delay = 0
        if self.cap is not None:
            delay = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)  # ms 단위
            self.cap.get(cv2.CAP_PROP_FPS)
            print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
            print("Delay between frames (s):", delay)
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue
            if self.frame_queue.empty():
                continue
            startTime = time.time()
            frame = self.frame_queue.get()
            resized_w = 640
            resized_h = 360

            # Resize to lower resolution for speed
            resized = cv2.resize(frame, (resized_w, resized_h))
            plotted_frame = resized.copy()
            detections = []
            
            # 추론
            # if self.mode == "detect":
            #     inference_strategy = PredictInference(self.model)
            #     vehicle_counter = PredictVehicleCounter()
            # else:
            #     inference_strategy = TrackInference(self.model)
            #     vehicle_counter = TrackVehicleCounter()
            
            # detections = inference_strategy.infer(resized, self.regions)
            # vehicle_counter.update(detections)
            
            # self.plot_detections(plotted_frame, detections)
                

            # 추론 전략 선택
            if self.regions:
                for region in self.regions:                    
                    x1, y1, x2, y2 = int(region['x']), int(region['y']), int(region['x'] + region['w']), int(region['y'] + region['h'])
                    results = None
                    # region 단위로 추론
                    regionFrame = resized[y1:y2, x1:x2]
                    if self.mode == "track":
                        results = self.model.track(
                            regionFrame,                # FIXED SIZE
                            persist=True,
                            classes= self.modelclasses,
                            verbose=False
                        )
                    else:
                        results = self.model.predict(
                            regionFrame,                # FIXED SIZE
                            classes= self.modelclasses,
                            verbose=False
                        )
                    
                    
                    # save frame to disk for debug
                    # cv2.imwrite(f"uploaded_videos/debug_region_{region['id']}.jpg", regionFrame)
                    # cv2.imwrite(f"uploaded_videos/debug_fullframe_{region['id']}.jpg", results[0].plot())
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        # track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        classes = results[0].boxes.cls.cpu().numpy().astype(int)

                        for box, cls in zip(boxes, classes):
                            x1, y1, x2, y2 = box
                            
                            # Adjust box coordinates to full resized frame
                            x1 += region['x']
                            y1 += region['y']
                            x2 += region['x']
                            y2 += region['y']
                            
                            #to int
                            x1 = int(x1)
                            y1 = int(y1)
                            x2 = int(x2)
                            y2 = int(y2)
                            
                            vehicle_class = class_names.get(cls, "unknown")

                            color = {
                                'car': (0, 255, 0),
                                'bus': (0, 0, 255),
                                'truck': (0, 165, 255),
                                'motorcycle': (255, 200, 0)
                            }.get(vehicle_class, (255, 255, 255))

                            # bbox
                            cv2.rectangle(plotted_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                            # label
                            cv2.putText(plotted_frame, f"{vehicle_class} #{track_id}",
                                        (int(x1), int(y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                            # region counting
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2

                            if (region['x'] <= cx <= region['x'] + region['w'] and
                                region['y'] <= cy <= region['y'] + region['h']):
                                counter.add_vehicle(region["id"], vehicle_class, track_id)

                        
            else:
                results = None
                # 전체 프레임 단위로 추론 
                if self.mode == "track":
                    results = self.model.track(
                        resized,
                        persist=True,
                        classes=self.modelclasses,
                        verbose=False
                    )
                else:
                    results = self.model.predict(
                        resized,
                        classes=self.modelclasses,
                        verbose=False
                    )
                
                # Get original frame size
                orig_h, orig_w = frame.shape[:2]   # 예: 1080, 1920
                # print(orig_h, orig_w , "")

                # Calculate scaling factors
                scale_x = orig_w / resized_w
                scale_y = orig_h / resized_h
                print(scale_x, scale_y , "scale")
                

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        vehicle_class = class_names.get(cls, "unknown")

                        color = {
                            'car': (0, 255, 0),
                            'bus': (0, 0, 255),
                            'truck': (0, 165, 255),
                            'motorcycle': (255, 200, 0)
                        }.get(vehicle_class, (255, 255, 255))

                        # bbox
                        cv2.rectangle(plotted_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # label
                        cv2.putText(plotted_frame, f"{vehicle_class} #{track_id}",
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # region counting
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        if len(self.regions) == 0:
                            counter.add_vehicle("global", vehicle_class, track_id)
                        else:
                            for region in self.regions:
                                if (region['x'] <= cx <= region['x'] + region['w'] and
                                    region['y'] <= cy <= region['y'] + region['h']):
                                    counter.add_vehicle(region["id"], vehicle_class, track_id)

            # region 그리기 (same resized canvas 기준)
            for region in self.regions:
                cv2.rectangle(
                    plotted_frame,
                    (int(region['x']), int(region['y'])),
                    (int(region['x'] + region['w']), int(region['y'] + region['h'])),
                    (0, 255, 255), 2
                )
                cv2.putText(plotted_frame, f"Region {region['id']}",
                            (int(region['x']), int(region['y']) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            

            # Encode frame to base64
            # _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _, buffer = cv2.imencode(".jpg", plotted_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # _, buffer = cv2.imencode(".jpg", results[0].plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # Push result to queue
            if self.result_queue.full():
                self.result_queue.get()

            self.result_queue.put({
                "frame": frame_base64,
                "detections": detections,
                "orig_size": (orig_w, orig_h),
                "resized_size": (resized_w, resized_h)
            })
            
            endTime = time.time()
            elapsed = endTime - startTime
            if elapsed > delay * 2: # 2배 이상 걸리면 본 프레임을 따라가기 위해 스킵
                skip_count = int(elapsed / delay) - 1
                skip_count = max(skip_count, 1)  # 최소 1개는 스킵
                print(f"[WARNING] Inference time {elapsed:.3f}s exceeds frame delay {delay:.3f}s, skipping {skip_count} frames")
                for _ in range(skip_count):
                    self.cap.grab()   # 프레임 건너뛰기

    def start_inference_thread(self, counter: VehicleCounter):
        if self.inference_thread and self.inference_thread.is_alive():
            return
        
        self.inference_thread = threading.Thread(
            target=self.inference_worker,
            args=(counter,),
            daemon=True
        )        
        self.inference_thread.start()
        print("Inference thread started")
        
    def plot_detections(self, plotted_frame, detections):
        for d in detections:
            color = {
                'car': (0, 255, 0),
                'bus': (0, 0, 255),
                'truck': (0, 165, 255),
                'motorcycle': (255, 200, 0)
            }.get(d.cls, (255, 255, 255))

            cv2.rectangle(plotted_frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)

            label = d.cls
            if d.track_id is not None:
                label += f" #{d.track_id}"

            cv2.putText(
                plotted_frame,
                label,
                (d.x1, d.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        

# --------------------------
# Predict Inference Strategy
# --------------------------
class PredictInference:
    def __init__(self, model, model_meta: ModelMeta):
        self.model = model
        self.class_map = model_meta.classes  # {"car":2, ...}
        self.class_ids = list(self.class_map.values())
        self.id_to_name = {v: k for k, v in self.class_map.items()}
        
    def infer(self, frame, regions):
        detections = []

        if not regions:
            regions = [{
                "id": "global",
                "x": 0,
                "y": 0,
                "w": frame.shape[1],
                "h": frame.shape[0]
            }]
        
        for region in regions:
            rx1, ry1 = int(region['x']), int(region['y'])
            rx2 = rx1 + int(region['w'])
            ry2 = ry1 + int(region['h'])

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            results = self.model.predict(crop, classes=[2,3,5,7], verbose=False)
            boxes = results[0].boxes
            if not boxes:
                continue

            for box, cls in zip(boxes.xyxy.cpu().numpy(),
                                boxes.cls.cpu().numpy().numpy()):
                bx1, by1, bx2, by2 = map(int, box)
                detections.append(
                    Detection(
                        x1=bx1+rx1, y1=by1+ry1,
                        x2=bx2+rx1, y2=by2+ry1,
                        cls=self.class_names[cls],
                        region_id=region['id']
                    )
                )
        return detections
    
class PredictVehicleCounter(VehicleCounter):
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts

# --------------------------
# Track Inference Strategy
# --------------------------
class TrackInference:
    def __init__(self, model, model_meta: ModelMeta):
        self.model = model
        self.class_map = model_meta.classes  # {"car":2, ...}
        self.class_ids = list(self.class_map.values())
        self.id_to_name = {v: k for k, v in self.class_map.items()}

    def infer(self, frame, regions):
        detections = []

        results = self.model.track(
            frame,
            persist=True,
            classes=[2,3,5,7],
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        for box, cls, tid in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.id.cpu().numpy().astype(int)
        ):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            region_id = None
            for r in regions:
                if r['x'] <= cx <= r['x']+r['w'] and r['y'] <= cy <= r['y']+r['h']:
                    region_id = r['id']
                    break

            detections.append(
                Detection(
                    x1=int(box[0]), y1=int(box[1]),
                    x2=int(box[2]), y2=int(box[3]),
                    cls=self.class_names[cls],
                    track_id=tid,
                    region_id=region_id
                )
            )
        return detections
    
class TrackVehicleCounter(VehicleCounter):
    def __init__(self):
        self.seen = set()
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            key = (d.region_id, d.track_id)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts
