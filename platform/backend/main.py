# backend/main.py
# Optimized FastAPI + WebSocket + YOLO11s Tracking (imgsz=640)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import json
import base64
from datetime import datetime
from collections import defaultdict
from queue import Queue
import threading
import time
import shutil
import os
from uuid import uuid4
from pathlib import Path

app = FastAPI()

# Create upload directory if not exists
UPLOAD_DIR = Path("uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Vehicle Counter
# --------------------------
class VehicleCounter:
    def __init__(self):
        self.vehicle_counts = defaultdict(lambda: {
            'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0
        })
        self.tracked_ids = defaultdict(set)
        self.last_minute = None

    def reset_minute(self):
        self.vehicle_counts.clear()
        self.tracked_ids.clear()

    def add_vehicle(self, region_id: str, vehicle_class: str, track_id: int):
        if track_id not in self.tracked_ids[region_id]:
            self.tracked_ids[region_id].add(track_id)
            self.vehicle_counts[region_id][vehicle_class] += 1

    def get_region_stats(self):
        return dict(self.vehicle_counts)

    def get_total_counts(self):
        total = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}
        for region_counts in self.vehicle_counts.values():
            for k in total:
                total[k] += region_counts[k]
        return total


counter = VehicleCounter()


# --------------------------
# CCTV Processor
# --------------------------
class CCTVProcessor:
    def __init__(self):
        self.model = None
        self.cap = None
        self.regions = []
        self.running = False

        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

        self.inference_thread = None

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

    def open_source(self, source_type, source):
        try:
            if self.cap:
                self.cap.release()

            if source_type == "rtsp":
                self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif source_type == "file":
                # source는 video_id
                path = resolve_video_path(source)
                self.cap = cv2.VideoCapture(path)

            elif source_type == "url":
                self.cap = cv2.VideoCapture(source)
            # else:
            #     # For testing, use a local video file
            #     # src = "C:/Users/Woori/Downloads/L030032.mp4"
            #     src = src.replace("\\", "/")
            #     self.cap = cv2.VideoCapture(src)

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
    def inference_worker(self):
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

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

            # 추론
            results = self.model.track(
                resized,
                imgsz=640,                # FIXED SIZE
                persist=True,
                classes=[2, 3, 5, 7],
                verbose=False
            )

            detections = []
            
            # Get original frame size
            orig_h, orig_w = frame.shape[:2]   # 예: 1080, 1920
            # print(orig_h, orig_w , "")

            # Calculate scaling factors
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h
            print(scale_x, scale_y , "scale")
            
            plotted_frame = resized.copy()

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

            # version control #1 
            # if results[0].boxes.id is not None:
            #     boxes = results[0].boxes.xyxy.cpu().numpy()
            #     track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            #     classes = results[0].boxes.cls.cpu().numpy().astype(int)

            #     for box, track_id, cls in zip(boxes, track_ids, classes):
            #         x1, y1, x2, y2 = box
            #         cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            #         vehicle_class = class_names.get(cls, "unknown")

            #         # Count vehicle
            #         if len(self.regions) == 0:
            #             counter.add_vehicle("global", vehicle_class, track_id)
            #         else:
            #             for region in self.regions:
            #                 if self.point_in_region(cx, cy, region):
            #                     counter.add_vehicle(region["id"], vehicle_class, track_id)
                                
            #         # Adjust box coordinates to original frame size
            #         x1 *= scale_x
            #         y1 *= scale_y
            #         x2 *= scale_x
            #         y2 *= scale_y


            #         detections.append({
            #             "x": int(x1),
            #             "y": int(y1),
            #             "w": int(x2 - x1),
            #             "h": int(y2 - y1),
            #             "class": vehicle_class,
            #             "track_id": int(track_id)
            #         })
            endTime = time.time()
            print("Inference Time:", (endTime - startTime) * 1000, "ms")

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

    def start_inference_thread(self):
        if self.inference_thread and self.inference_thread.is_alive():
            return
        self.inference_thread = threading.Thread(
            target=self.inference_worker, daemon=True
        )
        self.inference_thread.start()
        print("Inference thread started")


processor = CCTVProcessor()

def resolve_video_path(video_id: str) -> str:
        p = UPLOAD_DIR / video_id
        if not p.exists():
            raise FileNotFoundError(f"video not found: {video_id}")
        return str(p)

# --------------------------
# WEBSOCKET
# --------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    try:
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(data)

                if msg["type"] == "config":
                    ok_model = processor.load_model(msg.get("model_size","s"), msg.get("custom_weights"))
                    if not ok_model:
                        await websocket.send_json({"type":"error","message":"model load failed"})
                        continue

                    ok_src = processor.open_source(msg.get("source_type"), msg.get("source"))
                    if not ok_src:
                        await websocket.send_json({"type":"error","message":"source open failed"})
                        processor.running = False
                        continue

                    processor.regions = msg.get("regions", [])
                    processor.running = True
                    processor.start_inference_thread()
                    # processor.regions = msg.get("regions", [])
                    # processor.load_model(msg.get("model_size", "s"), msg.get("custom_weights"))
                    # processor.open_source(msg.get("source_type", "rtsp"), msg.get("source"))

                    # processor.running = True
                    # processor.start_inference_thread()
                    

                elif msg["type"] == "update_regions":
                    processor.regions = msg.get("regions", [])

            except asyncio.TimeoutError:
                pass

            # Read video frame
            if processor.running and processor.cap is not None and processor.cap.isOpened():
                ret, frame = processor.cap.read()
                if ret:
                    if processor.frame_queue.full():
                        processor.frame_queue.get()
                    processor.frame_queue.put(frame)

            # Send inference result if available
            if not processor.result_queue.empty():
                result = processor.result_queue.get()
                await websocket.send_json({
                    "type": "frame",
                    "frame": result["frame"],
                    "resized_size": result["resized_size"],
                    "orig_size": result["orig_size"],
                    "detections": result["detections"],
                    "stats": counter.get_region_stats()
                })

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        processor.running = False
        print("WebSocket disconnected")


@app.get("/")
async def root():
    return {"message": "CCTV YOLO11s Tracking System API"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": processor.model is not None,
        "source_active": processor.cap is not None and processor.cap.isOpened()
    }

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() or ".mp4"
    video_id = f"{uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / video_id

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "success", "video_id": video_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)