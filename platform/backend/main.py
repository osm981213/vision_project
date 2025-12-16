# backend/main.py
# Optimized FastAPI + WebSocket + YOLO11s Tracking (imgsz=640)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from region.regionSearch import CCTVProcessor
import cv2
import numpy as np
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


counter = VehicleCounter()


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