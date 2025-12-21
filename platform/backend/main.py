# backend/main.py
# Optimized FastAPI + WebSocket + YOLO11s Tracking (imgsz=640)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from module.region.regionsearch import CCTVProcessor
from module.region.vehiclecounter import VehicleCounter
from module.utils.model_loader import load_model_registry, save_model_registry, init_default_models, get_modelMeta_by_id
from model.model_registry import ModelMeta
import asyncio
import json
import shutil
import os
from uuid import uuid4
from pathlib import Path
from app_router import router

app = FastAPI()
counter = VehicleCounter()
processor = CCTVProcessor()

MODEL_DIR = Path("model")
CUSTOM_MODEL_DIR = MODEL_DIR / "custom"
UPLOAD_DIR = Path("uploaded_videos")

# Create upload directory if not exists
CUSTOM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize default models on startup
init_default_models()

# Clean up old uploaded videos on startup
for filename in os.listdir(UPLOAD_DIR):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# secruity - CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                    # counter reset
                    counter.reset_minute()
                    if(msg.get("modelTarget") is None):
                        await websocket.send_json({"type":"error","message":"modelTarget is None"})
                        continue
                    # set Model Meta
                    model_meta = get_modelMeta_by_id(msg.get("modelTarget").get("id"))
                    if model_meta is None:
                        await websocket.send_json({"type":"error","message":"model_meta is None"})
                        continue
                    processor.setModelMeta(model_meta)
                    
                    ok_model = processor.load_model(model_meta.id, msg.get("custom_weights"))
                    if not ok_model:
                        await websocket.send_json({"type":"error","message":"model load failed"})
                        continue

                    ok_src = processor.open_source(msg.get("source_type"), msg.get("source"), UPLOAD_DIR)
                    if not ok_src:
                        await websocket.send_json({"type":"error","message":"source open failed"})
                        processor.running = False
                        continue

                    processor.regions = msg.get("regions", [])
                    processor.running = True
                    counter.set_vehicle_types(list(model_meta.classes.values()))
                    processor.start_inference_thread(counter)
                    

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
                
                json_detections = [d.to_dict() for d in result["detections"]]
                # int32 safe
                for det in json_detections:
                    if det["track_id"] is not None:
                        det["track_id"] = int(det["track_id"])
                        
                        
                await websocket.send_json({
                    "type": "frame",
                    "frame": result["frame"],
                    "resized_size": result["resized_size"],
                    "orig_size": result["orig_size"],
                    "detections": json_detections,
                    "detections_by_min": counter.get_total_counts(),
                    "stats": counter.get_region_stats()
                })
                
            # TIMEOUT for no frames received
            # if not processor.running and processor.cap is not None:
            #     await websocket.send_json({"type":"timeout","message":"30초 동안 프레임이 수신되지 않아 종료되었습니다."})

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        processor.running = False
        print("WebSocket disconnected")

# --------------------------
# REST API
# --------------------------

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "CCTV YOLO11s Tracking System API"}

# Health Check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": processor.model is not None,
        "source_active": processor.cap is not None and processor.cap.isOpened()
    }
    
# Include additional routes from app_router
app.include_router(router)


# --------------------------
# RUN SERVER
# --------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)