# backend/main.py
# Optimized FastAPI + WebSocket + YOLO11s Tracking (imgsz=640)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from module.region.regionsearch import CCTVProcessor
from module.region.vehiclecounter import VehicleCounter
from module.utils.model_loader import load_model_registry, save_model_registry, init_default_models
from model.model_registry import ModelMeta
import asyncio
import json
import shutil
import os
from uuid import uuid4
from pathlib import Path

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
                    ok_model = processor.load_model(msg.get("model_size","s"), msg.get("custom_weights"))
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

# 영상 업로드
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() or ".mp4"
    video_id = f"{uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / video_id

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "success", "video_id": video_id}

# 모델 리스트 호출
@app.get("/models")
async def get_models():
    models = load_model_registry()

    return {
        "models": [
            {
                "id": m.id,
                "display_name": m.display_name,
                "description": m.description,
                "path": m.file,
                "default": m.default
            }
            for m in models
        ]
    }


# 모델 업로드
@app.post("/upload_model")
async def upload_model(
    file: UploadFile = File(...),
    display_name: str = "Custom YOLO Model",
    description: str = ""
):
    if not file.filename.endswith(".pt"):
        return {"error": "Only .pt files allowed"}

    model_id = uuid4().hex[:8]
    filename = f"{model_id}_{file.filename}"
    save_path = CUSTOM_MODEL_DIR / filename

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    models = load_model_registry()
    models.append(ModelMeta(
        id=model_id,
        file=str(save_path),
        display_name=display_name,
        description=description,
        type="custom"
    ))
    save_model_registry(models)

    return {"status": "success"}



# --------------------------
# RUN SERVER
# --------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)