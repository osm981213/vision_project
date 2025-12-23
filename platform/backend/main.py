# backend/main.py
# Optimized FastAPI + WebSocket + YOLO11s Tracking (imgsz=640)

import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from module.region.regionsearch import CCTVProcessor
from module.region.vehiclecounter import VehicleCounter
from module.CalibratedSpeed.calibrated_speed import CalibratedSpeedProcessor
from module.TOFSpeed.tof_speed import TOFSpeedProcessor
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
calibrated_processor = CalibratedSpeedProcessor()
tof_processor = TOFSpeedProcessor()

# Track active WebSocket connections for each processor
active_cctv_connections = set()
active_calibrated_connections = set()
active_tof_connections = set()

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
    connection_id = id(websocket)
    active_cctv_connections.add(connection_id)
    print(f"WebSocket connected (ID: {connection_id}, Active: {len(active_cctv_connections)})")

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
                        try:
                            await websocket.send_json({"type":"error","message":"modelTarget is None"})
                        except:
                            break
                        continue
                    # set Model Meta
                    model_meta = get_modelMeta_by_id(msg.get("modelTarget").get("id"))
                    if model_meta is None:
                        try:
                            await websocket.send_json({"type":"error","message":"model_meta is None"})
                        except:
                            break
                        continue
                    processor.setModelMeta(model_meta)
                        
                    ok_model = processor.load_model(model_meta.id, msg.get("custom_weights"))
                    if not ok_model:
                        try:
                            await websocket.send_json({"type":"error","message":"model load failed"})
                        except:
                            break
                        continue

                    ok_src = processor.open_source(msg.get("source_type"), msg.get("source"), UPLOAD_DIR)
                    if not ok_src:
                        try:
                            await websocket.send_json({"type":"error","message":"source open failed"})
                        except:
                            break
                        processor.running = False
                        continue

                    processor.regions = msg.get("regions", [])
                    processor.running = True
                    counter.set_vehicle_types(list(model_meta.classes.values()))
                    # stop any existing inference thread
                    # processor.stop_inference_thread()
                    # dev면 predict 모드 체크
                    if getattr(model_meta, "dev", False):
                        processor.mode = msg.get("modelTarget").get("mode", "track")
                        processor.start_inference_thread_auto(counter)
                    else:
                        print("test22")
                        processor.mode = "track"  # 기존대로
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
                
                try:        
                    await websocket.send_json({
                        "type": "frame",
                        "frame": result["frame"],
                        "resized_size": result["resized_size"],
                        "orig_size": result["orig_size"],
                        "detections": json_detections,
                        "detections_by_min": counter.get_total_counts(),
                        "stats": counter.get_region_stats()
                    })
                except:
                    break
                
            # TIMEOUT for no frames received
            # if not processor.running and processor.cap is not None:
            #     await websocket.send_json({"type":"timeout","message":"30초 동안 프레임이 수신되지 않아 종료되었습니다."})

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        active_cctv_connections.discard(connection_id)
        print(f"WebSocket disconnected (ID: {connection_id}, Active: {len(active_cctv_connections)})")
        
        # Only stop processor if no more active connections
        if len(active_cctv_connections) == 0:
            processor.running = False
            print("All CCTV connections closed - stopping processor")

# --------------------------
# CALIBRATED SPEED WEBSOCKET
# --------------------------
@app.websocket("/ws/calibrated-speed")
async def calibrated_speed_websocket(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_calibrated_connections.add(connection_id)
    print(f"Calibrated Speed WebSocket connected (ID: {connection_id}, Active: {len(active_calibrated_connections)})")

    try:
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(data)
                
                if msg["type"] == "config":
                    # Load model
                    model_path = msg.get("model_path", "model/s_best.pt")
                    ok_model = calibrated_processor.load_model(model_path)
                    if not ok_model:
                        try:
                            await websocket.send_json({"type":"error","message":"model load failed"})
                        except:
                            break
                        continue

                    # Open source
                    ok_src = calibrated_processor.open_source(
                        msg.get("source_type"), 
                        msg.get("source"), 
                        UPLOAD_DIR
                    )
                    if not ok_src:
                        try:
                            await websocket.send_json({"type":"error","message":"source open failed"})
                        except:
                            break
                        calibrated_processor.running = False
                        continue

                    # Capture and send first frame immediately for ROI setup
                    ret, first_frame = calibrated_processor.cap.read()
                    if ret:
                        import cv2
                        import base64
                        _, buffer = cv2.imencode('.jpg', first_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        try:
                            await websocket.send_json({
                                "type": "first_frame",
                                "frame": frame_base64,
                                "message": "First frame captured. Set ROI points if needed."
                            })
                        except:
                            break
                        # Reset position to start
                        calibrated_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    # Set calibration if provided
                    if msg.get("roi_points") and msg.get("width_meters") and msg.get("depth_meters"):
                        calibrated_processor.set_calibration(
                            msg.get("roi_points"),
                            msg.get("width_meters"),
                            msg.get("depth_meters")
                        )

                    calibrated_processor.running = True
                    calibrated_processor.start_inference_thread()
                    
                elif msg["type"] == "set_calibration":
                    # Update calibration
                    calibrated_processor.set_calibration(
                        msg.get("roi_points"),
                        msg.get("width_meters"),
                        msg.get("depth_meters")
                    )
                    try:
                        await websocket.send_json({"type":"calibration_set","message":"Calibration updated"})
                    except:
                        break
                
                elif msg["type"] == "clear_calibration":
                    # Clear ROI and calibration
                    calibrated_processor.roi_points = []
                    calibrated_processor.homography_matrix = None
                    calibrated_processor.object_tracks = {}
                    calibrated_processor.object_speeds = {}
                    try:
                        await websocket.send_json({"type":"calibration_cleared","message":"ROI cleared"})
                    except:
                        break

            except asyncio.TimeoutError:
                pass

            # Read video frame
            if calibrated_processor.running and calibrated_processor.cap is not None and calibrated_processor.cap.isOpened():
                ret, frame = calibrated_processor.cap.read()
                if ret:
                    # Clear old frames and keep only the latest
                    while not calibrated_processor.frame_queue.empty():
                        try:
                            calibrated_processor.frame_queue.get_nowait()
                        except:
                            break
                    calibrated_processor.frame_queue.put(frame)

            # Send inference result if available (only latest)
            if not calibrated_processor.result_queue.empty():
                # Get latest result and clear any older ones
                result = None
                while not calibrated_processor.result_queue.empty():
                    try:
                        result = calibrated_processor.result_queue.get_nowait()
                        # print("Calibrated Speed - Result queue wait time:", endTime - startTime)
                    except:
                        break
                if result:
                    try:
                        await websocket.send_json({
                            "type": "frame",
                            "frame": result["frame"],
                            "detections": result["detections"],
                            "stats": result["stats"]
                        })
                    except:
                        break

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        active_calibrated_connections.discard(connection_id)
        print(f"Calibrated Speed WebSocket disconnected (ID: {connection_id}, Active: {len(active_calibrated_connections)})")
        
        # Only stop processor if no more active connections
        if len(active_calibrated_connections) == 0:
            calibrated_processor.running = False
            print("All Calibrated Speed connections closed - stopping processor")

# --------------------------
# TOF SPEED WEBSOCKET
# --------------------------
@app.websocket("/ws/tof-speed")
async def tof_speed_websocket(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    active_tof_connections.add(connection_id)
    print(f"TOF Speed WebSocket connected (ID: {connection_id}, Active: {len(active_tof_connections)})")

    try:
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                msg = json.loads(data)
                
                if msg["type"] == "start":
                    source_type = msg.get("source_type", "http")
                    source = msg.get("source", "")
                    model_id = msg.get("model", "yolo11s")
                    custom_weights = msg.get("custom_weights", "")
                    settings = msg.get("settings", {})
                    
                    # Get model metadata
                    model_meta = get_modelMeta_by_id(model_id)
                    if model_meta:
                        tof_processor.setModelMeta(model_meta)
                    
                    # Load model
                    tof_processor.load_model(model_id, custom_weights)
                    
                    # Update settings
                    if settings:
                        tof_processor.update_settings(settings)
                    
                    # Open video source
                    tof_processor.open_source(source_type, source, UPLOAD_DIR)
                    
                    # Start processing
                    tof_processor.running = True
                    tof_processor.start_inference_thread()
                    
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": "Processing started"
                        })
                    except:
                        break
                
                elif msg["type"] == "stop":
                    tof_processor.running = False
                    
                elif msg["type"] == "update_settings":
                    settings = msg.get("settings", {})
                    tof_processor.update_settings(settings)
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "message": "Settings updated"
                        })
                    except:
                        break

            except asyncio.TimeoutError:
                pass

            # Read video frame
            if tof_processor.running and tof_processor.cap is not None and tof_processor.cap.isOpened():
                ret, frame = tof_processor.cap.read()
                if ret:
                    if not tof_processor.frame_queue.full():
                        tof_processor.frame_queue.put(frame)

            # Send inference result if available
            if not tof_processor.result_queue.empty():
                result = None
                try:
                    while not tof_processor.result_queue.empty():
                        result = tof_processor.result_queue.get_nowait()
                except:
                    pass
                
                if result:
                    try:
                        await websocket.send_json({
                            "type": "frame",
                            "frame": result["frame"],
                            "detections": result.get("detections", []),
                            "violations": result.get("violations", []),
                            "settings": {
                                "ppm_upward": tof_processor.ppm_upward,
                                "ppm_downward": tof_processor.ppm_downward
                            }
                        })
                    except:
                        break

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        active_tof_connections.discard(connection_id)
        print(f"TOF Speed WebSocket disconnected (ID: {connection_id}, Active: {len(active_tof_connections)})")
        
        # Only stop processor if no more active connections
        if len(active_tof_connections) == 0:
            tof_processor.running = False
            print("All TOF Speed connections closed - stopping processor")

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