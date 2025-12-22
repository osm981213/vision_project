from fastapi import APIRouter, Form, UploadFile, File
from pathlib import Path
import shutil
import json
from uuid import uuid4
from module.utils.model_loader import load_model_registry, save_model_registry, update_model_in_registry
from model.model_registry import ModelMeta
from module.region.vehiclecounter import VehicleCounter
from module.region.regionsearch import CCTVProcessor

UPLOAD_DIR = Path("uploaded_videos")
CUSTOM_MODEL_DIR = Path("model/custom")

router = APIRouter()

# 영상 업로드
@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() or ".mp4"
    video_id = f"{uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / video_id

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "success", "video_id": video_id}

# 모델 리스트 호출
@router.get("/models")
async def get_models():
    models = load_model_registry()

    return {
        "models": models
    }

@router.get("/models")
async def get_models():
    models = load_model_registry()
    print("Loaded models:", models)

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
@router.post("/upload_model")
async def upload_model(
    file: UploadFile = File(...),
    display_name: str = Form(""),
    description: str = Form("")
):
    if not file.filename.endswith(".pt"):
        return {"error": "Only .pt files allowed"}

    display_name = file.filename
    model_id = uuid4().hex[:8]
    save_path = CUSTOM_MODEL_DIR / f"{model_id}_{file.filename}"

    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    models = load_model_registry()
    
    models.append(ModelMeta(
        id=model_id,
        file=str(save_path),
        display_name=display_name,
        description=description,
        classes={"0": "자동차", "1": "버스", "2": "트럭", "3": "오토바이"},
        type="custom"
    ))
    save_model_registry(models)

    return {"status": "success", "id": model_id, "path": str(save_path)}

# 모델 수정
@router.patch("/models/{model_id}")
async def update_model(model_id: str, model_data: dict):
    print(f"Updating model {model_id} with data: {model_data}")
    return update_model_in_registry(model_id, model_data)

# TOF Speed CSV Export
@router.post("/tof-speed/export-csv")
async def export_tof_speed_csv():
    from module.TOFSpeed.tof_speed import TOFSpeedProcessor
    processor = TOFSpeedProcessor()
    csv_path = processor.export_violations_csv()
    
    if csv_path:
        return {"status": "success", "csv_path": csv_path}
    else:
        return {"status": "no_data", "message": "No violations to export"}

# TOF Speed Batch Processing
@router.post("/tof-speed/batch-process")
async def batch_process_videos(files: list[UploadFile] = File(...)):
    from module.TOFSpeed.tof_speed import TOFSpeedProcessor
    
    processor = TOFSpeedProcessor()
    results = []
    
    for file in files:
        try:
            # Save uploaded video temporarily
            temp_path = UPLOAD_DIR / f"batch_{uuid4().hex}{Path(file.filename).suffix}"
            with temp_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Process video
            processor.process_video_batch(str(temp_path))
            
            results.append({
                "filename": file.filename,
                "status": "success"
            })
            
            # Clean up temp file
            temp_path.unlink()
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}
