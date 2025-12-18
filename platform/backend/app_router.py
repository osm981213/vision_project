from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil
from uuid import uuid4
from module.utils.model_loader import load_model_registry, save_model_registry
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
    display_name: str = Form(...),
    description: str = Form(""),
    classes: str = Form(...)
):
    """
    classes: JSON string
    예: {"car":2,"bus":5}
    """
    if not file.filename.endswith(".pt"):
        return {"error": "Only .pt files allowed"}

    class_dict = json.loads(classes)

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
        classes=class_dict,
        type="custom"
    ))
    save_model_registry(models)

    return {"status": "success"}

# 모델 클래스 맵 수정
@router.patch("/models/{model_id}/classes")
async def update_model_classes(model_id: str, classes: dict):
    models = load_model_registry()
    for m in models:
        if m.id == model_id:
            m.classes = classes
            save_model_registry(models)
            return {"status": "updated"}
    return {"error": "model not found"}
