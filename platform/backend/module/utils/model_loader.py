import json
from pathlib import Path
from model.model_registry import ModelMeta
from dataclasses import dataclass
from pathlib import Path
import json
from ultralytics import YOLO
from pathlib import Path

# 모델 디렉토리 생성
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_JSON = MODEL_DIR / "models.json"

# 기본 제공되는 YOLO 모델 메타데이터 목록
BASE_MODELS = [
    {
        "id": "yolo11n",
        "file": "model/yolo11n.pt",
        "display_name": "YOLO11 Nano (Fast)",
        "description": "Fastest model, low latency",
        "type": "builtin",
        "default": False
    },
    {
        "id": "yolo11s",
        "file": "model/yolo11s.pt",
        "display_name": "YOLO11 Small (Balanced)",
        "description": "Balanced speed and accuracy",
        "type": "builtin",
        "default": True
    }
]


# 모델 레지스트리 JSON 파일 경로
MODEL_JSON = Path("model/models.json")

# 모델 레지스트리 로드 함수
# 모델 메타데이터를 포함하는 ModelMeta 객체 목록을 반환
# 존재하지 않는 모델 파일은 무시
# 예) {"models": [ { "id": "model1", "file": "path/to/model1.pt", "display_name": "Model 1", ... }, ... ]}
def load_model_registry():
    if not MODEL_JSON.exists():
        return []

    with MODEL_JSON.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    models = []
    for m in raw.get("models", []):
        meta = ModelMeta(**m)
        if meta.exists():
            models.append(meta)

    return models

# 모델 레지스트리 저장 함수
# ModelMeta 객체 목록을 JSON 파일로 저장
# 예) {"models": [ { "id": "model1", "file": "path/to/model1.pt", "display_name": "Model 1", ... }, ... ]}
def save_model_registry(models: list[ModelMeta]):
    data = {
        "models": [m.__dict__ for m in models]
    }
    with MODEL_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
# YOLO의 자동 다운로드 기능을 이용해
# 모델이 없으면 받아오고, 있으면 그대로 사용
def ensure_base_model(model_name: str) -> Path | None:
    try:
        model = YOLO(model_name)  # 없으면 다운로드
        return Path(model.ckpt_path)
    except Exception as e:
        print(f"[ERROR] Failed to load base model {model_name}: {e}")
        return None

def init_default_models():
    existing = load_model_registry()
    existing_ids = {m.id for m in existing}

    updated = False

    for base in BASE_MODELS:
        if base["id"] in existing_ids:
            continue

        ckpt_path = ensure_base_model(base["yolo_name"])
        if not ckpt_path:
            continue

        meta = ModelMeta(
            id=base["id"],
            file=str(ckpt_path),
            display_name=base["display_name"],
            description=base["description"],
            type="builtin",
            default=base["default"]
        )

        existing.append(meta)
        updated = True

    if not MODEL_JSON.exists() or updated:
        save_model_registry(existing)
        print("[INFO] Base YOLO models ensured and registry updated")
