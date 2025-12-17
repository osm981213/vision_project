from dataclasses import dataclass
from pathlib import Path

# 모델 메타데이터 클래스
# 모델의 ID, 파일 경로, 표시 이름, 설명, 유형 및 기본 여부를 포함
# 모델 파일의 존재 여부를 확인하는 메서드 포함
# 용도는 모델 레지스트리에서 모델 정보를 관리하는 것
@dataclass
class ModelMeta:
    id: str
    file: str
    display_name: str
    description: str = ""
    type: str = "builtin"
    default: bool = False

    def exists(self) -> bool:
        return Path(self.file).exists()
