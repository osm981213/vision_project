from dataclasses import dataclass
from typing import Optional

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: str
    region_id: Optional[str] = None
    track_id: Optional[int] = None

    def to_dict(self):
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "class": self.cls,
            "region_id": self.region_id,
            "track_id": self.track_id
        }
    def json_safe_dict(self):
        data = self.to_dict()
        if self.track_id is not None:
            data["track_id"] = int(self.track_id)
        if self.region_id is not None:
            data["region_id"] = str(self.region_id)
        return data