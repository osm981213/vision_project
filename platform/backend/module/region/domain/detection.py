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
