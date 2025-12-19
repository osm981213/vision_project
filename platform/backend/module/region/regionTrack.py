from ultralytics import solutions
from .vehiclecounter import VehicleCounter

class tracking:
    def __init__(self):
        self.tracker = solutions.track.Tracker(
            model="model/yolo11s.pt",
            persist=True,
            device="cpu",
            conf=0.3,
            iou=0.5,
            max_lost=30,
            tracker="bytetrack.yaml",
        )