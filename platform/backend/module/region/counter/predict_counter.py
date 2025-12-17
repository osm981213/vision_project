from collections import defaultdict
from .base import VehicleCounter

class PredictVehicleCounter(VehicleCounter):
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts
