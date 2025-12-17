from collections import defaultdict
from .base import VehicleCounter

class TrackVehicleCounter(VehicleCounter):
    def __init__(self):
        self.seen = set()
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            key = (d.region_id, d.track_id)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts
