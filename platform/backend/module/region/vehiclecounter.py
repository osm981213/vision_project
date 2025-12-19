    
from collections import defaultdict

class VehicleCounter:
    def __init__(self):
        self.vehicle_counts = defaultdict(lambda: {
            'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0
        })
        self.tracked_ids = defaultdict(set)
        self.last_minute = None
        self.default_vehicle_types = {}
        self.periodical_vehicle_counts = {} # For storing counts per minute

    def reset_minute(self):
        self.vehicle_counts.clear()
        self.tracked_ids.clear()
        self.default_vehicle_types = {}

    def add_vehicle(self, region_id: str, vehicle_class: str, track_id: int):
        if track_id not in self.tracked_ids[region_id]:
            self.tracked_ids[region_id].add(track_id)
            self.vehicle_counts[region_id][vehicle_class] += 1

    def get_region_stats(self):
        return dict(self.vehicle_counts)

    def get_total_counts(self):
        total = self.default_vehicle_types.copy()
        for region_counts in self.vehicle_counts.values():
            for k in total:
                total[k] += region_counts[k]
        return total
    
    def set_vehicle_types(self, vehicle_types: list[str]):
        for region_id in self.vehicle_counts:
            for vt in vehicle_types:
                if vt not in self.vehicle_counts[region_id]:
                    self.vehicle_counts[region_id][vt] = 0
        
        for vehicle_type in vehicle_types:
            self.default_vehicle_types[vehicle_type] = 0