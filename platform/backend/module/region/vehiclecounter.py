    
from collections import defaultdict

class VehicleCounter:
    def __init__(self):
        # Current detections (not accumulated) - tracks active objects in regions
        self.current_vehicle_counts = defaultdict(lambda: {
            'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0
        })
        # Track lifetime counts (for legacy support)
        self.lifetime_vehicle_counts = defaultdict(lambda: {
            'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0
        })
        self.lifetime_tracked_ids = defaultdict(set)
        self.default_vehicle_types = {}
        self.periodical_vehicle_counts = {} # For storing counts per minute

    def reset_minute(self):
        """Reset lifetime counts every minute"""
        self.lifetime_vehicle_counts.clear()
        self.lifetime_tracked_ids.clear()
        self.default_vehicle_types = {}

    def update_current_vehicles(self, detections):
        """Update current vehicles in regions based on current detections"""
        # Reset current counts
        self.current_vehicle_counts.clear()
        
        # Re-initialize with zero counts
        for vtype in self.default_vehicle_types:
            for region_id in set(d.region_id for d in detections if d.region_id):
                if region_id not in self.current_vehicle_counts:
                    self.current_vehicle_counts[region_id] = self.default_vehicle_types.copy()
        
        # Count current detections
        for detection in detections:
            region_id = detection.region_id or 'global'
            if region_id not in self.current_vehicle_counts:
                self.current_vehicle_counts[region_id] = self.default_vehicle_types.copy()
            
            vehicle_class = detection.cls
            if vehicle_class in self.current_vehicle_counts[region_id]:
                self.current_vehicle_counts[region_id][vehicle_class] += 1

    def add_vehicle(self, region_id: str, vehicle_class: str, track_id: int):
        """Track unique vehicles for lifetime counting (once per track_id)"""
        if track_id not in self.lifetime_tracked_ids[region_id]:
            self.lifetime_tracked_ids[region_id].add(track_id)
            self.lifetime_vehicle_counts[region_id][vehicle_class] += 1

    def get_region_stats(self):
        """Return current vehicle counts in regions"""
        return dict(self.current_vehicle_counts)

    def get_total_counts(self):
        """Return total current counts across all regions"""
        total = self.default_vehicle_types.copy()
        for region_counts in self.current_vehicle_counts.values():
            for k in total:
                total[k] += region_counts[k]
        return total
    
    def set_vehicle_types(self, vehicle_types: list[str]):
        print("Setting vehicle types:", vehicle_types)
        for vehicle_type in vehicle_types:
            self.default_vehicle_types[vehicle_type] = 0
        
        # Initialize current counts with defaults
        for region_id in list(self.current_vehicle_counts.keys()):
            for vt in vehicle_types:
                if vt not in self.current_vehicle_counts[region_id]:
                    self.current_vehicle_counts[region_id][vt] = 0