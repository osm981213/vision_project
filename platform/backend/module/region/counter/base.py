from abc import ABC, abstractmethod

class VehicleCounter(ABC):
    @abstractmethod
    def update(self, detections):
        pass

    @abstractmethod
    def get_stats(self):
        pass
