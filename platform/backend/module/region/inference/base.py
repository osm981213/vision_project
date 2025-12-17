from abc import ABC, abstractmethod
from typing import List
from domain.detection import Detection

# Abstract base class for inference strategies
class InferenceStrategy(ABC):
    @abstractmethod
    def infer(self, frame, regions) -> List[Detection]:
        pass
