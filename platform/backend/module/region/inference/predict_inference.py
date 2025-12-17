from ..domain.detection import Detection
from .base import InferenceStrategy

class PredictInference(InferenceStrategy):
    def __init__(self, model):
        self.model = model
        self.class_names = {2:'car',3:'motorcycle',5:'bus',7:'truck'}

    def infer(self, frame, regions):
        detections = []

        for region in regions:
            rx1, ry1 = int(region['x']), int(region['y'])
            rx2 = rx1 + int(region['w'])
            ry2 = ry1 + int(region['h'])

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            results = self.model.predict(crop, classes=[2,3,5,7], verbose=False)
            boxes = results[0].boxes
            if not boxes:
                continue

            for box, cls in zip(boxes.xyxy.cpu().numpy(),
                                boxes.cls.cpu().numpy().numpy()):
                bx1, by1, bx2, by2 = map(int, box)
                detections.append(
                    Detection(
                        x1=bx1+rx1, y1=by1+ry1,
                        x2=bx2+rx1, y2=by2+ry1,
                        cls=self.class_names[cls],
                        region_id=region['id']
                    )
                )
        return detections
