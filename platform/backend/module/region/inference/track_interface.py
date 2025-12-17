from ..domain.detection import Detection
from .base import InferenceStrategy

class TrackInference(InferenceStrategy):
    def __init__(self, model):
        self.model = model
        self.class_names = {2:'car',3:'motorcycle',5:'bus',7:'truck'}

    def infer(self, frame, regions):
        detections = []

        results = self.model.track(
            frame,
            persist=True,
            classes=[2,3,5,7],
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        for box, cls, tid in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.id.cpu().numpy().astype(int)
        ):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            region_id = None
            for r in regions:
                if r['x'] <= cx <= r['x']+r['w'] and r['y'] <= cy <= r['y']+r['h']:
                    region_id = r['id']
                    break

            detections.append(
                Detection(
                    x1=int(box[0]), y1=int(box[1]),
                    x2=int(box[2]), y2=int(box[3]),
                    cls=self.class_names[cls],
                    track_id=tid,
                    region_id=region_id
                )
            )
        return detections
