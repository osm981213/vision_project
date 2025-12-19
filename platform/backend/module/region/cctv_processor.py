class CCTVProcessor:
    def __init__(self, inference, counter):
        self.inference = inference
        self.counter = counter

    def process_frame(self, frame, regions):
        detections = self.inference.infer(frame, regions)
        self.counter.update(detections)
        return detections, self.counter.get_stats()