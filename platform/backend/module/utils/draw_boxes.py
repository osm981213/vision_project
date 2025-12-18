import cv2
def plot_detections(self, plotted_frame, detections):
    for d in detections:
        color = {
            'car': (0, 255, 0),
            'bus': (0, 0, 255),
            'truck': (0, 165, 255),
            'motorcycle': (255, 200, 0)
        }.get(d.cls, (255, 255, 255))

        cv2.rectangle(plotted_frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)

        label = d.cls
        if d.track_id is not None:
            label += f" #{d.track_id}"

        cv2.putText(
            plotted_frame,
            label,
            (d.x1, d.y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
def draw_regions(self, plotted_frame, regions):
    for region in regions:
        cv2.rectangle(
            plotted_frame,
            (int(region['x']), int(region['y'])),
            (int(region['x'] + region['w']), int(region['y'] + region['h'])),
            (0, 255, 255), 2
        )
        cv2.putText(plotted_frame, f"Region {region['id']}",
                    (int(region['x']), int(region['y']) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)