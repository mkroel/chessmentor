from dataclasses import dataclass

from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bbox: tuple  # x1, y1, x2, y2

    @property
    def foot_point(self):
        # lower center of the box
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class Detector:
    def __init__(self, weights, conf=0.1, imgsz=1280):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        return [
            Detection(
                int(box.cls),
                float(box.conf),
                tuple(float(v) for v in box.xyxy[0].tolist()),
            )
            for box in result.boxes
        ]
