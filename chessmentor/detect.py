from dataclasses import dataclass

import chess
from ultralytics import YOLO

from chessmentor.board import px_to_field

NAMES = [
    "white-pawn",
    "white-knight",
    "white-bishop",
    "white-rook",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-knight",
    "black-bishop",
    "black-rook",
    "black-queen",
    "black-king",
]


def class_id(piece):
    # PAWN=1..KING=6 -> 0..5 white, 6..11 black
    offset = 0 if piece.color == chess.WHITE else 6
    return (piece.piece_type - 1) + offset


def piece_from_class(class_index):
    piece_type = (class_index % 6) + 1
    color = chess.WHITE if class_index < 6 else chess.BLACK
    return chess.Piece(piece_type, color)


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


def predict_board(detections, H):
    # one piece per square, highest confidence wins
    best = {}
    outside = 0
    for det in detections:
        field = px_to_field(det.foot_point, H)
        if field is None:
            outside += 1
            continue
        if field not in best or det.confidence > best[field].confidence:
            best[field] = det

    board = chess.Board(None)
    for field, det in best.items():
        board.set_piece_at(chess.parse_square(field), piece_from_class(det.class_id))
    return board, outside, len(detections) - len(best) - outside
