# Converts the capture manifest (image + FEN + corners) into a YOLO dataset.
#
#   python scripts/json2yolo.py            convert
#   python scripts/json2yolo.py --check    draw generated boxes back onto images

import json
import random
import shutil
import sys
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import field_corners_px, field_to_px, homography

SOURCE_PATH = Path("data/capture")
DEST_PATH = Path("data/yolo")
MANIFEST = SOURCE_PATH / "labels.jsonl"

VAL_RATIO = 0.2
SEED = 42

REL_WIDTH = 0.8
FOOT_DROP = 0.8

REL_HEIGHT = {
    chess.PAWN: 1.2,
    chess.KNIGHT: 1.5,
    chess.BISHOP: 1.5,
    chess.ROOK: 1.5,
    chess.QUEEN: 1.8,
    chess.KING: 2.0,
}

MIN_BOX = 0.005

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
    # PAWN=1..KING=6 -> 0..5 white, 6..11 black.
    return (piece.piece_type - 1) + (0 if piece.color else 6)


def _clamp(v):
    return min(max(v, 0.0), 1.0)


def box_for_piece(fieldname, piece, H_inv, img_w, img_h):
    center = field_to_px(fieldname, H_inv)

    quad = field_corners_px(fieldname, H_inv)
    field_px = float(
        np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
    )

    near = quad[np.argsort(quad[:, 1])[-2:]].mean(axis=0)
    foot = center + FOOT_DROP * (near - center)

    width = REL_WIDTH * field_px
    height = REL_HEIGHT[piece.piece_type] * field_px

    x1 = _clamp((foot[0] - width / 2) / img_w)
    x2 = _clamp((foot[0] + width / 2) / img_w)
    y1 = _clamp((foot[1] - height) / img_h)
    y2 = _clamp(foot[1] / img_h)

    if (x2 - x1) < MIN_BOX or (y2 - y1) < MIN_BOX:
        return None

    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def usable_rows():
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    return len(rows), [r for r in rows if r.get("usable") is True]


def write_data_yaml():
    data = {
        "path": str(DEST_PATH.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(NAMES)),
    }
    (DEST_PATH / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def convert():
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DEST_PATH / sub).mkdir(parents=True, exist_ok=True)

    total, rows = usable_rows()
    random.Random(SEED).shuffle(rows)
    cut = round((1 - VAL_RATIO) * len(rows))

    missing, boxes, dropped = 0, 0, 0
    for i, row in enumerate(rows):
        part = "train" if i < cut else "val"
        src = SOURCE_PATH / row["image"]
        img = cv.imread(str(src))
        if img is None:
            missing += 1
            continue

        img_h, img_w = img.shape[:2]
        _, H_inv = homography(np.float32(row["corners"]))

        board = chess.Board(None)
        board.set_board_fen(row["fen"])

        lines = []
        for square, piece in board.piece_map().items():
            box = box_for_piece(chess.square_name(square), piece, H_inv, img_w, img_h)
            if box is None:
                dropped += 1
                continue
            lines.append(f"{class_id(piece)} " + " ".join(f"{v:.6f}" for v in box))

        boxes += len(lines)
        shutil.copy(src, DEST_PATH / "images" / part / row["image"])
        label = DEST_PATH / "labels" / part / f"{Path(row['image']).stem}.txt"
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_data_yaml()

    print(f"Manifest   : {total} Zeilen, davon {len(rows)} usable")
    print(f"Split      : {cut} train / {len(rows) - cut} val")
    print(f"Boxen      : {boxes}")
    if dropped:
        print(f"verworfen  : {dropped} (zu klein nach dem Beschneiden)")
    if missing:
        print(f"Bild fehlt : {missing}")
    print(f"Ziel       : {DEST_PATH.resolve()}")


def check(count=5):
    out = DEST_PATH / "kontrolle"
    out.mkdir(parents=True, exist_ok=True)

    images = sorted((DEST_PATH / "images" / "train").glob("*.png"))[:count]
    for path in images:
        img = cv.imread(str(path))
        img_h, img_w = img.shape[:2]
        label = DEST_PATH / "labels" / "train" / f"{path.stem}.txt"

        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))

            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)

            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                img,
                NAMES[int(cls)],
                (x1, y1 - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        cv.imwrite(str(out / path.name), img)

    print(f"{len(images)} Kontrollbilder in {out.resolve()}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        convert()
