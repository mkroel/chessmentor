# Converts capture manifests (image + FEN + corners) into a YOLO dataset.
# Several source directories can be combined; each is split separately so
# both end up represented in train and val.
#
#   python scripts/json2yolo.py                          data/capture
#   python scripts/json2yolo.py data/capture data/game    combined
#   python scripts/json2yolo.py --check                   draw boxes back

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
from chessmentor.pieces import NAMES, class_id

DEFAULT_SOURCES = [Path("data/capture")]
DEST_PATH = Path("data/yolo")

VAL_RATIO = 0.2
SEED = 42

REL_WIDTH = 0.9
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


def _clamp(v):
    return min(max(v, 0.0), 1.0)


def box_for_piece(fieldname, piece, H_inv, img_w, img_h):
    center = field_to_px(fieldname, H_inv)

    quad = field_corners_px(fieldname, H_inv)
    side_lengths = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(a - b))
    field_px = float(np.mean(side_lengths))

    # take the two lowest corner points and calculate the avg
    idx = quad[:, 1].argsort()[-2:]
    near = quad[idx].mean(axis=0)
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


def load_rows(source):
    manifest = source / "labels.jsonl"
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    keep = [r for r in rows if r.get("usable") is not False]
    unchecked = sum(1 for r in keep if "usable" not in r)
    return rows, keep, unchecked


def split(rows, ratio, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = round((1 - ratio) * len(shuffled))
    return shuffled[:cut], shuffled[cut:]


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


def plan(sources):
    tasks = []
    for source in sources:
        all_rows, keep, unchecked = load_rows(source)
        train, val = split(keep, VAL_RATIO, SEED)
        for part, group in (("train", train), ("val", val)):
            tasks += [(source, row, part) for row in group]
        print(
            f"{source!s:20} {len(all_rows):4} Zeilen, {len(keep):4} verwendet"
            f"  ({len(train)} train / {len(val)} val)"
            + (f"   davon {unchecked} ungeprueft" if unchecked else "")
        )


def convert(sources):
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DEST_PATH / sub).mkdir(parents=True, exist_ok=True)

    tasks = plan(sources)

    missing, boxes, dropped = 0, 0, 0
    for source, row, part in tasks:
        name = f"{source.name}_{row['image']}"
        src = source / row["image"]
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
        shutil.copy(src, DEST_PATH / "images" / part / name)
        label = DEST_PATH / "labels" / part / f"{Path(name).stem}.txt"
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_data_yaml()

    n_train = sum(1 for _, _, part in tasks if part == "train")
    print("")
    print(
        f"Gesamt     : {len(tasks)} Bilder "
        f"({n_train} train / {len(tasks) - n_train} val)"
    )
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
    dirs = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    if "--check" in sys.argv:
        check()
    else:
        convert(dirs or DEFAULT_SOURCES)
