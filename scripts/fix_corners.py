# Adds missing board corners to the capture manifest. Detection is tried first,
# remaining images are corrected by mouse input. Per-image corners are required
# to generate bounding boxes.
#
#   python scripts/fix_corners.py
#
# Keys:
#   left click   set corner, clockwise starting top left
#   z            undo last click
#   ENTER        accept, grid preview is shown after four clicks
#   n            skip image
#   q            quit, keeping what was processed

import json
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np

from chessmentor.board import grid_points, homography
from chessmentor.corners import find_corners
from chessmentor.render import draw_grid

CAPTURE_DIR = Path("data/capture")
MANIFEST = CAPTURE_DIR / "labels.jsonl"
MAX_VIEW_WIDTH = 1100  # window size; clicks are scaled back to image coordinates
WINDOW = "Ecken klicken"


def signed_area(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2


def order_clockwise(pts):
    # image coordinates have y pointing down, so clockwise yields positive area
    out = list(pts)
    if signed_area(out) < 0:
        out.reverse()
    return [[float(x), float(y)] for x, y in out]


def click_corners(frame, title):
    scale = min(1.0, MAX_VIEW_WIDTH / frame.shape[1])
    small = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv.namedWindow(WINDOW, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WINDOW, on_mouse)

    while True:
        view = small.copy()
        for i, p in enumerate(pts):
            q = (int(p[0] * scale), int(p[1] * scale))
            cv.circle(view, q, 5, (0, 255, 0), -1)
            cv.putText(
                view,
                str(i + 1),
                (q[0] + 8, q[1] - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if len(pts) == 4:
            # grid preview makes a misplaced corner visible before accepting
            _, H_inv = homography(np.float32(order_clockwise(pts)))
            overlay = frame.copy()
            draw_grid(overlay, grid_points(H_inv))
            view = cv.resize(
                overlay, (small.shape[1], small.shape[0]), interpolation=cv.INTER_AREA
            )
            for p in pts:
                cv.circle(
                    view, (int(p[0] * scale), int(p[1] * scale)), 5, (0, 255, 0), -1
                )

        for i, line in enumerate(
            [
                title,
                "im Uhrzeigersinn ab oben links   z zurueck   ENTER ok   n skip   q Ende",
            ]
        ):
            y = 26 + i * 26
            cv.putText(view, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv.putText(
                view, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )

        cv.imshow(WINDOW, view)
        key = cv.waitKey(20) & 0xFF

        if key == ord("q"):
            return "quit"
        if key == ord("n"):
            return None
        if key == ord("z") and pts:
            pts.pop()
        if key in (13, 10) and len(pts) == 4:
            return order_clockwise(pts)


def main():
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    todo = [r for r in rows if "corners" not in r]
    print(f"{len(rows)} Eintraege, davon {len(todo)} ohne Ecken\n")

    auto, manual, skipped = 0, 0, 0
    stopped = False

    for i, row in enumerate(todo, 1):
        frame = cv.imread(str(CAPTURE_DIR / row["image"]))
        if frame is None:
            print(f"  {row['image']}: Bild fehlt, uebersprungen")
            skipped += 1
            continue

        found = find_corners(frame, {})
        if found is not None:
            row["corners"] = np.asarray(found[0]).tolist()
            row["corners_source"] = "detected"
            auto += 1
            print(f"  {row['image']}: automatisch ({found[1]:.2f} px)")
            continue

        result = click_corners(frame, f"[{i}/{len(todo)}]  {row['image']}")
        if result == "quit":
            stopped = True
            break
        if result is None:
            skipped += 1
            continue

        row["corners"] = result
        row["corners_source"] = "clicked"
        manual += 1

    cv.destroyAllWindows()

    if MANIFEST.exists():
        shutil.copy(MANIFEST, MANIFEST.with_suffix(".jsonl.bak"))
    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    done = sum(1 for r in rows if "corners" in r)
    print(f"\nautomatisch {auto}, geklickt {manual}, uebersprungen {skipped}")
    print(f"{done}/{len(rows)} Eintraege haben jetzt Ecken")
    if stopped:
        print("Abgebrochen - beim naechsten Lauf geht es bei den fehlenden weiter.")


if __name__ == "__main__":
    main()
