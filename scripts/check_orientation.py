# Verifies that stored corners match the stored FEN and sets orientation_checked
# and usable in the manifest.
#
# findChessboardCornersSB returns the 7x7 grid in an arbitrary one of four
# rotations. Corners can be geometrically correct while the board coordinate
# system is rotated by 90, 180 or 270 degrees, placing every piece on a wrong
# square.
#
#   python scripts/check_orientation.py
#
# Keys:
#   r          rotate corners by 90 degrees
#   ENTER      accept, rotation is baked into the stored corners
#   n          skip, leave unchanged
#   x          mark image as unusable
#   q          quit, keeping what was processed

import json
import shutil
from pathlib import Path

import chess
import cv2 as cv
import numpy as np

from chessmentor.board import grid_points, homography
from chessmentor.render import draw_grid, draw_position

CAPTURE_DIR = Path("data/capture")
MANIFEST = CAPTURE_DIR / "labels.jsonl"
MAX_VIEW_WIDTH = 1100
WINDOW = "Orientierung pruefen"


def rotate_corners(corners, k):
    # corners map in order to (0,0), (800,0), (800,800), (0,800). Shifting the
    # list makes a different physical corner become a8, which rotates the board
    # coordinate system by 90 degrees.
    k %= 4
    return [list(map(float, c)) for c in (list(corners)[k:] + list(corners)[:k])]


def review(frame, fen, corners, title):
    scale = min(1.0, MAX_VIEW_WIDTH / frame.shape[1])
    board = chess.Board(None)
    board.set_board_fen(fen)
    k = 0

    cv.namedWindow(WINDOW, cv.WINDOW_AUTOSIZE)
    while True:
        _, H_inv = homography(np.float32(rotate_corners(corners, k)))

        overlay = frame.copy()
        draw_grid(overlay, grid_points(H_inv))
        draw_position(overlay, H_inv, board)
        view = cv.resize(overlay, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        for i, line in enumerate(
            [
                title,
                f"Drehung: {k * 90} Grad",
                "r drehen   ENTER uebernehmen   n skip   x unbrauchbar   q Ende",
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
            return "quit", None
        if key == ord("n"):
            return "skip", None
        if key == ord("x"):
            return "bad", None
        if key == ord("r"):
            k = (k + 1) % 4
        if key in (13, 10):
            return "ok", (rotate_corners(corners, k), k)


def main():
    rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8")]
    todo = [r for r in rows if "corners" in r and not r.get("orientation_checked")]
    print(f"{len(rows)} Eintraege, {len(todo)} noch nicht geprueft\n")

    ok, rotated_n, skipped, bad = 0, 0, 0, 0
    stopped = False

    for i, row in enumerate(todo, 1):
        frame = cv.imread(str(CAPTURE_DIR / row["image"]))
        if frame is None:
            print(f"  {row['image']}: Bild fehlt")
            skipped += 1
            continue

        title = f"[{i}/{len(todo)}]  {row['image']}  ({row.get('corners_source', '?')})"
        action, result = review(frame, row["fen"], row["corners"], title)

        if action == "quit":
            stopped = True
            break
        if action == "skip":
            skipped += 1
            continue
        if action == "bad":
            row["usable"] = False
            row["orientation_checked"] = True
            bad += 1
            continue

        corners, k = result
        row["corners"] = corners
        row["orientation_checked"] = True
        row["usable"] = True
        ok += 1
        if k:
            rotated_n += 1
            print(f"  {row['image']}: um {k * 90} Grad gedreht")

    cv.destroyAllWindows()

    if MANIFEST.exists():
        shutil.copy(MANIFEST, MANIFEST.with_suffix(".jsonl.bak"))
    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    checked = sum(1 for r in rows if r.get("orientation_checked"))
    print(
        f"\nbestaetigt {ok} (davon {rotated_n} gedreht), skip {skipped}, unbrauchbar {bad}"
    )
    print(f"{checked}/{len(rows)} Eintraege geprueft")
    if stopped:
        print("Abgebrochen - naechster Lauf macht bei den ungeprueften weiter.")


if __name__ == "__main__":
    main()
