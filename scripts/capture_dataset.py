# Dataset capture. Draws a randomly generated target position onto the live
# frame; the operator rebuilds it on the board and captures. Image and label
# therefore come from the same source.
#
#   python scripts/capture_dataset.py
#
# Keys:
#   1 2 3      density: scattered / medium / full
#   n          new random position
#   SPACE      capture
#   g          toggle grid
#   r          recalibrate, board must be empty
#   q          quit

import json
import time
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import grid_points, homography
from chessmentor.camera import configure_camera
from chessmentor.corners import get_corners
from chessmentor.render import LEGEND_DE, draw_grid, draw_position

OUT_DIR = Path("data/capture")
MANIFEST = OUT_DIR / "labels.jsonl"
WINDOW = "capture"

# name, piece count, at least one of each type
MODES = {
    ord("1"): ("sparse", 12, True),
    ord("2"): ("medium", 18, False),
    ord("3"): ("dense", 28, False),
}

FILLER = [chess.PAWN] * 8 + [chess.ROOK, chess.KNIGHT, chess.BISHOP] * 2 + [chess.QUEEN]


def random_position(n_pieces, one_of_each, rng):
    board = chess.Board(None)
    free = list(rng.permutation(64))

    def take(allowed=None):
        for i, sq in enumerate(free):
            if allowed is None or allowed(sq):
                return free.pop(i)
        return None

    # kings first, they must not be adjacent
    wk = take()
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    bk = take(lambda sq: chess.square_distance(sq, wk) > 1)
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    wanted = []
    if one_of_each:
        for t in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
            wanted += [(t, chess.WHITE), (t, chess.BLACK)]

    while len(wanted) + 2 < n_pieces:
        t = FILLER[rng.integers(len(FILLER))]
        c = chess.WHITE if rng.integers(2) else chess.BLACK
        wanted.append((t, c))

    for piece_type, color in wanted:
        ok = None
        if piece_type == chess.PAWN:
            # pawns do not occur on rank 1 or 8

            def ok(sq):
                return 0 < chess.square_rank(sq) < 7

        sq = take(ok)
        if sq is None:
            break
        board.set_piece_at(sq, chess.Piece(piece_type, color))

    return board


def next_index():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        return 0
    return sum(1 for _ in MANIFEST.open(encoding="utf-8"))


def save(frame, board, corners, mode):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = next_index()
    name = f"{idx:04d}_{time.strftime('%H%M%S')}.png"
    cv.imwrite(str(OUT_DIR / name), frame)

    row = {
        "image": name,
        "fen": board.board_fen(),
        "mode": mode,
        "corners": np.asarray(corners).tolist(),
        "corners_source": "capture",
    }
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return name, idx + 1


def status_lines(mode, n_pieces, saved, show_grid):
    return [
        f"Modus {mode} ({n_pieces} Figuren)   aufgenommen: {saved}"
        f"   Gitter: {'an' if show_grid else 'aus'}",
        LEGEND_DE + "     weiss = weisse Schrift, schwarz = rot",
        "1/2/3 Dichte   n neue Stellung   LEERTASTE aufnehmen   g Gitter"
        "   r neu kalibrieren   q Ende",
    ]


def main():
    with Path("config.yaml").open() as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng()
    cap = cv.VideoCapture(config["camera"]["index"], cv.CAP_DSHOW)
    try:
        configure_camera(cap, config)
        corners = get_corners(cap, config)
        _, H_inv = homography(np.float32(corners))
        img_grid = grid_points(H_inv)

        mode, n_pieces, one_of_each = MODES[ord("1")]
        board = random_position(n_pieces, one_of_each, rng)
        saved = next_index()
        show_grid = True

        cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
        print("Stellung nachbauen, dann LEERTASTE. q beendet.")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            view = frame.copy()
            if show_grid:
                draw_grid(view, img_grid)
            draw_position(view, H_inv, board)

            y = 30
            for line in status_lines(mode, n_pieces, saved, show_grid):
                cv.putText(
                    view, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4
                )
                cv.putText(
                    view, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
                )
                y += 26

            cv.imshow(WINDOW, view)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in MODES:
                mode, n_pieces, one_of_each = MODES[key]
                board = random_position(n_pieces, one_of_each, rng)
            elif key == ord("n"):
                board = random_position(n_pieces, one_of_each, rng)
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("r"):
                # a stale homography places the overlay on the wrong squares,
                # which silently invalidates every label captured afterwards
                print("Brett leerraeumen, dann Enter im Terminal...")
                input()
                corners = get_corners(cap, config, override=True)
                _, H_inv = homography(np.float32(corners))
                img_grid = grid_points(H_inv)
                print("Neu kalibriert")
            elif key == ord(" "):
                # store the raw frame, not the overlay view
                name, saved = save(frame, board, corners, mode)
                print(f"{name}  {board.board_fen()}")
                board = random_position(n_pieces, one_of_each, rng)

        print(f"\n{saved} Bilder in {OUT_DIR.resolve()}")
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
