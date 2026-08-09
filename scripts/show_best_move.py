from pathlib import Path

import chess
import cv2 as cv
import yaml

from chessmentor.board import grid_points, homography
from chessmentor.camera import configure_camera
from chessmentor.corners import get_corners
from chessmentor.engine import get_best_move
from chessmentor.render import draw_arrow, draw_grid


def main():
    with Path("config.yaml").open() as f:
        config = yaml.safe_load(f)

    print("Config loaded")

    cap = cv.VideoCapture(config["camera"]["index"], cv.CAP_DSHOW)
    try:
        configure_camera(cap, config)
        corners = get_corners(cap, config, override=True)
        _, H_inv = homography(corners)

        fen_string = "8/2K1k3/3N4/1P1n1p1R/2b3B1/7q/7r/2Q5 b - - 0 1"

        best_move, score = get_best_move(fen_string, config)

        move, score = get_best_move(fen_string, config)
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        print(f"Best move: {best_move}, score: {score}, from: {from_sq}, to: {to_sq}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            draw_grid(frame, grid_points(H_inv))
            draw_arrow(frame, H_inv, from_sq, to_sq)

            cv.imshow("Best Move", frame)
            cv.waitKey(1)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        raise RuntimeError(f"Error: {e}") from e
    finally:
        cap.release()


if __name__ == "__main__":
    main()
