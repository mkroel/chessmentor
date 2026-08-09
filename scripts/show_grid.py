from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import grid_points, homography
from chessmentor.camera import configure_camera, get_frame
from chessmentor.corners import get_corners

GRID_COLOR = (0, 255, 0)


def _pt(p):
    return tuple(map(int, np.round(p)))


def draw_grid(frame, img_grid):
    for k in range(9):
        cv.line(frame, _pt(img_grid[k][0]), _pt(img_grid[k][8]), GRID_COLOR, 1)
        cv.line(frame, _pt(img_grid[0][k]), _pt(img_grid[8][k]), GRID_COLOR, 1)


def main():
    with Path("config.yaml").open() as f:
        config = yaml.safe_load(f)

    print("Config loaded")

    cap = cv.VideoCapture(config["camera"]["index"], cv.CAP_DSHOW)
    try:
        configure_camera(cap, config)
        corners = get_corners(cap, config, override=True)

        frame = get_frame(cap)
        _, H_inv = homography(corners)
        draw_grid(frame, grid_points(H_inv))

        cv.imshow("Grid", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        raise RuntimeError(f"Error: {e}") from e
    finally:
        cap.release()


if __name__ == "__main__":
    main()
