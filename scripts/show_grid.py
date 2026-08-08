from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import grid_points, homography
from chessmentor.camera import configure_camera
from chessmentor.corners import pick_corners

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
    configure_camera(cap, config)
    corners, frame = pick_corners(cap, config)

    _, H_inv = homography(corners)
    draw_grid(frame, grid_points(H_inv))

    cv.imshow("Grid", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
