import cv2 as cv
import numpy as np

BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8


def homography(corners):
    src = np.float32(corners)
    dst = np.float32([[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]])
    return cv.getPerspectiveTransform(src, dst), cv.getPerspectiveTransform(dst, src)


def grid_points(H_inv):
    grid = np.float32(
        [[[col * SQUARE_PX, row * SQUARE_PX]] for row in range(9) for col in range(9)]
    )
    return cv.perspectiveTransform(grid, H_inv).reshape(9, 9, 2)
