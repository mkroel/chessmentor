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


def field_to_sq(fieldname):
    file = ["a", "b", "c", "d", "e", "f", "g", "h"].index(fieldname[0])
    rank = 8 - int(fieldname[1])

    x = file * SQUARE_PX + SQUARE_PX / 2
    y = rank * SQUARE_PX + SQUARE_PX / 2

    return (x, y)


def field_to_px(fieldname, H_inv):
    center = field_to_sq(fieldname)
    return cv.perspectiveTransform(np.float32([[center]]), H_inv).reshape(2)


def field_corners(fieldname):
    file = ["a", "b", "c", "d", "e", "f", "g", "h"].index(fieldname[0])
    rank = 8 - int(fieldname[1])

    left = file * SQUARE_PX
    top = rank * SQUARE_PX

    return [
        (left, top),
        (left + SQUARE_PX, top),
        (left + SQUARE_PX, top + SQUARE_PX),
        (left, top + SQUARE_PX),
    ]


def field_corners_px(fieldname, H_inv):
    pts = np.float32(field_corners(fieldname)).reshape(-1, 1, 2)
    return cv.perspectiveTransform(pts, H_inv).reshape(-1, 2)
