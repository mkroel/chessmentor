import cv2 as cv
import numpy as np

BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8


def transform_points(points, matrix):
    """Transform points using a matrix. Points are given as a list of (x, y) tuples.

    px coords -> board coords.
    board coords -> px coords.
    """
    pts = np.float32(points).reshape(-1, 1, 2)
    return cv.perspectiveTransform(pts, matrix).reshape(-1, 2)


def homography(corners):
    src = np.float32(corners)
    dst = np.float32([[0, 0], [BOARD_PX, 0], [BOARD_PX, BOARD_PX], [0, BOARD_PX]])
    return cv.getPerspectiveTransform(src, dst), cv.getPerspectiveTransform(dst, src)


def grid_points(H_inv):
    grid = []

    for row in range(9):
        for col in range(9):
            grid.append((col * SQUARE_PX, row * SQUARE_PX))

    return transform_points(grid, H_inv).reshape(9, 9, 2)


def field_to_sq(fieldname):
    file = ["a", "b", "c", "d", "e", "f", "g", "h"].index(fieldname[0])
    rank = 8 - int(fieldname[1])

    x = file * SQUARE_PX + SQUARE_PX / 2
    y = rank * SQUARE_PX + SQUARE_PX / 2

    return (x, y)


def field_to_px(fieldname, H_inv):
    return transform_points([field_to_sq(fieldname)], H_inv)[0]


def field_corners(fieldname):
    file = ["a", "b", "c", "d", "e", "f", "g", "h"].index(fieldname[0])
    rank = 8 - int(fieldname[1])

    left, top = file * SQUARE_PX, rank * SQUARE_PX

    return [
        (left, top),
        (left + SQUARE_PX, top),
        (left + SQUARE_PX, top + SQUARE_PX),
        (left, top + SQUARE_PX),
    ]


def field_corners_px(fieldname, H_inv):
    return transform_points(field_corners(fieldname), H_inv)


def rotate_corners(corners, k):
    # corners map in order to (0,0), (B,0), (B,B), (0,B)
    k %= 4
    seq = list(corners)
    return [list(map(float, c)) for c in seq[k:] + seq[:k]]


def px_to_field(point, H):
    p = transform_points([point], H)[0]
    col = int(p[0] // SQUARE_PX)
    row = int(p[1] // SQUARE_PX)
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    return "abcdefgh"[col] + str(8 - row)
