import cv2 as cv
import numpy as np

from chessmentor.board import BOARD_PX, SQUARE_PX, field_to_sq


def board_view(frame, H):
    warped = cv.warpPerspective(frame, H, (BOARD_PX, BOARD_PX))
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    return gray


def get_tile(board_view, fieldname):
    # "e4" -> rank 4, file e -> 100x100
    field = field_to_sq(fieldname)
    x, y = map(int, field)

    tl = (x - SQUARE_PX // 2, y - SQUARE_PX // 2)
    br = (x + SQUARE_PX // 2, y + SQUARE_PX // 2)

    return board_view[tl[1] : br[1], tl[0] : br[0]]


def get_diff(board_view, prev_board_view):
    d = cv.absdiff(board_view, prev_board_view)
    cells = d.reshape(8, SQUARE_PX, 8, SQUARE_PX).swapaxes(1, 2)

    # compute mean difference per cell
    diff = np.mean(cells, axis=(2, 3))
    field_diffs = {}

    # index to field / ranks
    for row in range(8):
        for col in range(8):
            file = chr(ord("a") + col)
            rank = 8 - row
            field = f"{file}{rank}"

            field_diffs[field] = diff[row, col]

    return field_diffs


def is_still(view, prev_frame, threshold):
    movement = cv.absdiff(view, prev_frame)
    return movement.mean() < threshold
