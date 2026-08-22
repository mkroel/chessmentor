import math

import chess
import cv2 as cv
import numpy as np

from chessmentor.board import field_to_px

GRID_COLOR = (0, 255, 0)
BOARD_PX = 800
SQUARE_PX = BOARD_PX // 8
TIP_LENGTH_PX = 25

WHITE_PIECE_COLOR = (255, 255, 255)
BLACK_PIECE_COLOR = (0, 0, 255)
OUTLINE_COLOR = (0, 0, 0)

# german piece letters
PIECE_LETTER_DE = {
    chess.PAWN: "B",
    chess.KNIGHT: "S",
    chess.BISHOP: "L",
    chess.ROOK: "T",
    chess.QUEEN: "D",
    chess.KING: "K",
}

LEGEND_DE = "B=Bauer  S=Springer  L=Laeufer  T=Turm  D=Dame  K=Koenig"


def _pt(p):
    return tuple(map(int, np.round(p)))


def _text_centered(frame, text, center, color, scale, thickness):
    (w, h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = (int(center[0] - w / 2), int(center[1] + h / 2))
    cv.putText(
        frame,
        text,
        org,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv.LINE_AA,
    )


def draw_position(frame, H_inv, board, scale=0.9, thickness=2):
    # letter encodes the type, text color encodes the side
    for square, piece in board.piece_map().items():
        center = field_to_px(chess.square_name(square), H_inv)
        color = WHITE_PIECE_COLOR if piece.color == chess.WHITE else BLACK_PIECE_COLOR
        _text_centered(
            frame, PIECE_LETTER_DE[piece.piece_type], center, color, scale, thickness
        )


def draw_grid(frame, img_grid):
    for k in range(9):
        cv.line(frame, _pt(img_grid[k][0]), _pt(img_grid[k][8]), GRID_COLOR, 1)
        cv.line(frame, _pt(img_grid[0][k]), _pt(img_grid[8][k]), GRID_COLOR, 1)

        # draw the square labels
        if k < 8:
            # files
            cv.putText(
                frame,
                chr(ord("a") + k),
                _pt(img_grid[8][k] + np.array([SQUARE_PX / 2, 20])),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                GRID_COLOR,
                1,
                cv.LINE_AA,
            )
            # ranks
            cv.putText(
                frame,
                str(8 - k),
                _pt(img_grid[k][0] + np.array([-20, SQUARE_PX / 2])),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                GRID_COLOR,
                1,
                cv.LINE_AA,
            )


def draw_arrow(frame, H_inv, from_sq: str, to_sq: str, color=(0, 0, 255), thickness=2):
    from_px = field_to_px(from_sq, H_inv)
    to_px = field_to_px(to_sq, H_inv)

    dx = to_px[0] - from_px[0]
    dy = to_px[1] - from_px[1]
    arrow_len = math.hypot(dx, dy)
    arrow_tip = min(TIP_LENGTH_PX / arrow_len, 0.4)

    cv.arrowedLine(
        frame,
        _pt(from_px),
        _pt(to_px),
        color,
        thickness,
        tipLength=arrow_tip,
        line_type=cv.LINE_AA,
    )
