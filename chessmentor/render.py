import math
from pathlib import Path

import chess
import chess.svg
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


def update_browser_view(board, turn):
    # Status
    status = "Game in Progress"
    if board.is_checkmate():
        status = "Checkmate!"
    elif board.is_check():
        status = "Check!"
    elif board.is_stalemate():
        status = "Stalemate!"

    lastmove = board.peek() if board.move_stack else None
    last_move_str = lastmove.uci() if lastmove else "-"
    svg = chess.svg.board(board, size=400, lastmove=lastmove)

    inventory = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    uni_map = chess.UNICODE_PIECE_SYMBOLS

    lost_w, lost_b = [], []
    for pt, count in inventory.items():
        missing_w = max(0, count - len(board.pieces(pt, chess.WHITE)))
        missing_b = max(0, count - len(board.pieces(pt, chess.BLACK)))
        lost_w.extend([uni_map[chess.piece_symbol(pt).upper()]] * missing_w)
        lost_b.extend([uni_map[chess.piece_symbol(pt).lower()]] * missing_b)

    html = f"""
    <html>
    <head>
        <meta http-equiv='refresh' content='1'>
        <style>
            body {{ font-family: sans-serif; display: flex; gap: 20px; padding: 20px; background: #2c2c2c; color: white; }}
            .info {{ background: #3c3c3c; padding: 20px; border-radius: 8px; min-width: 250px; }}
            .status {{ color: #ff5252; font-weight: bold; }}
            .graveyard {{ font-size: 28px; letter-spacing: 5px; }}
        </style>
    </head>
    <body>
        <div>{svg}</div>
        <div class="info">
            <h2>Zug {turn}</h2>
            <p><strong>Status:</strong> <span class="status">{status}</span></p>
            <p><strong>Letzter Zug:</strong> {last_move_str}</p>
            <hr>
            <h3>Ausgeschieden:</h3>
            <p>Weiß: <span class="graveyard">{"".join(lost_w) or "-"}</span></p>
            <p>Schwarz: <span class="graveyard">{"".join(lost_b) or "-"}</span></p>
        </div>
    </body>
    </html>
    """

    path = Path("visu/board.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(html)

    return path
