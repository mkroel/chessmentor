import chess

NAMES = [
    "white-pawn",
    "white-knight",
    "white-bishop",
    "white-rook",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-knight",
    "black-bishop",
    "black-rook",
    "black-queen",
    "black-king",
]


def class_id(piece):
    # PAWN=1..KING=6 -> 0..5 white, 6..11 black
    offset = 0 if piece.color == chess.WHITE else 6
    return (piece.piece_type - 1) + offset


def piece_from_class(class_index):
    piece_type = (class_index % 6) + 1
    color = chess.WHITE if class_index < 6 else chess.BLACK
    return chess.Piece(piece_type, color)
