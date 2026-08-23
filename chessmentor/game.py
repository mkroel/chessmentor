import chess


def get_expected_fields(board, move):
    after = board.copy()
    after.push(move)

    diff_fields = {}
    for fieldname in chess.SQUARE_NAMES:
        before_piece = board.piece_at(chess.parse_square(fieldname))
        after_piece = after.piece_at(chess.parse_square(fieldname))
        if before_piece != after_piece:
            diff_fields[fieldname] = (before_piece, after_piece)

    return diff_fields


def diff_score(board, move, diff, penalty=1.0 / 5.0):
    expected_fields = get_expected_fields(board, move)

    total_diff = sum(diff.values())
    if total_diff == 0:
        return 0.0

    expected_diff = []
    for fields in expected_fields:
        expected_diff.append(diff.get(fields, 0.0))

    hits = sum(expected_diff) / total_diff
    score = hits

    max_diff = max(diff.values()) if diff.values() else 0.0
    dynamic_threshold = max_diff / 2.0

    for val in expected_diff:
        if val < dynamic_threshold:
            score -= penalty

    return score


def filter_moves_by_inventory(board):
    color = board.turn

    inventory = {
        chess.QUEEN: max(0, 1 - len(board.pieces(chess.QUEEN, color))),
        chess.ROOK: max(0, 2 - len(board.pieces(chess.ROOK, color))),
        chess.BISHOP: max(0, 2 - len(board.pieces(chess.BISHOP, color))),
        chess.KNIGHT: max(0, 2 - len(board.pieces(chess.KNIGHT, color))),
    }

    filtered_moves = []
    promotions_by_square = {}

    for move in board.legal_moves:
        if move.promotion:
            sq_pair = (move.from_square, move.to_square)
            if sq_pair not in promotions_by_square:
                promotions_by_square[sq_pair] = []
            promotions_by_square[sq_pair].append(move)
        else:
            filtered_moves.append(move)

    # Filter promotion moves based on inventory
    for _, promo_moves in promotions_by_square.items():
        valid_promos = [m for m in promo_moves if inventory.get(m.promotion, 0) > 0]
        # If there are valid promotions, add them to the filtered moves; otherwise, add all promotion moves
        if valid_promos:
            filtered_moves.extend(valid_promos)
        else:
            filtered_moves.extend(promo_moves)

    return filtered_moves
