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
