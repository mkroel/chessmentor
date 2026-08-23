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


def diff_score(board, move, diff, penalty=0.6):
    expected_fields = get_expected_fields(board, move)

    total_diff = sum(diff.values())
    if total_diff == 0:
        return 0.0

    expected_diff = []
    for fields in expected_fields:
        expected_diff.append(diff.get(fields, 0.0))

    hits = sum(expected_diff) / total_diff

    max_diff = max(diff.values()) if diff.values() else 0.0
    dynamic_threshold = max_diff / 2.0

    # Count how many expected fields have a diff below the dynamic threshold
    quiet = sum(1 for val in expected_diff if val < dynamic_threshold)
    quiet_ratio = quiet / len(expected_diff) if expected_diff else 0.0

    return hits - penalty * quiet_ratio


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


def compare_position(expected, detected):
    # fields where the expected and detected positions differ -> {fieldname: (expected, detected)}
    mismatches = {}
    for square in chess.SQUARES:
        exp = expected.piece_at(square)
        det = detected.piece_at(square)
        if exp != det:
            mismatches[chess.square_name(square)] = (exp, det)

    return mismatches


def describe_mismatches(mismatches, limit=8):
    # short description
    parts = []
    for field, (exp, det) in list(mismatches.items())[:limit]:
        want = exp.symbol() if exp else "leer"
        got = det.symbol() if det else "leer"
        parts.append(f"{field}: {want}->{got}")
    if len(mismatches) > limit:
        parts.append(f"... (+{len(mismatches) - limit})")

    return ", ".join(parts)


def match_moves_to_position(board, detected, moves, min_agreement=58, margin=2):
    # compare the expected position after each move with the detected position
    scored = []
    for move in moves:
        board.push(move)
        agreement = sum(
            1
            for square in chess.SQUARES
            if board.piece_at(square) == detected.piece_at(square)
        )
        board.pop()
        scored.append((agreement, move))

    if not scored:
        return None, "no legal moves to compare"

    scored.sort(key=lambda entry: -entry[0])
    best_agreement, best_move = scored[0]
    second_agreement = scored[1][0] if len(scored) > 1 else -1

    if best_agreement < min_agreement:
        return None, f"No best move found ({best_agreement}/64)"
    if best_agreement - second_agreement < margin:
        return None, f"ambiguous ({best_agreement} to {second_agreement})"

    return best_move, f"{best_agreement}/64 Felder"
