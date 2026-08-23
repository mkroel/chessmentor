import chess
import numpy as np

from chessmentor.detect import predict_board
from chessmentor.vision import get_diff


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


def detect_played_move(
    frame, current_view, prev_board_view, board, H, config, detector, turn
):
    diff = get_diff(current_view, prev_board_view)
    active = [
        field
        for field, value in diff.items()
        if value > config["active_field_threshold"]
    ]

    if len(active) > config["max_active_fields"]:
        print(
            f"Active fields: {len(active)}, max diff: {max(diff.values()):.1f}, "
            f"median diff: {np.median(list(diff.values())):.1f}"
        )
        return retry_with_model(
            frame, H, board, f"Turn {turn}: too many changes ({len(active)})", detector
        )

    valid_moves = filter_moves_by_inventory(board)
    move_scores = {move: diff_score(board, move, diff) for move in valid_moves}

    # sort moves by score and promotion value
    promo_values = {chess.QUEEN: 4, chess.ROOK: 3, chess.BISHOP: 2, chess.KNIGHT: 1}
    sorted_moves = sorted(
        valid_moves,
        key=lambda m: (move_scores[m], promo_values.get(m.promotion, 0)),
        reverse=True,
    )

    if not sorted_moves:
        return None

    best_move = sorted_moves[0]
    second_detected = next(
        (
            m
            for m in sorted_moves[1:]
            if (m.from_square, m.to_square)
            != (best_move.from_square, best_move.to_square)
        ),
        None,
    )

    if move_scores[best_move] < 0.1:
        return retry_with_model(
            frame,
            H,
            board,
            f"Turn {turn}: no clear move (best score {move_scores[best_move]:.3f})",
            detector,
        )
    elif second_detected and (
        move_scores[best_move] - move_scores[second_detected] < 0.05
    ):
        return retry_with_model(
            frame,
            H,
            board,
            f"Turn {turn}: ambiguous ({best_move.uci()} against {second_detected.uci()})",
            detector,
        )

    return best_move


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


def check_against_image(frame, H, expected_board, label, detector):
    detected, outside, collisions = predict_board(detector.detect(frame), H)
    mismatches = compare_position(expected_board, detected)
    if mismatches:
        print(f"{label}: {len(mismatches)} Fields differ")
        print(f"  {describe_mismatches(mismatches)}")
    else:
        print(f"{label}: Position matches")
    if outside:
        print(
            f"  {outside} Detections outside the board, {collisions} Multiple placements"
        )

    return mismatches


def retry_with_model(frame, H, board, reason, detector):
    # if diff fails, try to detect the position with the model and match it to a legal move
    print(f"  {reason} -> second attempt with the model")
    detected, outside, _ = predict_board(detector.detect(frame), H)

    # filter_moves_by_inventory ist ja bereits in game.py
    move, info = match_moves_to_position(
        board, detected, filter_moves_by_inventory(board)
    )
    if move is None:
        print(f"  Model could not decide: {info}")
        if outside:
            print(f"  ({outside} Detections outside the board)")
        return None

    print(f"  Modell sagt {move.uci()} ({info})")
    return move


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
