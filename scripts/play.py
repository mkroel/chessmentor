# imports
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import webbrowser
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import grid_points, homography, rotate_corners
from chessmentor.camera import configure_camera
from chessmentor.corners import get_corners
from chessmentor.detect import Detector, predict_board
from chessmentor.engine import get_best_move
from chessmentor.game import (
    compare_position,
    describe_mismatches,
    diff_score,
    filter_moves_by_inventory,
    match_moves_to_position,
)
from chessmentor.render import draw_arrow, draw_grid, draw_position, update_browser_view
from chessmentor.vision import board_view, framing_check, get_diff, is_still

# load config
with Path("config.yaml").open() as f:
    config = yaml.safe_load(f)

# load detector model
det_cfg = config["detector"]
detector = Detector(
    det_cfg["weights_path"],
    conf=det_cfg["confidence_threshold"],
    imgsz=det_cfg["imgsz"],
)
print("Detector loaded")


def check_against_image(frame, H, expected_board, label):
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


def retry_with_model(frame, H, board, reason):
    # if diff fails, try to detect the position with the model and match it to a legal move
    print(f"  {reason} -> second attempt with the model")
    detected, outside, _ = predict_board(detector.detect(frame), H)
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


# setup cam
cap = cv.VideoCapture(config["camera"]["index"], cv.CAP_DSHOW)
try:
    configure_camera(cap, config)

    # align camera to board
    cv.namedWindow("Framing Check", cv.WINDOW_NORMAL)
    if not framing_check(cap):
        raise RuntimeError("Framing check failed")

    # get / load corners
    corners = get_corners(cap, config, override=True)
    if corners is None:
        raise RuntimeError("Failed to get corners")

    # calculate homography and grid points
    H, H_inv = homography(np.float32(corners))
    img_grid = grid_points(H_inv)

    # setup chess board
    board = chess.Board()
    game_finished = False
    print("Setup completed")

    # setup overlay
    show_grid = True
    show_pos = True
    suggestion = None
    best_score = None
    view_stack = []

    # create windows for display
    cv.namedWindow("Game Capture", cv.WINDOW_NORMAL)
    cv.namedWindow("Board View", cv.WINDOW_NORMAL)
    cv.namedWindow("Difference", cv.WINDOW_NORMAL)

    dummy = np.zeros((500, 500, 3), dtype=np.uint8)
    cv.imshow("Game Capture", dummy)
    cv.imshow("Board View", dummy)
    cv.imshow("Difference", dummy)
    cv.waitKey(1)

    cv.resizeWindow("Game Capture", 640, 480)
    cv.resizeWindow("Board View", 500, 500)
    cv.resizeWindow("Difference", 500, 500)

    cv.moveWindow("Game Capture", 0, 0)
    cv.moveWindow("Board View", 640, 0)
    cv.moveWindow("Difference", 1140, 0)

    game_ready = False
    show_start = True
    print("Setting up starting position")
    while not game_ready:
        ok, frame = cap.read()
        if not ok:
            continue

        view = frame.copy()
        if show_grid:
            draw_grid(view, img_grid)
        if show_pos:
            draw_position(view, H_inv, board)

        cv.imshow("Game Capture", view)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            game_finished = True
            break
        elif key == ord("s"):
            # switch side: white <-> black
            corners = rotate_corners(corners, 2)
            H, H_inv = homography(np.float32(corners))
            img_grid = grid_points(H_inv)
            print("Side switched")
        elif key == ord("p"):
            # check starting position without starting the game
            check_against_image(frame, H, board, "Setting up starting position")
        elif key == ord("g"):
            # check starting position and start the game
            check_against_image(frame, H, board, "Setting up starting position")

            game_ready = True
            prev_board_view = board_view(frame, H)

            print("Game started")

    # game loop
    turn = 0
    still_since = 0
    browser_opened = False
    Path("visu").mkdir(exist_ok=True)

    last_view = board_view(frame, H)
    prev_board_view = board_view(frame, H)

    while not game_finished:
        ok, frame = cap.read()
        if not ok:
            continue

        if suggestion is None:
            suggestion, best_score = get_best_move(board, config)

        view = frame.copy()
        current_view = board_view(frame, H)

        # overlays
        if show_grid:
            draw_grid(view, img_grid)
        if show_pos:
            draw_position(view, H_inv, board)

        if suggestion:
            from_sq = chess.square_name(suggestion.from_square)
            to_sq = chess.square_name(suggestion.to_square)
            draw_arrow(view, H_inv, from_sq, to_sq)
            cv.putText(
                view,
                f"Eval: {best_score}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # tracking diff for move detection
        still_since = (
            still_since + 1
            if is_still(current_view, last_view, config["movement_threshold"])
            else 0
        )
        last_view = current_view.copy()

        cv.imshow("Game Capture", view)
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            game_finished = True
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("o"):
            show_pos = not show_pos
        elif key == ord("u"):
            if len(board.move_stack) > 0:
                board.pop()
                prev_board_view = view_stack.pop()
                suggestion = None
                best_score = None
                print("undo move")
        # space for game logic, move detection, etc.
        elif key == ord(" ") or (
            still_since == config["still_frames_required"] and turn > 0
        ):
            if board.is_game_over():
                print("Game is already over (checkmate/stalemate).")
                continue

            turn = board.ply() + 1

            current_view = board_view(frame, H)
            cv.imshow("Board View", current_view)

            if prev_board_view is not None:
                diff = cv.absdiff(current_view, prev_board_view)
                cv.imshow("Difference", diff)

            diff = get_diff(current_view, prev_board_view)

            # get the move based on the diff and update the board
            active = []
            for field, value in diff.items():
                if value > config["active_field_threshold"]:
                    active.append(field)

            detected_move = None

            if len(active) > config["max_active_fields"]:
                print(
                    f"Active fields: {len(active)}, max diff: {max(diff.values()):.1f}, "
                    f"median diff: {np.median(list(diff.values())):.1f}"
                )
                detected_move = retry_with_model(
                    frame,
                    H,
                    board,
                    f"Turn {turn}: too many changes ({len(active)})",
                )
                if detected_move is None:
                    continue

            # get move scores for each legal move, filter by inventory
            valid_moves = filter_moves_by_inventory(board)

            move_scores = {}
            for move in valid_moves:
                move_scores[move] = diff_score(board, move, diff)

            # sort moves by score and promotion value - pick the best one
            def sort_key(m, move_scores=move_scores):
                promo_val = {
                    chess.QUEEN: 4,
                    chess.ROOK: 3,
                    chess.BISHOP: 2,
                    chess.KNIGHT: 1,
                }.get(m.promotion, 0)
                return (move_scores[m], promo_val)

            sorted_moves = sorted(valid_moves, key=sort_key, reverse=True)

            # if no move detected yet, pick the best scored move and check for ambiguity
            if detected_move is None:
                best_move = sorted_moves[0]

                # check if the second best move is close in score to the best move
                second_detected = None
                for m in sorted_moves[1:]:
                    if (m.from_square, m.to_square) != (
                        best_move.from_square,
                        best_move.to_square,
                    ):
                        second_detected = m
                        break

                if move_scores[best_move] < 0.1:
                    detected_move = retry_with_model(
                        frame,
                        H,
                        board,
                        f"Turn {turn}: no clear move "
                        f"(best score {move_scores[best_move]:.3f})",
                    )
                elif (
                    second_detected
                    and move_scores[best_move] - move_scores[second_detected] < 0.05
                ):
                    detected_move = retry_with_model(
                        frame,
                        H,
                        board,
                        f"Turn {turn}: ambiguous ({best_move.uci()} against "
                        f"{second_detected.uci()})",
                    )
                else:
                    detected_move = best_move

                if detected_move is None:
                    continue

            view_stack.append(prev_board_view)

            board.push(detected_move)
            print(f"Turn {turn}: Move detected: {detected_move.uci()}")
            print(f"Turn {turn}: Board FEN: {board.board_fen()}")

            visu_path = update_browser_view(board, turn, best_score)
            if not browser_opened:
                webbrowser.open(visu_path.resolve().as_uri())
                browser_opened = True

            prev_board_view = current_view.copy()
            suggestion = None
            best_score = None


except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv.destroyAllWindows()
