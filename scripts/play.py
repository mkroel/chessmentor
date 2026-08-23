# imports
import webbrowser
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
import yaml

from chessmentor.board import grid_points, homography, rotate_corners
from chessmentor.camera import configure_camera
from chessmentor.corners import get_corners
from chessmentor.engine import get_best_move
from chessmentor.game import diff_score, filter_moves_by_inventory
from chessmentor.render import draw_arrow, draw_grid, draw_position, update_browser_view
from chessmentor.vision import board_view, framing_check, get_diff, is_still

# load config
with Path("config.yaml").open() as f:
    config = yaml.safe_load(f)


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

    game_ready = False
    show_start = True
    print("Startaufstellung aufbauen")
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
            print("Seite gewechselt")
        elif key == ord("g"):
            game_ready = True
            prev_board_view = board_view(frame, H)

            print("Spiel gestartet")

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
                print("Partie ist bereits beendet (Matt/Patt).")
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
                if value > 5.0:  # threshold for detecting a change
                    active.append(field)

            if len(active) > 6:
                print(f"Turn {turn}: Too many changes detected, ignoring.")
                print(
                    f"Active fields: {active}, max diff: {max(diff.values())}, median diff: {np.median(list(diff.values()))}"
                )
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
            detected_move = sorted_moves[0]

            if move_scores[detected_move] < 0.1:
                print(f"Turn {turn}: No valid move detected, ignoring.")
                continue

            # check for ambiguity: if the second best move has a similar score, ignore the detection
            second_detected = None
            for m in sorted_moves[1:]:
                if (m.from_square, m.to_square) != (
                    detected_move.from_square,
                    detected_move.to_square,
                ):
                    second_detected = m
                    break

            if (
                second_detected
                and move_scores[detected_move] - move_scores[second_detected] < 0.05
            ):
                print(f"Turn {turn}: Ambiguous move detected, ignoring.")
                continue

            view_stack.append(prev_board_view)

            board.push(detected_move)
            print(f"Turn {turn}: Move detected: {detected_move.uci()}")
            print(f"Turn {turn}: Board FEN: {board.board_fen()}")

            visu_path = update_browser_view(board, turn)
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
