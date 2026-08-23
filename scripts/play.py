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
from chessmentor.camera import configure_camera, get_corners, pick_corners
from chessmentor.detect import Detector
from chessmentor.engine import get_best_move
from chessmentor.game import check_against_image, detect_played_move
from chessmentor.render import (
    draw_arrow,
    draw_grid,
    draw_position,
    run_setup_menu,
    setup_windows,
    update_browser_view,
)
from chessmentor.vision import board_view, framing_check, is_still


def setup_phase(cap, config, corners, board, detector):
    # calculate homography and grid points
    H, H_inv = homography(np.float32(corners))
    img_grid = grid_points(H_inv)

    show_grid = True
    show_pos = True
    game_ready = False
    prev_board_view = None

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
            return True, corners, H, H_inv, img_grid, prev_board_view
        elif key == ord("s"):
            # switch side: white <-> black
            corners = rotate_corners(corners, 2)
            H, H_inv = homography(np.float32(corners))
            img_grid = grid_points(H_inv)
            print("Side switched")
        elif key == ord("p"):
            # check starting position without starting the game
            check_against_image(
                frame, H, board, "Setting up starting position", detector
            )
        elif key == ord("g"):
            # check starting position and start the game
            check_against_image(
                frame, H, board, "Setting up starting position", detector
            )
            game_ready = True
            prev_board_view = board_view(frame, H)
            print("Game started")

    return False, corners, H, H_inv, img_grid, prev_board_view


def game_phase(
    cap, config, board, H, H_inv, img_grid, prev_board_view, detector, players
):
    game_finished = False
    show_grid = True
    show_pos = True
    suggestion = None
    best_score = None
    view_stack = []

    turn = 0
    still_since = 0
    browser_opened = False
    Path("visu").mkdir(exist_ok=True)

    ok, frame = cap.read()
    last_view = board_view(frame, H) if ok else prev_board_view

    # game loop
    while not game_finished:
        ok, frame = cap.read()
        if not ok:
            continue

        current_player = players[board.turn]
        is_engine_turn = current_player["type"] == "engine"

        if suggestion is None and not board.is_game_over():
            suggestion, best_score = get_best_move(
                board, config, current_player["skill"]
            )

        view = frame.copy()
        current_view = board_view(frame, H)

        cv.imshow("Board View", current_view)

        # overlays
        if show_grid:
            draw_grid(view, img_grid)
        if show_pos:
            draw_position(view, H_inv, board)

        if suggestion:
            from_sq = chess.square_name(suggestion.from_square)
            to_sq = chess.square_name(suggestion.to_square)

            if is_engine_turn:
                draw_arrow(view, H_inv, from_sq, to_sq, color=(255, 0, 0))  # Blue
            elif current_player["mentor"]:
                draw_arrow(view, H_inv, from_sq, to_sq, color=(0, 255, 0))  # Green

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
            if len(board.move_stack) > 0 and len(view_stack) > 0:
                board.pop()
                prev_board_view = view_stack.pop()
                suggestion = None
                best_score = None
                print("undo move")
        elif key == ord("c"):
            print("Re-picking corners...")
            new_corners, _ = pick_corners(cap, config)
            if new_corners and len(new_corners) == 4:
                H, H_inv = homography(np.float32(new_corners))
                img_grid = grid_points(H_inv)
                current_view = board_view(frame, H)
                prev_board_view = current_view.copy()
                last_view = current_view.copy()
                view_stack.clear()
                print("Corners updated successfully.")
        # space for game logic, move detection, etc.
        elif key == ord(" ") or (
            still_since == config["still_frames_required"] and turn > 0
        ):
            if board.is_game_over():
                print("Game is already over (checkmate/stalemate).")
                continue

            turn = board.ply() + 1

            if prev_board_view is not None:
                cv.imshow("Difference", cv.absdiff(current_view, prev_board_view))

            detected_move = detect_played_move(
                frame, current_view, prev_board_view, board, H, config, detector, turn
            )

            if detected_move:
                if is_engine_turn:
                    if detected_move == suggestion:
                        view_stack.append(prev_board_view)
                        board.push(detected_move)
                        print(
                            f"Turn {turn}: Engine move executed: {detected_move.uci()}"
                        )
                    else:
                        print(
                            f"Wrong piece moved! Please execute engine move {suggestion.uci()}."
                        )
                        continue
                else:
                    view_stack.append(prev_board_view)
                    board.push(detected_move)
                    print(f"Turn {turn}: Move detected: {detected_move.uci()}")
            else:
                continue

            print(f"Turn {turn}: Board FEN: {board.board_fen()}")

            visu_path = update_browser_view(board, turn, best_score)
            if not browser_opened:
                webbrowser.open(visu_path.resolve().as_uri())
                browser_opened = True

            prev_board_view = current_view.copy()
            suggestion = None
            best_score = None


def main():
    # load config
    with Path("config.yaml").open() as f:
        config = yaml.safe_load(f)

    setup_windows()

    # load detector model
    det_cfg = config["detector"]
    detector = Detector(
        det_cfg["weights_path"],
        conf=det_cfg["confidence_threshold"],
        imgsz=det_cfg["imgsz"],
    )
    print("Detector loaded")

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

        players = run_setup_menu(cap)

        # setup chess board
        board = chess.Board()
        print("Setup completed")

        game_finished, corners, H, H_inv, img_grid, prev_board_view = setup_phase(
            cap, config, corners, board, detector
        )

        if not game_finished:
            game_phase(
                cap,
                config,
                board,
                H,
                H_inv,
                img_grid,
                prev_board_view,
                detector,
                players,
            )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
