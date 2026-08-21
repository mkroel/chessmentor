import json
import math
from pathlib import Path

import cv2 as cv
import numpy as np

from chessmentor.board import BOARD_PX, SQUARE_PX
from chessmentor.camera import get_frame


def pick_corners(cap: cv.VideoCapture, config: dict):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera")

    title = "Click the Corners"
    cv.putText(
        frame,
        title,
        (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        2,
    )
    subheading = "Clockwise order starting from top left"
    cv.putText(
        frame,
        subheading,
        (frame.shape[1] // 2 - 200, frame.shape[0] // 2 + 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv.imshow("click Corners", frame)

    # callback function to get mouse click coordinates
    set_corners = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            set_corners.append((x, y))
            print(f"Corner {len(set_corners)}: ({x}, {y})")
            cv.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv.imshow("click Corners", frame)
            if len(set_corners) == 4:
                cv.destroyAllWindows()

    cv.setMouseCallback("click Corners", mouse_callback)

    # wait for 4 clicks
    while len(set_corners) < 4:
        cv.waitKey(1)

    # write corners to config
    data = {"corners": set_corners, "width": frame.shape[1], "height": frame.shape[0]}

    with Path(config["corners_file"]).open("w") as f:
        json.dump(data, f, indent=4)

    return set_corners, frame


def load_corners(config: dict):
    with Path(config["corners_file"]).open("r") as f:
        data = json.load(f)
    return data["corners"]


DETECT_WIDTHS = (800, 640, 560)


def _detect_scaled(gray, cols, rows):
    for max_width in DETECT_WIDTHS:
        scale = max(1.0, gray.shape[1] / max_width)
        small = cv.resize(
            gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv.INTER_AREA
        )
        # try both the original and enhanced images
        variants = (small, cv.createCLAHE(2.0, (8, 8)).apply(small))
        for variant in variants:
            ret, corners = cv.findChessboardCornersSB(
                variant, (cols, rows), cv.CALIB_CB_ACCURACY
            )
            if ret:
                return corners, small
    return None, None


def find_corners(frame, config: dict):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    rows, cols = 7, 7

    corners, small = _detect_scaled(gray, cols, rows)
    if corners is None:
        print("Chessboard not detected")
        return None

    img_pts = corners.reshape(-1, 2).astype(np.float32)
    img_pts *= np.float32(
        [gray.shape[1] / small.shape[1], gray.shape[0] / small.shape[0]]
    )

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_pts = cv.cornerSubPix(
        gray, img_pts.reshape(-1, 1, 2).copy(), (11, 11), (-1, -1), criteria
    ).reshape(-1, 2)

    ideal = []
    for row in range(rows):
        for col in range(cols):
            board_x = (col + 1) * SQUARE_PX
            board_y = (row + 1) * SQUARE_PX
            ideal.append([board_x, board_y])

    ideal = np.array(ideal, dtype=np.float32)

    H, _ = cv.findHomography(img_pts, ideal, 0)
    if H is None:
        print("Failed to compute homography")
        return None

    proj = cv.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    error = float(np.linalg.norm(proj - ideal, axis=1).mean())

    outer_2d = np.float32(
        [
            [0, 0],
            [BOARD_PX, 0],
            [BOARD_PX, BOARD_PX],
            [0, BOARD_PX],
        ]
    )
    outer = outer_2d.reshape(4, 1, 2)

    tl, tr, br, bl = cv.perspectiveTransform(outer, np.linalg.inv(H)).reshape(-1, 2)
    return [tl, tr, br, bl], error


def compare_corners(detected: list, loaded: list, config: dict):
    threshold = config.get("corner_threshold", 10)
    for i, (d, ln) in enumerate(zip(detected, loaded, strict=True)):
        dist = math.dist(d, ln)
        if dist > threshold:
            print(f"Corner {i}: {dist:.1f} px off (threshold {threshold})")
            return False
    return True


def save_corners(corners: list, config: dict):
    data = {"corners": np.asarray(corners).tolist()}
    with Path(config["corners_file"]).open("w") as f:
        json.dump(data, f, indent=4)


def get_corners(cap: cv.VideoCapture, config: dict, override: bool = False):
    frame = get_frame(cap)
    found = find_corners(frame, config)

    stored = load_corners(config) if Path(config["corners_file"]).exists() else None

    if found is not None:
        detected, error = found

        # no reference: save detected corners as new reference
        if stored is None:
            print(f"Corners detected ({error:.2f} px) - saved as new reference")
            save_corners(detected, config)
            return detected

        # detected corners are the same as stored corners, but with a smaller error, refresh the reference
        if compare_corners(detected, stored, config):
            print(f"Corners detected ({error:.2f} px) - reference refreshed")
            save_corners(detected, config)
            return detected

        # corners differ: board moved, or detection failed
        print("Detected corners differ from reference - board moved?")
        if override:
            print("override=True - reference overwritten")
            save_corners(detected, config)
            return detected

    print("No reference available - please click the corners")
    corners, _ = pick_corners(cap, config)
    return corners
