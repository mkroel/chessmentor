import json
from pathlib import Path

import cv2 as cv


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
