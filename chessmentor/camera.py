import cv2 as cv


def configure_camera(cap: cv.VideoCapture, config: dict):
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    cap.set(cv.CAP_PROP_AUTOFOCUS, config["camera"]["autofocus"])
    cap.set(cv.CAP_PROP_FOCUS, config["camera"]["focus"])

    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, config["camera"]["auto_exposure"])
    cap.set(cv.CAP_PROP_EXPOSURE, config["camera"]["exposure"])

    cap.set(cv.CAP_PROP_AUTO_WB, config["camera"]["auto_wb"])
    cap.set(cv.CAP_PROP_WB_TEMPERATURE, config["camera"]["wb_temperature"])

    cap.set(cv.CAP_PROP_FRAME_WIDTH, config["camera"]["frame_width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])

    cap.set(cv.CAP_PROP_GAIN, config["camera"]["gain"])

    # discard first frames
    for _ in range(5):
        cap.read()
