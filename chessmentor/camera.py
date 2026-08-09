import cv2 as cv


def configure_camera(cap: cv.VideoCapture, config: dict):
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    cam = config["camera"]

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cam["frame_width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam["frame_height"])

    cap.set(cv.CAP_PROP_AUTOFOCUS, cam["autofocus"])
    cap.set(cv.CAP_PROP_FOCUS, cam["focus"])

    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, cam["auto_exposure"])
    cap.set(cv.CAP_PROP_EXPOSURE, cam["exposure"])

    cap.set(cv.CAP_PROP_AUTO_WB, cam["auto_wb"])
    cap.set(cv.CAP_PROP_WB_TEMPERATURE, cam["wb_temperature"])

    cap.set(cv.CAP_PROP_GAIN, cam["gain"])


def get_frame(cap: cv.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera")
    return frame
