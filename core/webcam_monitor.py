import cv2

def is_webcam_active():
    cap = cv2.VideoCapture(0)
    if cap.isOpened:
        ret, _ = cap.read()
        cap.release()
        return ret
    return False