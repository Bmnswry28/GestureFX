import time
import threading
from core.webcam_monitor import is_webcam_active
from core.gesture_detector import run_gesture_detection
from config import CHECK_INTERVAL

def monitor_webcam():
    while True:
        if is_webcam_active():
            print('webcam is active')
            run_gesture_detection()
        else:
            print('web is idle')
        time.sleep(CHECK_INTERVAL)



if __name__ == "__main__":
    threading.Thread(target=monitor_webcam).start()
    time.sleep(1)
