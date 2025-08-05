import time
import threading
from core.gesture_detector import run_gesture_detection

def monitor_webcam():
    print("Webcam is active")
    run_gesture_detection()

if __name__ == "__main__":
    threading.Thread(target=monitor_webcam).start()