import cv2
import mediapipe as mp
from core.effect_trigger import trigger_effect
from utils.gesture_utils import detect_gesture

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def run_gesture_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                gesture = detect_gesture(handLms)
                if gesture:
                    trigger_effect(gesture)
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("GestureFX", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()