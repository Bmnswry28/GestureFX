import cv2
import mediapipe as mp
import pyvirtualcam
from core.effects import load_effect, apply_effect_at

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < thumb_ip.y and thumb_tip.y < index_mcp.y

def run_gesture_detection():
    cap = cv2.VideoCapture(0)
    effect = load_effect("like")

    with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
        print(f"🎥 Virtual camera started: {cam.device}")

        with mp_hands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if is_thumb_up(hand_landmarks):
                            h, w = frame.shape[:2]
                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            x = int(thumb_tip.x * w)
                            y = int(thumb_tip.y * h)
                            frame = apply_effect_at(frame, effect, x, y)

                cv2.imshow("GestureFX", frame)
                cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()