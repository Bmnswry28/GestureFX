from config import GESTURE_SENSITIVITY
import mediapipe as mp

def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]


    dist = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5
    if dist < GESTURE_SENSITIVITY:
        return "heart"


    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "like"

    return None