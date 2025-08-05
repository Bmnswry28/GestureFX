import os
import json
import cv2
import numpy as np

with open("config.json", "r") as f:
    config = json.load(f)

EFFECTS = config["effects"]

def load_effect(name):
    path = os.path.join('assets', EFFECTS.get(name, ''))
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None

def apply_effect_at(frame, effect, x, y, size=100):
    # بررسی اینکه افکت خالی نباشه
    if effect is None:
        print("❌ افکت بارگذاری نشده یا مسیر اشتباهه")
        return frame

    # تبدیل به BGRA اگر فقط ۳ کانال داشت
    if effect.shape[2] == 3:
        effect = cv2.cvtColor(effect, cv2.COLOR_BGR2BGRA)

    # تغییر اندازه افکت
    effect_resized = cv2.resize(effect, (size, size))

    # جدا کردن کانال‌ها
    channels = cv2.split(effect_resized)
    if len(channels) == 4:
        b, g, r, a = channels
    else:
        b, g, r = channels
        a = np.ones_like(b) * 255  # آلفای کامل

    # ساختن ماسک آلفا
    alpha_mask = a / 255.0
    alpha_inv = 1.0 - alpha_mask

    # موقعیت قرارگیری افکت روی فریم
    h, w = frame.shape[:2]
    x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
    x2, y2 = min(w, x1 + size), min(h, y1 + size)

    # برش فریم اصلی
    roi = frame[y1:y2, x1:x2]

    # برش افکت به اندازه ROI
    effect_crop = effect_resized[0:(y2 - y1), 0:(x2 - x1)]
    alpha_mask = alpha_mask[0:(y2 - y1), 0:(x2 - x1)]
    alpha_inv = alpha_inv[0:(y2 - y1), 0:(x2 - x1)]

    # ترکیب افکت با فریم
    for c in range(3):
        roi[:, :, c] = (alpha_inv * roi[:, :, c] + alpha_mask * effect_crop[:, :, c]).astype(np.uint8)

    # جایگزینی ROI در فریم اصلی
    frame[y1:y2, x1:x2] = roi
    return frame
