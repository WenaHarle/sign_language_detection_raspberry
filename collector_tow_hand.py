"""
two_hand_collector.py - MediaPipe Tasks API (v0.10.30+)
-------------------------------------------------------
Menyimpan gesture dua tangan ke gestures_two_hand.json.

Kontrol:
  S  -> simpan pose dua tangan saat ini
  Q  -> keluar
"""

import json
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = "hand_landmarker.task"
OUTPUT_FILE = "gestures_two_hand.json"

JOINT_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (0, 5, 9), (0, 9, 13), (0, 13, 17),
    (1, 0, 17), (5, 0, 17),
]
JOINT_NAMES = [
    "thumb_pip", "thumb_dip",
    "index_pip", "index_dip",
    "middle_pip", "middle_dip",
    "ring_pip", "ring_dip",
    "pinky_pip", "pinky_dip",
    "knuckle_1_2", "knuckle_2_3", "knuckle_3_4",
    "wrist_ref1", "wrist_ref2",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 210, 80)
YELLOW = (0, 220, 220)
RED = (60, 60, 220)
DARK = (30, 30, 30)
CYAN = (210, 180, 0)

HAND_COLORS = {
    "Left": ((80, 200, 80), (0, 220, 255)),
    "Right": ((255, 140, 0), (255, 220, 120)),
}


def load_gestures():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: {OUTPUT_FILE} bukan JSON valid, pakai data kosong.")
            return {}
    return {}


def save_gestures(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def normalize_vec(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return vec / norm


def extract_angles(landmarks):
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return {
        name: round(compute_angle(pts[a], pts[b], pts[c]), 2)
        for name, (a, b, c) in zip(JOINT_NAMES, JOINT_TRIPLETS)
    }


def extract_rotation(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    pinky_mcp = pts[17]

    palm_forward = normalize_vec(middle_mcp - wrist)
    palm_side = normalize_vec(index_mcp - pinky_mcp)
    palm_normal = normalize_vec(np.cross(palm_side, palm_forward))

    palm_side = normalize_vec(np.cross(palm_forward, palm_normal))
    palm_normal = normalize_vec(np.cross(palm_side, palm_forward))

    return {
        "palm_forward_x": round(float(palm_forward[0]), 4),
        "palm_forward_y": round(float(palm_forward[1]), 4),
        "palm_forward_z": round(float(palm_forward[2]), 4),
        "palm_side_x": round(float(palm_side[0]), 4),
        "palm_side_y": round(float(palm_side[1]), 4),
        "palm_side_z": round(float(palm_side[2]), 4),
        "palm_normal_x": round(float(palm_normal[0]), 4),
        "palm_normal_y": round(float(palm_normal[1]), 4),
        "palm_normal_z": round(float(palm_normal[2]), 4),
    }


def get_handed_label(handedness_item):
    if not handedness_item:
        return None
    category = handedness_item[0]
    label = getattr(category, "category_name", None) or getattr(category, "display_name", None)
    if label in ("Left", "Right"):
        return label
    return None


def build_two_hand_sample(result):
    hands = {}
    wrists = {}

    if not result.hand_landmarks:
        return None

    for idx, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[idx] if idx < len(result.handedness) else None
        hand_label = get_handed_label(handedness)
        if hand_label not in ("Left", "Right"):
            hand_label = "Left" if "Left" not in hands else "Right"

        hands[hand_label] = {
            "angles": extract_angles(landmarks),
            "rotation": extract_rotation(landmarks),
        }
        wrist = landmarks[0]
        wrists[hand_label] = np.array([wrist.x, wrist.y, wrist.z], dtype=np.float32)

    if "Left" not in hands or "Right" not in hands:
        return None

    wrist_delta = wrists["Right"] - wrists["Left"]
    hand_distance = np.linalg.norm(wrist_delta)
    relation = {
        "wrist_dx": round(float(wrist_delta[0]), 4),
        "wrist_dy": round(float(wrist_delta[1]), 4),
        "wrist_dz": round(float(wrist_delta[2]), 4),
        "wrist_distance": round(float(hand_distance), 4),
    }

    return {
        "left_hand": hands["Left"],
        "right_hand": hands["Right"],
        "relation": relation,
    }


def draw_landmarks_manual(frame, landmarks, hand_label):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    line_color, point_color = HAND_COLORS.get(hand_label, ((80, 200, 80), (0, 220, 255)))
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], line_color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, point_color, -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
    cv2.putText(frame, hand_label, (pts[0][0] + 8, pts[0][1] - 8), FONT, 0.55, line_color, 2)


def open_camera(camera_index=0):
    candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in candidates:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, backend
        cap.release()
    return None, None


def draw_panel(frame, gestures, mode, input_name, status_msg, status_color, hand_state):
    h, w = frame.shape[:2]
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, 56), DARK, -1)
    cv2.addWeighted(top, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Two-Hand Collector  |  MediaPipe", (10, 34), FONT, 0.75, WHITE, 2)
    cv2.putText(frame, f"Saved: {len(gestures)}", (w - 145, 34), FONT, 0.60, GREEN, 1)

    bottom = frame.copy()
    cv2.rectangle(bottom, (0, h - 104), (w, h), DARK, -1)
    cv2.addWeighted(bottom, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Deteksi: {hand_state}", (10, 84), FONT, 0.60, CYAN, 2)

    if mode == "naming":
        cv2.putText(frame, f"Nama: {input_name}_", (10, h - 68), FONT, 0.80, YELLOW, 2)
        cv2.putText(frame, "ENTER = simpan   ESC = batal", (10, h - 28), FONT, 0.55, WHITE, 1)
    else:
        cv2.putText(frame, "S = simpan gesture dua tangan   Q = keluar", (10, h - 68), FONT, 0.56, WHITE, 1)
        names = list(gestures.keys())[-4:]
        preview = "  |  ".join(names) if names else "(belum ada)"
        cv2.putText(frame, f"Gestures: {preview}", (10, h - 28), FONT, 0.50, GREEN, 1)

    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 124), FONT, 0.70, status_color, 2)


def main():
    if not os.path.exists(MODEL_PATH):
        print("=" * 65)
        print(" ERROR: File model tidak ditemukan!")
        print(f" Dibutuhkan: {MODEL_PATH}")
        print("=" * 65)
        return

    gestures = load_gestures()
    mode = "live"
    input_name = ""
    pending_sample = None
    status_msg = ""
    status_color = GREEN
    status_timer = 0

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    cap, backend = open_camera(0)
    if cap is None:
        print("=" * 65)
        print(" ERROR: Webcam tidak bisa dibuka.")
        print("=" * 65)
        return

    print(f"Two-Hand Collector siap. Output: {OUTPUT_FILE}")
    print(f"Camera backend: {backend}")
    print("S = simpan | Q = keluar")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        last_ts_ms = -1
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Warning: frame kamera gagal dibaca, collector dihentikan.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 33
            if ts_ms <= 0:
                ts_ms = frame_idx * 33
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, ts_ms)
            current_sample = build_two_hand_sample(result)
            hand_labels = []

            if result.hand_landmarks:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx] if idx < len(result.handedness) else None
                    hand_label = get_handed_label(handedness) or f"Hand{idx + 1}"
                    hand_labels.append(hand_label)
                    draw_landmarks_manual(frame, landmarks, hand_label)

            hand_state = " + ".join(hand_labels) if hand_labels else "tidak ada tangan"
            if current_sample is None:
                cv2.putText(frame, "Butuh dua tangan (Left + Right) untuk menyimpan",
                            (10, 116), FONT, 0.62, RED, 2)
            else:
                rel = current_sample["relation"]
                cv2.putText(frame, f"dx:{rel['wrist_dx']:+.2f} dy:{rel['wrist_dy']:+.2f} dz:{rel['wrist_dz']:+.2f}",
                            (10, 116), FONT, 0.50, YELLOW, 1)

            if status_timer > 0:
                status_timer -= 1
            else:
                status_msg = ""

            draw_panel(frame, gestures, mode, input_name, status_msg, status_color, hand_state)
            cv2.imshow("Two-Hand Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if mode == "live":
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if current_sample is not None:
                        pending_sample = current_sample
                        mode = "naming"
                        input_name = ""
                    else:
                        status_msg = "Dua tangan belum terdeteksi lengkap"
                        status_color = RED
                        status_timer = 75
            else:
                if key == 27:
                    mode = "live"
                    input_name = ""
                    pending_sample = None
                elif key == 13:
                    name = input_name.strip()
                    if not name:
                        status_msg = "Nama tidak boleh kosong!"
                        status_color = RED
                        status_timer = 80
                    else:
                        gestures[name] = {
                            "label": name,
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            **pending_sample,
                        }
                        save_gestures(gestures)
                        print(f"  Tersimpan: '{name}'")
                        status_msg = f"'{name}' berhasil disimpan!"
                        status_color = GREEN
                        status_timer = 90
                        mode = "live"
                        input_name = ""
                        pending_sample = None
                elif key == 8:
                    input_name = input_name[:-1]
                elif 32 <= key <= 126:
                    input_name += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Total: {len(gestures)} gesture(s) tersimpan.")


if __name__ == "__main__":
    main()
