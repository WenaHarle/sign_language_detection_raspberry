"""
sign_collector.py  —  MediaPipe Tasks API (v0.10.30+)
──────────────────────────────────────────────────────
Simpan sudut landmark tangan ke gestures.json.

Kontrol:
  S  → simpan sudut tangan saat ini (lalu ketik nama)
  Q  → keluar

Setup awal (WAJIB):
  Download model hand_landmarker.task lalu taruh di folder ini.
  Link: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  Atau jalankan perintah ini di terminal:
  curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "hand_landmarker.task"
OUTPUT_FILE = "gestures.json"

# ── Joint definition ──────────────────────────────────────────────────────────
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
    "ring_pip",  "ring_dip",
    "pinky_pip", "pinky_dip",
    "knuckle_1_2", "knuckle_2_3", "knuckle_3_4",
    "wrist_ref1",  "wrist_ref2",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    ba, bc  = a - b, c - b
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

    wrist      = pts[0]
    index_mcp  = pts[5]
    middle_mcp = pts[9]
    pinky_mcp  = pts[17]

    palm_forward = normalize_vec(middle_mcp - wrist)
    palm_side    = normalize_vec(index_mcp - pinky_mcp)
    palm_normal  = normalize_vec(np.cross(palm_side, palm_forward))

    # Re-orthogonalize the basis so matching stays stable frame-to-frame.
    palm_side    = normalize_vec(np.cross(palm_forward, palm_normal))
    palm_normal  = normalize_vec(np.cross(palm_side, palm_forward))

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

def draw_landmarks_manual(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 5, (0, 220, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)

# ── UI ────────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_SIMPLEX
WHITE  = (255, 255, 255)
GREEN  = (0, 210,  80)
YELLOW = (0, 220, 220)
RED    = (60,  60, 220)
DARK   = (30,  30,  30)

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

def draw_panel(frame, gestures, mode, input_name, status_msg, status_color):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 52), DARK, -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Sign Collector  |  MediaPipe", (10, 32), FONT, 0.75, WHITE, 2)
    cv2.putText(frame, f"Saved: {len(gestures)}", (w - 150, 32), FONT, 0.6, GREEN, 1)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 100), (w, h), DARK, -1)
    cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)

    if mode == "naming":
        cv2.putText(frame, f"Nama: {input_name}_",
                    (10, h - 65), FONT, 0.80, YELLOW, 2)
        cv2.putText(frame, "ENTER = simpan   ESC = batal",
                    (10, h - 28), FONT, 0.55, WHITE, 1)
    else:
        cv2.putText(frame, "S = simpan gesture   Q = keluar",
                    (10, h - 65), FONT, 0.60, WHITE, 1)
        names   = list(gestures.keys())[-5:]
        preview = "  |  ".join(names) if names else "(belum ada)"
        cv2.putText(frame, f"Gestures: {preview}",
                    (10, h - 28), FONT, 0.50, GREEN, 1)

    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 120),
                    FONT, 0.70, status_color, 2)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print("=" * 65)
        print(" ERROR: File model tidak ditemukan!")
        print(f" Dibutuhkan: {MODEL_PATH}")
        print()
        print(" Download dengan perintah ini di terminal:")
        print()
        print(" curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print()
        print(" Taruh file di folder yang sama dengan script ini.")
        print("=" * 65)
        return

    gestures       = load_gestures()
    mode           = "live"
    input_name     = ""
    pending_sample = None
    status_msg     = ""
    status_color   = GREEN
    status_timer   = 0

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options      = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    cap, backend = open_camera(0)
    if cap is None:
        print("=" * 65)
        print(" ERROR: Webcam tidak bisa dibuka.")
        print(" Coba tutup aplikasi lain yang sedang memakai kamera.")
        print(" Jika ada beberapa kamera, ubah index di open_camera(0) menjadi 1 atau 2.")
        print("=" * 65)
        return

    print(f"Sign Collector siap. Output: {OUTPUT_FILE}")
    print(f"Camera backend: {backend}")
    print("S = simpan | Q = keluar")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        last_ts_ms = -1
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                status_msg = "Frame kamera gagal dibaca"
                status_color = RED
                print("Warning: frame kamera gagal dibaca, collector dihentikan.")
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            result           = landmarker.detect_for_video(mp_image, ts_ms)
            current_angles   = None
            current_rotation = None

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_landmarks_manual(frame, lm)
                current_angles = extract_angles(lm)
                current_rotation = extract_rotation(lm)
                h, w = frame.shape[:2]
                for i, (k, v) in enumerate(list(current_angles.items())[:7]):
                    cv2.putText(frame, f"{k}: {v:.1f}",
                                (w - 215, 65 + i * 20), FONT, 0.42, YELLOW, 1)
                rot_preview = [
                    ("fwd_z", current_rotation["palm_forward_z"]),
                    ("nrm_z", current_rotation["palm_normal_z"]),
                    ("side_x", current_rotation["palm_side_x"]),
                ]
                for i, (k, v) in enumerate(rot_preview):
                    cv2.putText(frame, f"{k}: {v:+.2f}",
                                (w - 215, 225 + i * 20), FONT, 0.42, GREEN, 1)
            else:
                cv2.putText(frame, "Tangan tidak terdeteksi",
                            (10, 90), FONT, 0.65, RED, 2)

            if status_timer > 0:
                status_timer -= 1
            else:
                status_msg = ""

            draw_panel(frame, gestures, mode, input_name, status_msg, status_color)
            cv2.imshow("Sign Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if mode == "live":
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if current_angles and current_rotation:
                        pending_sample = {
                            "angles": current_angles,
                            "rotation": current_rotation,
                        }
                        mode       = "naming"
                        input_name = ""
                    else:
                        status_msg   = "Tidak ada tangan terdeteksi!"
                        status_color = RED
                        status_timer = 60

            elif mode == "naming":
                if key == 27:
                    mode = "live"; input_name = ""; pending_sample = None
                elif key == 13:
                    name = input_name.strip()
                    if not name:
                        status_msg   = "Nama tidak boleh kosong!"
                        status_color = RED
                        status_timer = 80
                    else:
                        gestures[name] = {
                            "label"    : name,
                            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "angles"   : pending_sample["angles"],
                            "rotation" : pending_sample["rotation"],
                        }
                        save_gestures(gestures)
                        print(f"  Tersimpan: '{name}'")
                        status_msg   = f"'{name}' berhasil disimpan!"
                        status_color = GREEN
                        status_timer = 90
                        mode = "live"; input_name = ""; pending_sample = None
                elif key == 8:
                    input_name = input_name[:-1]
                elif 32 <= key <= 126:
                    input_name += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Total: {len(gestures)} gesture(s) tersimpan.")

if __name__ == "__main__":
    main()
