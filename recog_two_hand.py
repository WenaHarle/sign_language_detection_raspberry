"""
two_hand_recognizer.py - MediaPipe Tasks API (v0.10.30+)
--------------------------------------------------------
Membaca gestures_two_hand.json dan mengenali gesture dua tangan.

Kontrol:
  R  -> reload gestures_two_hand.json
  Q  -> keluar
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = "hand_landmarker.task"
GESTURES_FILE = "gestures_two_hand.json"

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
ROTATION_NAMES = [
    "palm_forward_x", "palm_forward_y", "palm_forward_z",
    "palm_side_x", "palm_side_y", "palm_side_z",
    "palm_normal_x", "palm_normal_y", "palm_normal_z",
]
RELATION_NAMES = ["wrist_dx", "wrist_dy", "wrist_dz", "wrist_distance"]

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


def load_gestures():
    if not os.path.exists(GESTURES_FILE):
        return {}
    try:
        with open(GESTURES_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: {GESTURES_FILE} bukan JSON valid, pakai data kosong.")
        return {}


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
    relation = {
        "wrist_dx": round(float(wrist_delta[0]), 4),
        "wrist_dy": round(float(wrist_delta[1]), 4),
        "wrist_dz": round(float(wrist_delta[2]), 4),
        "wrist_distance": round(float(np.linalg.norm(wrist_delta)), 4),
    }

    return {
        "left_hand": hands["Left"],
        "right_hand": hands["Right"],
        "relation": relation,
    }


def compare_hand(live_hand, saved_hand, angle_threshold=22.0, rotation_threshold=0.32):
    angle_diffs = {
        name: abs(live_hand["angles"][name] - saved_hand["angles"][name])
        for name in JOINT_NAMES
        if name in live_hand["angles"] and name in saved_hand["angles"]
    }
    rotation_diffs = {
        name: abs(live_hand["rotation"][name] - saved_hand["rotation"][name])
        for name in ROTATION_NAMES
        if name in live_hand["rotation"] and name in saved_hand["rotation"]
    }

    angle_score = ((sum(angle_diffs.values()) / len(angle_diffs)) / angle_threshold) if angle_diffs else 0.0
    rotation_score = ((sum(rotation_diffs.values()) / len(rotation_diffs)) / rotation_threshold) if rotation_diffs else 0.0
    weight_sum = 1.0 + (0.85 if rotation_diffs else 0.0)
    combined = (angle_score + (0.85 * rotation_score if rotation_diffs else 0.0)) / weight_sum

    detail = {
        **{k: round(v, 2) for k, v in angle_diffs.items()},
        **{k: round(v, 3) for k, v in rotation_diffs.items()},
    }
    return combined, detail


def match_gesture(live_sample, gestures, relation_threshold=0.20):
    if not live_sample or not gestures:
        return None, 0.0, {}

    best_label = None
    best_score = float("inf")
    best_detail = {}

    for label, entry in gestures.items():
        saved_left = entry.get("left_hand")
        saved_right = entry.get("right_hand")
        saved_relation = entry.get("relation", {})
        if not saved_left or not saved_right:
            continue

        left_score, left_detail = compare_hand(live_sample["left_hand"], saved_left)
        right_score, right_detail = compare_hand(live_sample["right_hand"], saved_right)

        relation_diffs = {
            name: abs(live_sample["relation"][name] - saved_relation[name])
            for name in RELATION_NAMES
            if name in live_sample["relation"] and name in saved_relation
        }
        relation_score = ((sum(relation_diffs.values()) / len(relation_diffs)) / relation_threshold) if relation_diffs else 0.0

        combined_score = (left_score + right_score + (0.75 * relation_score if relation_diffs else 0.0)) / (
            2.0 + (0.75 if relation_diffs else 0.0)
        )
        detail = {
            **{f"L_{k}": v for k, v in left_detail.items()},
            **{f"R_{k}": v for k, v in right_detail.items()},
            **{f"REL_{k}": round(v, 3) for k, v in relation_diffs.items()},
        }

        if combined_score < best_score:
            best_label = label
            best_score = combined_score
            best_detail = detail

    if best_label is None:
        return "???", 0.0, {}

    confidence = max(0.0, (1.0 - best_score)) * 100.0
    label_out = best_label if best_score <= 1.0 else "???"
    return label_out, confidence, best_detail


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


def conf_color(confidence):
    return GREEN if confidence >= 75 else (YELLOW if confidence >= 45 else RED)


def draw_ui(frame, label, confidence, detail, gestures, hand_state):
    h, w = frame.shape[:2]

    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, 52), DARK, -1)
    cv2.addWeighted(top, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Two-Hand Recognizer  |  MediaPipe", (10, 34), FONT, 0.72, WHITE, 2)
    cv2.putText(frame, f"{len(gestures)} gesture  |  R=reload  Q=quit", (w - 290, 34), FONT, 0.50, CYAN, 1)

    cv2.putText(frame, f"Deteksi: {hand_state}", (10, 86), FONT, 0.62, CYAN, 2)
    if hand_state != "Left + Right":
        cv2.putText(frame, "Butuh dua tangan lengkap untuk recognizer", (10, 112), FONT, 0.62, RED, 2)
        return

    pred = frame.copy()
    cv2.rectangle(pred, (10, 125), (w - 10, 245), DARK, -1)
    cv2.addWeighted(pred, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 125), (w - 10, 245), conf_color(confidence), 2)

    shown_label = label if label else "???"
    cv2.putText(frame, shown_label, (24, 205), FONT, 2.1, conf_color(confidence), 3)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (24, 234), FONT, 0.62, WHITE, 1)

    bx, by, bw = 210, 215, w - 230
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * confidence / 100), by + 14), conf_color(confidence), -1)

    if detail:
        top_detail = sorted(detail.items(), key=lambda item: item[1], reverse=True)[:12]
        px, py = w - 230, 260
        panel_height = len(top_detail) * 19 + 6
        cv2.rectangle(frame, (px - 5, py - 18), (w - 5, py + panel_height), DARK, -1)
        cv2.putText(frame, "Selisih fitur", (px, py - 2), FONT, 0.42, CYAN, 1)

        for i, (name, diff) in enumerate(top_detail):
            is_rotation = "palm_" in name
            is_relation = name.startswith("REL_")
            good_limit = 0.08 if (is_rotation or is_relation) else 10.0
            warn_limit = 0.18 if (is_rotation or is_relation) else 20.0
            color = GREEN if diff < good_limit else (YELLOW if diff < warn_limit else RED)
            scale = 180 if (is_rotation or is_relation) else 3
            bar = min(int(diff * scale), 75)
            cv2.rectangle(frame, (px, py + i * 19 + 5), (px + bar, py + i * 19 + 14), color, -1)
            diff_text = f"{diff:>5.3f}" if (is_rotation or is_relation) else f"{diff:>5.1f}"
            cv2.putText(frame, f"{name[:15]:<15} {diff_text}", (px, py + i * 19 + 18), FONT, 0.36, color, 1)

    bottom = frame.copy()
    cv2.rectangle(bottom, (0, h - 42), (w, h), DARK, -1)
    cv2.addWeighted(bottom, 0.60, frame, 0.40, 0, frame)
    names = "  |  ".join(list(gestures.keys())) if gestures else "(kosong)"
    cv2.putText(frame, f"Gestures: {names}", (10, h - 14), FONT, 0.48, GREEN, 1)


def main():
    if not os.path.exists(MODEL_PATH):
        print("=" * 65)
        print(" ERROR: File model tidak ditemukan!")
        print(f" Dibutuhkan: {MODEL_PATH}")
        print("=" * 65)
        return

    gestures = load_gestures()
    smooth_n = 6
    label_history = []

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.70,
        min_tracking_confidence=0.60,
    )

    print("=" * 52)
    print(" Two-Hand Recognizer siap")
    print(f" Loaded: {len(gestures)} gesture(s): {list(gestures.keys())}")
    print(" R = reload | Q = keluar")
    print("=" * 52)

    cap, backend = open_camera(0)
    if cap is None:
        print("=" * 65)
        print(" ERROR: Webcam tidak bisa dibuka.")
        print("=" * 65)
        return

    print(f" Camera backend: {backend}")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        last_ts_ms = -1
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Warning: frame kamera gagal dibaca, recognizer dihentikan.")
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
            live_sample = build_two_hand_sample(result)
            hand_labels = []

            if result.hand_landmarks:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx] if idx < len(result.handedness) else None
                    hand_label = get_handed_label(handedness) or f"Hand{idx + 1}"
                    hand_labels.append(hand_label)
                    draw_landmarks_manual(frame, landmarks, hand_label)

            hand_state = " + ".join(sorted(set(hand_labels))) if hand_labels else "tidak ada tangan"
            label = None
            confidence = 0.0
            detail = {}

            if live_sample is not None:
                hand_state = "Left + Right"
                label, confidence, detail = match_gesture(live_sample, gestures)
                label_history.append((label, confidence))
                if len(label_history) > smooth_n:
                    label_history.pop(0)

                grouped = {}
                for hist_label, hist_conf in label_history:
                    grouped.setdefault(hist_label, []).append(hist_conf)
                best = max(grouped, key=lambda item: np.mean(grouped[item]))
                label = best
                confidence = float(np.mean(grouped[best]))
            else:
                label_history.clear()

            draw_ui(frame, label, confidence, detail, gestures, hand_state)
            cv2.imshow("Two-Hand Recognizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                gestures = load_gestures()
                label_history.clear()
                print(f"  Reloaded: {len(gestures)} gesture(s): {list(gestures.keys())}")

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()
