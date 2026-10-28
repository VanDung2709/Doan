import os
import uuid
import cv2
import time
import math
import urllib.request
import argparse
import threading
import queue
import subprocess
import shutil

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

# V13: gioi han thread TensorFlow de tranh AI chiem het CPU cua camera/YOLO.
if TF_AVAILABLE:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"
FACE_MODEL_PATH = BASE_DIR / "face_landmarker.task"
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# AI models da duoc chot tu cac tuan truoc.
FER_MODEL_PATH = BASE_DIR / "mobilenetv2_fer2013_final.keras"
ENGAGEMENT_MODEL_CANDIDATES = [
    BASE_DIR / "Week7_Baseline" / "mobilenetv2_daisee_engagement_baseline.keras",
    BASE_DIR / "mobilenetv2_daisee_engagement_official_final.keras",
    BASE_DIR / "mobilenetv2_daisee_engagement_final.keras",
]


# ============================================================
# WEEK 8 - BACKEND MULTI-TRACK (KHOI)
# ============================================================
# V14 van la nguon quyet dinh AI/Fusion. Backend CHI luu + phat realtime.
BACKEND_ENABLED = True
BACKEND_URL = "http://localhost:3001"
BACKEND_ENV_PATH = BASE_DIR / ".env"
BACKEND_SEND_INTERVAL_PER_TRACK_SECONDS = 2.0
BACKEND_REQUEST_TIMEOUT_SECONDS = 2.0
BACKEND_MAX_REQUESTS_PER_SECOND = 6.0
BACKEND_SOURCE_VERSION = "V14"

# Track ID cua ByteTrack la tam thoi, KHONG phai studentId.
# Chi dien mapping nay khi da co quy trinh gan danh tinh ro rang.
# Vi du: TRACK_TO_STUDENT_ID = {1: 5, 2: 8}
TRACK_TO_STUDENT_ID = {}

FER_CLASSES = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]
ENGAGEMENT_CLASSES = ["Very Low", "Low", "High", "Very High"]

FER_DISPLAY = {
    "angry": "Tức giận",
    "disgust": "Khó chịu",
    "fear": "Sợ hãi",
    "happy": "Vui",
    "neutral": "Trung tính",
    "sad": "Buồn",
    "surprise": "Bất ngờ",
}
ENGAGEMENT_DISPLAY = {
    "Very Low": "Rất thấp",
    "Low": "Thấp",
    "High": "Cao",
    "Very High": "Rất cao",
}

CAMERA_INDEX = 0
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 576
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
PERSON_CLASS = 0

EAR_THRESHOLD = 0.21
MAX_FACE_ANALYSES_PER_FRAME = 3  # Round-robin; improve coverage of crowds, bounded per frame
SHOW_DEBUG_EYE_LANDMARKS = False  # True only when diagnosing landmark quality
MAX_POSE_ANALYSES_PER_FRAME = 1

# CROWD_V16: >5 nguoi = uu tien MAT, <=5 nguoi = phan tich MAT + TU THE.
# Day la nguong thiet ke theo yeu cau, khong phai nguong duoc nghien cuu thuc nghiem.
CROWD_FACE_ONLY_THRESHOLD = 5
CROWD_RETURN_TO_FULL_SECONDS = 1.5
FACE_SIGNAL_FRESH_SECONDS = 2.5  # Bridge brief scheduling gaps, never use unbounded stale evidence
EYE_MIN_DISTINCT_SAMPLES = 3
FACE_OBSERVATION_MAX_GAP_SECONDS = 2.5  # Require repeated NEW face observations
HEAD_AWAY_ESCALATE_SECONDS = 10.0  # Describe behavior only; NOT an anxiety diagnosis

YOLO_EVERY_N_FRAMES = 2
POSE_EVERY_N_FRAMES = 2
UPPER_BODY_RATIO = 0.50
CACHE_MAX_AGE_FRAMES = 90

MIN_PERSON_WIDTH = 70
MIN_PERSON_HEIGHT = 120

YAW_THRESHOLD = 25.0
PITCH_DOWN_THRESHOLD = 18.0
PITCH_UP_THRESHOLD = -18.0

FACE_MISSING_TURNED_SECONDS = 1.5

# Body Pose - nguong ban dau, se hieu chinh sau khi test that
POSE_VISIBILITY_MIN = 0.45
BODY_TURN_RATIO_THRESHOLD = 0.58
STANDING_KNEE_ANGLE = 150.0
SITTING_KNEE_ANGLE = 135.0

# Giữ trạng thái Body Pose đáng tin cậy cho từng Track ID.
# Nếu một lần phân tích kế tiếp bị bàn/ghế che hoặc landmark thiếu,
# không xóa ngay trạng thái Đứng/Ngồi/Quay người của học sinh đó.
BODY_STATUS_HOLD_SECONDS = 3.0

# Vị trí chuẩn HOÀN TOÀN TỰ ĐỘNG.
# Mỗi Track ID mới được quan sát vài giây; khi vị trí đủ ổn định,
# hệ thống tự lấy median bounding box làm mốc, không cần nhấn C.
AUTO_SEAT_CALIBRATION_SECONDS = 2.5
AUTO_SEAT_MIN_SAMPLES = 8
AUTO_SEAT_SAMPLE_WINDOW_SECONDS = 4.0
AUTO_SEAT_MAX_X_SPREAD_RATIO = 0.28
AUTO_SEAT_MAX_Y_SPREAD_RATIO = 0.22

# Chỉ báo "Rời vị trí" khi Track ID ra khỏi vùng chuẩn đủ lâu.
SEAT_EXIT_SECONDS = 2.0
SEAT_EXIT_X_RATIO = 0.75
SEAT_EXIT_Y_RATIO = 0.50
SEAT_RETURN_X_RATIO = 0.55
SEAT_RETURN_Y_RATIO = 0.35

# ============================================================
# FUSION HANH VI - RIENG TUNG TRACK ID
# ============================================================
# Dung / ngoi mot minh KHONG bi xem la mat tap trung.
# Chi canh bao khi mot tin hieu bat thuong keo dai du lau,
# hoac khi Track ID da duoc xac nhan roi vi tri.
EYE_CLOSED_ALERT_SECONDS = 1.8
HEAD_AWAY_ALERT_SECONDS = 5.0  # Observable prolonged head turn; NOT proof of distraction

# Body Pose de bi nhieu hon Head Pose, nen V11 xac nhan nhieu lan
# va chi tinh khi ket qua Pose con moi.
BODY_TURN_ALERT_SECONDS = 5.0
BODY_TURN_START_CONFIRMATIONS = 2
BODY_TURN_ALERT_CONFIRMATIONS = 3
BODY_TURN_FRESH_SECONDS = 1.2

# Neu ca dau va than cung lech huong, co the canh bao som hon
# vi da co 2 tin hieu doc lap.
HEAD_BODY_COMBINED_SECONDS = 2.0

# Giu canh bao ngan de giao dien khong nhap nhay do/xanh.
FUSION_ALERT_HOLD_SECONDS = 1.5

# ============================================================
# FER + DAiSEE ENGAGEMENT - V12
# ============================================================
# V13 OPTIMIZED:
# - AI chay tren 1 worker nen camera/YOLO khong phai dung cho TensorFlow.
# - Chay thua hon V12 de giu FPS khi co 5-8 hoc sinh.
AI_GLOBAL_INTERVAL_SECONDS = 0.75  # If worker is free: prefer responsive FER for visible faces
AI_PER_TRACK_INTERVAL_SECONDS = 2.50  # Round robin across visible, measurable faces
AI_RESULT_FRESH_SECONDS = 10.0
FACE_BBOX_FRESH_SECONDS = 4.5
AI_EMA_ALPHA = 0.35
PRINT_AI_RESULTS = (os.getenv("KLCN313_DEBUG_AI", "0").strip() == "1")  # Opt-in console debug; overlay never shows percentages
AI_QUEUE_MAXSIZE = 1
AI_MIN_PERSON_WIDTH = 110
AI_MIN_PERSON_HEIGHT = 160

# Threshold Low Engagement da chon tren Validation o Tuan 6/7.
LOW_ENGAGEMENT_THRESHOLD = 0.27
LOW_ENGAGEMENT_ALERT_SECONDS = 2.0

# LEGACY: diem emotion/engagement chi luu telemetries, KHONG duoc tao canh bao tam ly.
NEGATIVE_EMOTION_WEIGHT = 0.75
ENGAGEMENT_WEIGHT = 0.25
ANXIETY_FUSION_THRESHOLD = 0.45
ANXIETY_ALERT_SECONDS = 3.0

# LEGACY: khong duoc bien mot mau FER thanh canh bao tam ly
# duoc giu lai qua nhieu frame. Nguong CAN KIEM CHUNG tren video co nhan,
# day chi la hang rao giam duong tinh gia khi demo, khong phai do accuracy.
ANXIETY_MIN_INDEPENDENT_SAMPLES = 3
ANXIETY_MIN_SAMPLE_SPAN_SECONDS = 6.0
ANXIETY_SAMPLE_WINDOW_SECONDS = 20.0
ANXIETY_STRONG_NEGATIVE_PROB = 0.65
ANXIETY_STRONG_LOW_ENG_PROB = 0.45
ANXIETY_MIN_FER_TOP_CONF = 0.55
ANXIETY_SAMPLE_SYNC_SECONDS = 2.0

# Khong su dung thay doi vi tri de ket luan mat tap trung (khong so do ghe).
ENABLE_SEAT_AWAY_ALERTS = False

HEAD_AWAY_STATUSES = {
    "QUAY TRAI",
    "QUAY PHAI",
    "CUI DAU",
    "NGUOC LEN",
    "QUAY KHUAT / KHONG NHIN PHIA TRUOC",
}

BODY_TURN_STATUSES = {
    "QUAY NGUOI",
    "DUNG / QUAY NGUOI",
    "NGOI / QUAY NGUOI",
}

RELIABLE_BODY_STATUSES = {
    "DUNG",
    "NGOI",
    "QUAY NGUOI",
    "DUNG / QUAY NGUOI",
    "NGOI / QUAY NGUOI",
}

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)



# ============================================================
# BACKEND SENDER MULTI-TRACK - THREAD RIENG, KHONG BLOCK CAMERA
# ============================================================

def load_backend_env(path):
    values = {}

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file .env: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ('"', "'")
        ):
            value = value[1:-1]

        values[key.strip()] = value

    return values


class MultiTrackBackendSender:
    """
    Gui ket qua V14 cua nhieu Track ID sang Backend.

    - Khong thay doi logic Fusion.
    - Main camera chi cap nhat pending dict, khong cho HTTP.
    - Moi Track ID co nhip gui rieng de 5-8 hoc sinh khong de len nhau.
    - Neu mang cham, chi giu payload moi nhat cua tung Track ID.
    """

    def __init__(self):
        self.enabled = bool(BACKEND_ENABLED and REQUESTS_AVAILABLE)
        self.base_url = BACKEND_URL.rstrip("/")
        self.interval_per_track = BACKEND_SEND_INTERVAL_PER_TRACK_SECONDS
        self.timeout = BACKEND_REQUEST_TIMEOUT_SECONDS
        self.min_request_gap = 1.0 / max(BACKEND_MAX_REQUESTS_PER_SECOND, 0.1)

        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.token = None
        self.email = None
        self.password = None

        self.pending = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None

        self.last_enqueue_by_track = {}
        self.last_status_by_track = {}

        self.sent_count = 0
        self.failed_count = 0
        self.alert_count = 0
        self.service_key = None

    def start(self):
        if not BACKEND_ENABLED:
            self.enabled = False
            print("[BACKEND] Da tat theo cau hinh.")
            return

        if not REQUESTS_AVAILABLE:
            self.enabled = False
            print("[BACKEND] Thieu requests -> V14 van chay, khong gui Backend.")
            print("[BACKEND] Cai neu can: python -m pip install requests")
            return

        try:
            env = load_backend_env(BACKEND_ENV_PATH)

            self.service_key = env.get("AI_SERVICE_KEY")

            if not self.service_key or len(self.service_key) < 32:
                raise RuntimeError("AI_SERVICE_KEY rieng phai co it nhat 32 ky tu trong .env")

            self.worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="V14-Backend-MultiTrack",
            )
            self.worker.start()

            print(
                "[BACKEND] V14 service-key ket noi san sang:",
                self.base_url
            )
            print(
                "[BACKEND] Moi Track gui toi da 1 lan/",
                f"{self.interval_per_track:.1f}s | global <= {BACKEND_MAX_REQUESTS_PER_SECOND:.1f} req/s"
            )
            print(
                "[BACKEND] Khong con phu thuoc mat khau Admin."
            )
            print(
                "[BACKEND] Track ID duoc luu rieng, khong gia lam studentId."
            )

        except Exception as error:
            self.enabled = False
            print("[BACKEND] Khong khoi tao duoc:", error)
            print("[BACKEND] V14 van tiep tuc camera/AI binh thuong.")

    def _headers(self):
        return {
            "X-AI-Service-Key": self.service_key
        }

    def maybe_send(self, track_id, payload, fusion_status):
        if not self.enabled:
            return

        tid = int(track_id)
        now = time.perf_counter()
        last_time = float(self.last_enqueue_by_track.get(tid, -1e9))
        last_status = self.last_status_by_track.get(tid)

        status_changed = fusion_status != last_status
        interval_reached = (now - last_time) >= self.interval_per_track

        if not (status_changed or interval_reached):
            return

        self.last_enqueue_by_track[tid] = now
        self.last_status_by_track[tid] = fusion_status

        with self.lock:
            self.pending[tid] = dict(payload)

    def _post_once(self, payload):
        response = self.session.post(
            f"{self.base_url}/api/ai/results",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code == 401:
            raise RuntimeError(
                "AI service key bi Backend tu choi. "
                "Kiem tra AI_SERVICE_KEY trong .env."
            )

        response.raise_for_status()
        data = response.json()
        self.sent_count += 1

        if data.get("alertCreated"):
            self.alert_count += 1
            alert = data.get("alert") or {}
            print(
                "[BACKEND ALERT]",
                f"Track={payload.get('trackId')}",
                alert.get("alert_type"),
                "confidence=",
                alert.get("confidence"),
            )

    def _worker_loop(self):
        while not self.stop_event.is_set():
            payload = None

            with self.lock:
                if self.pending:
                    tid = next(iter(self.pending))
                    payload = self.pending.pop(tid)

            if payload is None:
                time.sleep(0.03)
                continue

            try:
                self._post_once(payload)
            except Exception as error:
                self.failed_count += 1
                print(
                    "[BACKEND] Gui that bai | Track",
                    payload.get("trackId"),
                    "|",
                    error,
                )
                # Khong dua lai payload cu vo han; frame moi se cap nhat lai.
                time.sleep(0.30)

            time.sleep(self.min_request_gap)

    def close(self):
        self.stop_event.set()

        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.5)

        if self.session is not None:
            self.session.close()



def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(points):
    a = dist(points[1], points[5])
    b = dist(points[2], points[4])
    c = dist(points[0], points[3])

    if c <= 1e-6:
        return 0.0

    return (a + b) / (2.0 * c)


def landmarks_to_points(landmarks, indices, width, height):
    pts = []

    for idx in indices:
        lm = landmarks[idx]
        pts.append(
            (
                int(lm.x * width),
                int(lm.y * height)
            )
        )

    return pts


def landmark_xy(landmarks, idx, width, height):
    lm = landmarks[idx]

    return (
        float(lm.x * width),
        float(lm.y * height)
    )


def normalize_pitch(pitch):
    if pitch > 90:
        pitch = 180 - pitch
    elif pitch < -90:
        pitch = -180 - pitch

    return pitch


def estimate_head_pose(landmarks, width, height):
    image_points = np.array([
        landmark_xy(landmarks, NOSE_TIP, width, height),
        landmark_xy(landmarks, CHIN, width, height),
        landmark_xy(landmarks, LEFT_EYE_CORNER, width, height),
        landmark_xy(landmarks, RIGHT_EYE_CORNER, width, height),
        landmark_xy(landmarks, LEFT_MOUTH, width, height),
        landmark_xy(landmarks, RIGHT_MOUTH, width, height),
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = normalize_pitch(float(angles[0]))
    yaw = float(angles[1])
    roll = float(angles[2])

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll
    }


def landmark_face_alignment(landmarks, image_size=None):
    """Face orientation from PIXEL-scale landmarks (never raw unequal x/y ratios).

    FRONT means the head is approximately frontal, NOT that eye gaze or
    attention has been proved. A blurry/occluded face stays UNCERTAIN.
    """
    try:
        indices = (33, 263, 1, 61, 291, 234, 454)
        if landmarks is None or len(landmarks) <= max(indices):
            return "UNCERTAIN"
        pixel_mode = image_size is not None
        if pixel_mode:
            width, height = [float(x) for x in image_size]
            if not (math.isfinite(width) and math.isfinite(height)) or min(width, height) <= 0:
                return "UNCERTAIN"
        else:
            # Backward-compatible normalized-landmark calls (e.g. old tests).
            width, height = 1.0, 1.0

        points = {}
        for idx in indices:
            xx, yy = float(landmarks[idx].x), float(landmarks[idx].y)
            if not (math.isfinite(xx) and math.isfinite(yy)):
                return "UNCERTAIN"
            if not (-0.05 <= xx <= 1.05 and -0.05 <= yy <= 1.05):
                return "UNCERTAIN"
            points[idx] = np.array([xx * width, yy * height], dtype=float)

        left, right = points[33], points[263]
        eye_vector = right - left
        eye_sep = float(np.linalg.norm(eye_vector))
        min_eye_sep = 14.0 if pixel_mode else 0.045
        if eye_sep < min_eye_sep:
            return "UNCERTAIN"  # Too few pixels for trustworthy head direction.
        horizontal = eye_vector / eye_sep
        down = np.array([-horizontal[1], horizontal[0]])
        eye_middle = (left + right) / 2.0
        nose = points[1]
        mouth_middle = (points[61] + points[291]) / 2.0

        # Rotation-invariant: head tilt should not change frontal/side result.
        if float(np.dot(mouth_middle - eye_middle, down)) < 0.0:
            down = -down
        nose_y = float(np.dot(nose - eye_middle, down)) / eye_sep
        mouth_y = float(np.dot(mouth_middle - eye_middle, down)) / eye_sep
        if not (0.06 < nose_y < mouth_y + 0.12 and 0.24 < mouth_y < 2.15):
            return "UNCERTAIN"

        nose_shift = abs(float(np.dot(nose - eye_middle, horizontal))) / eye_sep
        mouth_shift = abs(float(np.dot(mouth_middle - eye_middle, horizontal))) / eye_sep
        cheek_l = abs(float(np.dot(nose - points[234], horizontal)))
        cheek_r = abs(float(np.dot(points[454] - nose, horizontal)))
        if min(cheek_l, cheek_r) < 0.10 * eye_sep:
            return "UNCERTAIN"
        balance = min(cheek_l, cheek_r) / max(cheek_l, cheek_r)

        # Allow moderate individual asymmetry; prefer UNKNOWN over a false turn.
        if nose_shift <= 0.34 and mouth_shift <= 0.43 and balance >= 0.33:
            return "FRONT"
        if nose_shift >= 0.53 or (nose_shift >= 0.40 and balance <= 0.22):
            return "TURNED"
        return "UNCERTAIN"
    except (IndexError, TypeError, AttributeError, ValueError,
            ZeroDivisionError, OverflowError):
        return "UNCERTAIN"


def landmarks_strongly_front_facing(landmarks):
    return landmark_face_alignment(landmarks) == "FRONT"


def classify_head_pose(yaw, pitch, landmarks=None, image_size=None):
    """Resolve face direction conservatively using pixel-correct face geometry.

    solvePnP is only an auxiliary cue: its focal point in a cropped person
    box is not a calibrated physical camera. Do not mistake its artefacts for
    sustained turning, or for a clinical/attention inference.
    """
    alignment = landmark_face_alignment(landmarks, image_size) if landmarks is not None else "UNCERTAIN"

    # A clear frontal face overrides an artefact in solvePnP pitch/yaw.
    if alignment == "FRONT":
        return "NHIN THANG"

    try:
        yaw = float(yaw)
        pitch = float(pitch)
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            return "POSE UNKNOWN"
    except (TypeError, ValueError):
        return "POSE UNKNOWN"

    yaw_display = -yaw
    if alignment == "TURNED" and abs(yaw) >= 18.0:
        return "QUAY TRAI" if yaw_display < 0 else "QUAY PHAI"

    # A face found but too small/occluded or with conflicting cues stays gray.
    if landmarks is not None:
        return "POSE UNKNOWN"

    # Backwards compatible angle-only path for external callers/tests.
    if yaw_display <= -YAW_THRESHOLD:
        return "QUAY TRAI"
    if yaw_display >= YAW_THRESHOLD:
        return "QUAY PHAI"
    if pitch >= PITCH_DOWN_THRESHOLD:
        return "CUI DAU"
    if pitch <= PITCH_UP_THRESHOLD:
        return "NGUOC LEN"
    return "NHIN THANG"


def analyze_face(face_landmarker, person_crop, upper_body_ratio=UPPER_BODY_RATIO):
    if person_crop is None or person_crop.size == 0:
        return False, None, "NO FACE", None, [], [], None

    h, w = person_crop.shape[:2]

    if h < MIN_PERSON_HEIGHT or w < MIN_PERSON_WIDTH:
        return False, None, "TOO SMALL", None, [], [], None

    upper_h = max(1, int(h * upper_body_ratio))
    upper_crop = person_crop[0:upper_h, :]

    scale = 2.0

    upscaled = cv2.resize(
        upper_crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    rgb = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2RGB
    ).copy()

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return False, None, "NO FACE", None, [], [], None

    landmarks = result.face_landmarks[0]

    if len(landmarks) < 468:
        return False, None, "NO FACE", None, [], [], None

    left_points_up = landmarks_to_points(
        landmarks,
        LEFT_EYE,
        upscaled.shape[1],
        upscaled.shape[0]
    )

    right_points_up = landmarks_to_points(
        landmarks,
        RIGHT_EYE,
        upscaled.shape[1],
        upscaled.shape[0]
    )

    # A tiny/blurry eye may collapse to identical pixel points and produce
    # EAR=0.0, which the old code incorrectly called "eyes closed".
    horizontal_left = dist(left_points_up[0], left_points_up[3])
    horizontal_right = dist(right_points_up[0], right_points_up[3])
    if min(horizontal_left, horizontal_right) < 8.0:
        ear = None
        eye_status = "MAT CHUA RO"
    else:
        left_ear = eye_aspect_ratio(left_points_up)
        right_ear = eye_aspect_ratio(right_points_up)
        ear = (left_ear + right_ear) / 2.0
        eye_status = "NHAM MAT" if ear < EAR_THRESHOLD else "MO MAT"

    head_pose = estimate_head_pose(
        landmarks,
        upscaled.shape[1],
        upscaled.shape[0]
    )
    if head_pose is None:
        # 2D landmarks may still be reliable when uncalibrated solvePnP fails.
        head_pose = {"yaw": None, "pitch": None, "roll": None}
    head_pose["landmarks"] = landmarks
    head_pose["image_size"] = (upscaled.shape[1], upscaled.shape[0])

    # Bounding box khuon mat theo toa do CHUAN HOA cua person_crop.
    # Dung lai bbox nay cho FER de khong can chay them face detector.
    xs = [float(lm.x) for lm in landmarks]
    ys = [float(lm.y) for lm in landmarks]
    face_margin_x = 0.08
    face_margin_y = 0.10
    x1n = max(0.0, min(xs) - face_margin_x)
    x2n = min(1.0, max(xs) + face_margin_x)
    y1_upper = max(0.0, min(ys) - face_margin_y)
    y2_upper = min(1.0, max(ys) + face_margin_y)
    upper_fraction = float(upper_h) / max(float(h), 1.0)
    y1n = max(0.0, y1_upper * upper_fraction)
    y2n = min(1.0, y2_upper * upper_fraction)
    face_bbox_norm = (x1n, y1n, x2n, y2n)

    left_points = [
        (int(x / scale), int(y / scale))
        for x, y in left_points_up
    ]

    right_points = [
        (int(x / scale), int(y / scale))
        for x, y in right_points_up
    ]

    return (
        True,
        ear,
        eye_status,
        head_pose,
        left_points,
        right_points,
        face_bbox_norm
    )




# ============================================================
# BODY POSE
# ============================================================

# MediaPipe Pose landmark indices
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28


def download_pose_model_if_needed():
    if POSE_MODEL_PATH.exists() and POSE_MODEL_PATH.stat().st_size > 1_000_000:
        print("[POSE MODEL] Da co:", POSE_MODEL_PATH)
        return

    print("======================================================")
    print("[POSE MODEL] Dang tai pose_landmarker_lite.task")
    print("======================================================")

    request = urllib.request.Request(
        POSE_MODEL_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    temp_path = POSE_MODEL_PATH.with_suffix(".task.part")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded / total * 100
                        print(
                            f"\r[POSE MODEL] {downloaded/1024/1024:.2f} MB / "
                            f"{total/1024/1024:.2f} MB ({percent:.1f}%)",
                            end=""
                        )
                    else:
                        print(
                            f"\r[POSE MODEL] Da tai {downloaded/1024/1024:.2f} MB",
                            end=""
                        )

        print()

        if not temp_path.exists() or temp_path.stat().st_size < 1_000_000:
            raise RuntimeError("File pose_landmarker_lite.task tai ve khong hop le.")

        temp_path.replace(POSE_MODEL_PATH)
        print("[POSE MODEL] Tai thanh cong:", POSE_MODEL_PATH)

    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        print("[LOI] Khong tai duoc pose_landmarker_lite.task")
        print("[LOI]", e)
        raise SystemExit(1)


def lm_visibility(lm):
    value = getattr(lm, "visibility", 1.0)
    try:
        return float(value)
    except Exception:
        return 1.0


def point2d(landmarks, idx, width, height):
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float64)


def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)

    if nba < 1e-6 or nbc < 1e-6:
        return None

    cos_angle = np.dot(ba, bc) / (nba * nbc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def analyze_body_pose(pose_landmarker, person_crop):
    """
    Tra ve body_found, body_status, body_metrics.
    Khong ep ket luan neu landmark chan bi che boi ban/ghe.
    """
    if person_crop is None or person_crop.size == 0:
        return False, "BODY UNKNOWN", {}

    h, w = person_crop.shape[:2]

    if h < 120 or w < 60:
        return False, "BODY TOO SMALL", {}

    rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB).copy()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return False, "BODY UNKNOWN", {}

    landmarks = result.pose_landmarks[0]

    if len(landmarks) < 29:
        return False, "BODY UNKNOWN", {}

    needed_upper = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if any(lm_visibility(landmarks[i]) < POSE_VISIBILITY_MIN for i in needed_upper):
        return False, "BODY UNKNOWN", {}

    ls = point2d(landmarks, L_SHOULDER, w, h)
    rs = point2d(landmarks, R_SHOULDER, w, h)
    lh = point2d(landmarks, L_HIP, w, h)
    rh = point2d(landmarks, R_HIP, w, h)

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    shoulder_width = float(np.linalg.norm(ls - rs))
    torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))

    body_turn_ratio = shoulder_width / max(torso_height, 1.0)

    # Khi xoay ngang, khoang cach 2 vai theo anh thuong hep lai.
    body_turned = body_turn_ratio < BODY_TURN_RATIO_THRESHOLD

    knee_angles = []

    for hip_idx, knee_idx, ankle_idx in [
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    ]:
        vis_ok = all(
            lm_visibility(landmarks[i]) >= POSE_VISIBILITY_MIN
            for i in (hip_idx, knee_idx, ankle_idx)
        )

        if not vis_ok:
            continue

        hip = point2d(landmarks, hip_idx, w, h)
        knee = point2d(landmarks, knee_idx, w, h)
        ankle = point2d(landmarks, ankle_idx, w, h)

        ang = angle_3pts(hip, knee, ankle)
        if ang is not None:
            knee_angles.append(ang)

    posture = "TU THE CHUA RO"

    if knee_angles:
        mean_knee = float(sum(knee_angles) / len(knee_angles))

        if mean_knee >= STANDING_KNEE_ANGLE:
            posture = "DUNG"
        elif mean_knee <= SITTING_KNEE_ANGLE:
            posture = "NGOI"
    else:
        mean_knee = None

    if body_turned:
        if posture == "DUNG":
            status = "DUNG / QUAY NGUOI"
        elif posture == "NGOI":
            status = "NGOI / QUAY NGUOI"
        else:
            status = "QUAY NGUOI"
    else:
        status = posture

    return True, status, {
        "body_turn_ratio": body_turn_ratio,
        "mean_knee_angle": mean_knee,
    }


# ============================================================
# GIAO DIEN - TIENG VIET CO DAU + 1 DONG / NGUOI
# ============================================================

FONT_PATHS = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]


def load_vi_font(size):
    if not PIL_AVAILABLE:
        return None

    for font_path in FONT_PATHS:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass

    try:
        return ImageFont.load_default()
    except Exception:
        return None


FONT_PERSON = load_vi_font(18)
FONT_HUD = load_vi_font(16)


DISPLAY_TEXT = {
    "NHIN THANG": "Nhìn thẳng",
    "QUAY TRAI": "Quay trái",
    "QUAY PHAI": "Quay phải",
    "CUI DAU": "Cúi đầu",
    "NGUOC LEN": "Ngửa đầu",
    "QUAY KHUAT / KHONG NHIN PHIA TRUOC": "Quay khuất",
    "QUA XA / MAT QUA NHO": "Mặt quá nhỏ / quá xa",
    "DANG TIM MAT": "Đang tìm mặt",
    "POSE UNKNOWN": "Chưa xác định",
    "CHUA PHAN TICH": "Chưa phân tích",
    "DUNG": "Đứng",
    "NGOI": "Ngồi",
    "QUAY NGUOI": "Quay người",
    "DUNG / QUAY NGUOI": "Đứng • Quay người",
    "NGOI / QUAY NGUOI": "Ngồi • Quay người",
    "TU THE CHUA RO": "Tư thế chưa rõ",
    "BODY UNKNOWN": "Tư thế chưa rõ",
    "BODY TOO SMALL": "Người quá nhỏ",
    "ROI VI TRI": "Rời vị trí",
}


def get_person_display(head_status, eye_status, body_status, seat_away=False):
    """
    Luôn ghép trạng thái của CHÍNH Track ID đó trên 1 dòng.
    Ví dụ:
      ID 18 • Quay trái • Đứng
      ID 14 • Quay khuất • Ngồi
      ID 2  • Nhắm mắt • Ngồi
    """

    parts = []

    # 0) Rời vị trí - tín hiệu hành vi ưu tiên hiển thị.
    if seat_away:
        parts.append("Rời vị trí")

    # 1) Mắt - đỏ nếu đang nhắm
    if eye_status == "NHAM MAT":
        parts.append("Nhắm mắt")

    # 2) Hướng đầu
    head_text = DISPLAY_TEXT.get(head_status, head_status)

    if head_status in (
        "NHIN THANG",
        "QUAY TRAI",
        "QUAY PHAI",
        "CUI DAU",
        "NGUOC LEN",
        "QUAY KHUAT / KHONG NHIN PHIA TRUOC",
        "DANG TIM MAT",
        "QUA XA / MAT QUA NHO",
    ):
        # Khi đã có "Nhắm mắt" và đầu nhìn thẳng thì không cần thêm
        # "Nhìn thẳng" để nhãn khỏi quá dài.
        if not (
            eye_status == "NHAM MAT"
            and head_status == "NHIN THANG"
        ):
            parts.append(head_text)

    # 3) Tư thế thân người - luôn thêm nếu đã có kết quả đáng tin cậy.
    if body_status in RELIABLE_BODY_STATUSES:
        body_text = DISPLAY_TEXT.get(body_status, body_status)
        parts.append(body_text)

    # Nếu chưa có gì đủ tốt thì vẫn hiển thị trạng thái đầu hiện tại.
    if not parts:
        parts.append(
            DISPLAY_TEXT.get(
                head_status,
                "Đang phân tích"
            )
        )

    # Bỏ phần trùng nhau nhưng giữ thứ tự.
    unique_parts = []
    for part in parts:
        if part and part not in unique_parts:
            unique_parts.append(part)

    text = " • ".join(unique_parts)

    # Màu ưu tiên theo mức tín hiệu.
    if seat_away:
        return text, (0, 70, 255)  # đỏ-cam: rời vị trí

    if eye_status == "NHAM MAT":
        return text, (0, 0, 255)  # đỏ

    if head_status in (
        "QUAY TRAI",
        "QUAY PHAI",
        "CUI DAU",
        "NGUOC LEN",
        "QUAY KHUAT / KHONG NHIN PHIA TRUOC",
    ):
        return text, (0, 165, 255)  # cam

    if body_status in (
        "QUAY NGUOI",
        "DUNG / QUAY NGUOI",
        "NGOI / QUAY NGUOI",
    ):
        return text, (0, 165, 255)  # cam

    # Đứng chỉ là tín hiệu hành vi, chưa phải cảnh báo mất tập trung.
    if body_status == "DUNG":
        return text, (255, 180, 0)  # xanh lam/cyan

    if head_status == "NHIN THANG" and body_status == "NGOI":
        return text, (0, 200, 0)  # xanh lá

    if head_status == "DANG TIM MAT":
        return text, (0, 215, 255)  # vàng

    if head_status == "QUA XA / MAT QUA NHO":
        return text, (150, 150, 150)  # xám

    return text, (0, 200, 0)


# CROWD_V16_DISPLAY_FIX: the box color denotes FACE DIRECTION only,
# not a separate Fusion/Engagement flag. FER is shown independently.
def face_direction_and_fer_labels(head_status, fer_label, fer_conf, fer_fresh,
                                  eye_status="", eye_seconds=0.0, head_seconds=0.0):
    direction, color = visual_face_state(head_status, eye_status, eye_seconds, head_seconds)

    # V16.4: short overlay labels; the underlying FER model is unchanged.
    # Low confidence is not a verified emotion: show only "Chưa rõ".
    emotion = "Đang phân tích"
    if fer_fresh and fer_label in FER_CLASSES:
        try:
            score = float(fer_conf)
        except (TypeError, ValueError):
            score = float("nan")
        if math.isfinite(score) and 0 <= score <= 1:
            if score >= 0.55:
                candidate = FER_DISPLAY.get(fer_label, fer_label)
                emotion = candidate
            else:
                emotion = "Chưa rõ"
    return direction, color, emotion


def draw_unicode_labels(frame, labels, hud_lines):
    """
    Vẽ toàn bộ chữ Unicode một lần bằng Pillow để đỡ tốn FPS.
    labels: [(text, x, y, bgr_color), ...]
    """
    if not PIL_AVAILABLE or FONT_PERSON is None or FONT_HUD is None:
        # Fallback nếu máy chưa có Pillow: chữ không dấu bằng OpenCV.
        for text, x, y, color in labels:
            ascii_text = (
                text.replace("Nhìn", "Nhin")
                    .replace("thẳng", "thang")
                    .replace("Quay trái", "Quay trai")
                    .replace("Quay phải", "Quay phai")
                    .replace("Cúi đầu", "Cui dau")
                    .replace("Ngửa đầu", "Ngua dau")
                    .replace("khuất", "khuat")
                    .replace("Nhắm mắt", "Nham mat")
                    .replace("Đang tìm mặt", "Dang tim mat")
                    .replace("Mặt quá nhỏ / quá xa", "Mat qua nho / qua xa")
            )
            cv2.putText(
                frame,
                ascii_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).convert("RGBA")

    # Layer riêng để vẽ nền bán trong suốt.
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # HUD nhỏ 2 dòng
    if hud_lines:
        padding = 8
        line_gap = 4
        boxes = [
            draw.textbbox((0, 0), line, font=FONT_HUD)
            for line in hud_lines
        ]
        hud_w = max(b[2] - b[0] for b in boxes) + padding * 2
        hud_h = (
            sum(b[3] - b[1] for b in boxes)
            + line_gap * (len(hud_lines) - 1)
            + padding * 2
        )

        draw.rounded_rectangle(
            (8, 8, 8 + hud_w, 8 + hud_h),
            radius=8,
            fill=(0, 0, 0, 165),
        )

        yy = 8 + padding
        for line, bbox in zip(hud_lines, boxes):
            draw.text(
                (8 + padding, yy),
                line,
                font=FONT_HUD,
                fill=(255, 255, 255, 255),
            )
            yy += (bbox[3] - bbox[1]) + line_gap

    # 1 dòng cho mỗi người
    for text, x, y, bgr_color in labels:
        rgb_color = (
            int(bgr_color[2]),
            int(bgr_color[1]),
            int(bgr_color[0]),
        )

        bbox = draw.textbbox((0, 0), text, font=FONT_PERSON)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        px = max(2, min(x, frame.shape[1] - tw - 14))
        py = max(2, y - th - 10)

        # Nền tối để chữ không chìm vào ảnh
        draw.rounded_rectangle(
            (px - 5, py - 4, px + tw + 7, py + th + 5),
            radius=6,
            fill=(0, 0, 0, 155),
            outline=(*rgb_color, 235),
            width=2,
        )

        draw.text(
            (px, py),
            text,
            font=FONT_PERSON,
            fill=(*rgb_color, 255),
        )

    pil_image = Image.alpha_composite(pil_image, overlay).convert("RGB")
    result = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    return result



download_pose_model_if_needed()

if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {YOLO_MODEL_PATH}"
    )

# ============================================================
# FER + ENGAGEMENT AI
# ============================================================

def resolve_engagement_model_path():
    for path in ENGAGEMENT_MODEL_CANDIDATES:
        if path.exists():
            return path
    return ENGAGEMENT_MODEL_CANDIDATES[0]


def load_ai_models():
    if not TF_AVAILABLE:
        raise RuntimeError(
            "Chua co TensorFlow trong Python hien tai. "
            "Hay dung moi truong Python dang co TensorFlow va Ultralytics."
        )

    engagement_path = resolve_engagement_model_path()

    if not FER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay FER model: {FER_MODEL_PATH}"
        )

    if not engagement_path.exists():
        tried = "\n".join(str(p) for p in ENGAGEMENT_MODEL_CANDIDATES)
        raise FileNotFoundError(
            "Khong tim thay Engagement model. Da thu:\n" + tried
        )

    print("======================================================")
    print("LOAD FER + ENGAGEMENT MODELS")
    print("FER        :", FER_MODEL_PATH)
    print("Engagement :", engagement_path)
    print("======================================================")

    fer_model = tf.keras.models.load_model(
        str(FER_MODEL_PATH),
        compile=False
    )
    engagement_model = tf.keras.models.load_model(
        str(engagement_path),
        compile=False
    )

    print("FER input/output       :", fer_model.input_shape, fer_model.output_shape)
    print("Engagement input/output:", engagement_model.input_shape, engagement_model.output_shape)

    return fer_model, engagement_model, engagement_path


def model_hw(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    h = int(shape[1]) if shape[1] is not None else 224
    w = int(shape[2]) if shape[2] is not None else 224
    return h, w


def prepare_ai_input(bgr_crop, model):
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = model_hw(model)
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    # Ca 2 model cua do an da co Rescaling/preprocessing ben trong model.
    # Vi vay giu float32 o mien 0..255, KHONG preprocess_input lan nua.
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return batch


def predict_probs(model, bgr_crop):
    batch = prepare_ai_input(bgr_crop, model)
    if batch is None:
        return None

    output = model(batch, training=False)
    if hasattr(output, "numpy"):
        output = output.numpy()

    probs = np.asarray(output, dtype=np.float32).reshape(-1)
    total = float(np.sum(probs))

    # Neu model da la softmax thi tong xap xi 1. Neu khong, chuan hoa an toan.
    if total > 1e-6 and not (0.98 <= total <= 1.02):
        exp = np.exp(probs - np.max(probs))
        probs = exp / max(float(np.sum(exp)), 1e-6)

    return probs


def ema_probs(old_probs, new_probs, alpha=AI_EMA_ALPHA):
    if new_probs is None:
        return old_probs
    if old_probs is None:
        return new_probs.astype(np.float32)

    old = np.asarray(old_probs, dtype=np.float32)
    new = np.asarray(new_probs, dtype=np.float32)

    if old.shape != new.shape:
        return new

    smoothed = (1.0 - alpha) * old + alpha * new
    s = float(np.sum(smoothed))
    if s > 1e-6:
        smoothed = smoothed / s
    return smoothed


def ready_for_face_ai(cache, current_time):
    """Do not queue single-worker FER jobs before a fresh face ROI exists.

    This avoids spending time evaluating engagement only, and therefore
    improves label coverage in small groups. No guess is made when a face
    is too small, lost, or occluded.
    """
    if cache.get("ai_pending", False):
        return False
    if not cache.get("face_found", False):
        return False
    if cache.get("face_bbox_norm") is None:
        return False
    last = float(cache.get("face_bbox_time", -1e9))
    if not (0 <= float(current_time) - last <= FACE_SIGNAL_FRESH_SECONDS):
        return False
    try:
        x1, y1, x2, y2 = cache["face_bbox_norm"]
        return (x2 - x1) > 0.03 and (y2 - y1) > 0.02
    except (ValueError, TypeError):
        return False


def crop_face_from_norm(person_crop, bbox_norm):
    if person_crop is None or person_crop.size == 0 or bbox_norm is None:
        return None

    h, w = person_crop.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm

    x1 = max(0, min(w - 1, int(x1n * w)))
    y1 = max(0, min(h - 1, int(y1n * h)))
    x2 = max(x1 + 1, min(w, int(x2n * w)))
    y2 = max(y1 + 1, min(h, int(y2n * h)))

    crop = person_crop[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    return crop


def run_track_ai(
    fer_model,
    engagement_model,
    person_crop,
    cache,
    current_time,
):
    """Chay AI cho 1 Track ID va luu ket qua vao cache cua ID do."""
    cache["ai_last_inference_time"] = float(current_time)

    # FACE_ONLY: AI chi nhin ROI mat neu chat luong crop du de phan tich.
    # Khong chay suy luan tren toan bo nguoi trong crowd mode.
    face_bbox_time = float(cache.get("face_bbox_time", -1e9))
    face_bbox_fresh = (
        cache.get("face_bbox_norm") is not None
        and (float(current_time) - face_bbox_time) <= FACE_BBOX_FRESH_SECONDS
    )
    face_crop = crop_face_from_norm(person_crop, cache.get("face_bbox_norm")) if face_bbox_fresh else None
    if cache.get("crowd_face_only", False):
        if face_crop is None:
            cache["ai_error"] = "Mặt không đủ rõ để phân tích biểu cảm"
            return
        eng_crop = face_crop
    else:
        eng_crop = person_crop

    eng_new = predict_probs(engagement_model, eng_crop)
    if eng_new is not None and len(eng_new) >= 4:
        eng_probs = ema_probs(cache.get("eng_probs"), eng_new[:4])
        cache["eng_probs"] = eng_probs
        cache["eng_last_time"] = float(current_time)

        eng_idx = int(np.argmax(eng_probs))
        cache["eng_label"] = ENGAGEMENT_CLASSES[eng_idx]
        cache["eng_conf"] = float(eng_probs[eng_idx])
        cache["eng_low_score"] = float(eng_probs[0] + eng_probs[1])

    # FER luon dung khuon mat (khong dung nguoi), va phai la mat moi.
    # A weak top score stays 'Chưa rõ' in the UI; NEVER fabricate emotion.
    if face_crop is not None:
        fer_new = predict_probs(fer_model, face_crop)
        if fer_new is not None and len(fer_new) >= 7:
            fer_probs = ema_probs(cache.get("fer_probs"), fer_new[:7])
            cache["fer_probs"] = fer_probs
            cache["fer_last_time"] = float(current_time)

            fer_idx = int(np.argmax(fer_probs))
            cache["fer_label"] = FER_CLASSES[fer_idx]
            cache["fer_conf"] = float(fer_probs[fer_idx])

            # angry + fear + sad
            cache["negative_emotion"] = float(
                fer_probs[0] + fer_probs[2] + fer_probs[5]
            )

    neg = float(cache.get("negative_emotion", 0.0))
    low = float(cache.get("eng_low_score", 0.0))
    engagement_factor = min(low / max(LOW_ENGAGEMENT_THRESHOLD, 1e-6), 1.0)
    anxiety_score = (
        NEGATIVE_EMOTION_WEIGHT * neg
        + ENGAGEMENT_WEIGHT * engagement_factor
    )
    cache["anxiety_score"] = float(anxiety_score)


def ai_summary(cache):
    fer_label = cache.get("fer_label")
    eng_label = cache.get("eng_label")

    fer_text = FER_DISPLAY.get(fer_label, fer_label) if fer_label else "--"
    eng_text = ENGAGEMENT_DISPLAY.get(eng_label, eng_label) if eng_label else "--"
    return fer_text, eng_text


def ai_worker_loop(
    job_queue,
    stop_event,
    busy_event,
    fer_model,
    engagement_model,
):
    """
    V13: worker AI duy nhat. TensorFlow chay ngoai vong ve camera de
    tracking/GUI khong bi block moi lan suy luan FER + Engagement.
    """
    while not stop_event.is_set():
        try:
            track_id, person_crop, cache, job_time = job_queue.get(timeout=0.20)
        except queue.Empty:
            continue

        busy_event.set()
        try:
            run_track_ai(
                fer_model,
                engagement_model,
                person_crop,
                cache,
                job_time,
            )

            if PRINT_AI_RESULTS:
                fer_text, eng_text = ai_summary(cache)
                print(
                    f"[AI] ID {track_id} | FER={fer_text} "
                    f"({cache.get('fer_conf', 0.0):.2f}) | "
                    f"ENG={eng_text} "
                    f"Low={cache.get('eng_low_score', 0.0):.2f} | "
                    f"Anxiety={cache.get('anxiety_score', 0.0):.2f}"
                )

        except Exception as ai_error:
            cache["ai_error"] = str(ai_error)
            print(f"[AI WARNING] ID {track_id}: {ai_error}")

        finally:
            cache["ai_pending"] = False
            busy_event.clear()
            job_queue.task_done()


# ============================================================
# NGUON VIDEO / WEBCAM + TU DONG HOC VI TRI
# ============================================================

def parse_source(value):
    value = str(value)
    if value.isdigit():
        return int(value), False
    return value, True


def get_stream_time(cap, is_video):
    """Webcam dung thoi gian thuc; video dung timestamp cua video."""
    if is_video:
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms is not None and ms >= 0:
            return float(ms) / 1000.0
    return time.time()


def update_auto_seat_anchor(cache, box, current_time):
    """Tu dong khoa vi tri chuan khi Track ID dung on dinh du lau."""
    if cache.get("seat_anchor") is not None:
        return False

    x1, y1, x2, y2 = box
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    cx = (x1 + x2) / 2.0
    bottom_y = float(y2)

    samples = cache.setdefault("seat_auto_samples", [])
    samples.append({
        "t": float(current_time),
        "cx": float(cx),
        "bottom_y": bottom_y,
        "w": w,
        "h": h,
    })

    min_t = current_time - AUTO_SEAT_SAMPLE_WINDOW_SECONDS
    samples[:] = [s for s in samples if s["t"] >= min_t]

    if len(samples) < AUTO_SEAT_MIN_SAMPLES:
        cache["seat_auto_status"] = "DANG HOC VI TRI"
        return False

    duration = samples[-1]["t"] - samples[0]["t"]
    if duration < AUTO_SEAT_CALIBRATION_SECONDS:
        cache["seat_auto_status"] = "DANG HOC VI TRI"
        return False

    cxs = np.array([s["cx"] for s in samples], dtype=np.float32)
    bys = np.array([s["bottom_y"] for s in samples], dtype=np.float32)
    ws = np.array([s["w"] for s in samples], dtype=np.float32)
    hs = np.array([s["h"] for s in samples], dtype=np.float32)

    med_w = max(float(np.median(ws)), 1.0)
    med_h = max(float(np.median(hs)), 1.0)

    x_spread = float(np.percentile(cxs, 90) - np.percentile(cxs, 10)) / med_w
    y_spread = float(np.percentile(bys, 90) - np.percentile(bys, 10)) / med_h

    cache["seat_auto_x_spread"] = x_spread
    cache["seat_auto_y_spread"] = y_spread

    if (
        x_spread <= AUTO_SEAT_MAX_X_SPREAD_RATIO
        and y_spread <= AUTO_SEAT_MAX_Y_SPREAD_RATIO
    ):
        cache["seat_anchor"] = {
            "cx": float(np.median(cxs)),
            "bottom_y": float(np.median(bys)),
            "w": med_w,
            "h": med_h,
        }
        cache["seat_away_since"] = None
        cache["seat_away"] = False
        cache["seat_auto_status"] = "DA HOC VI TRI"
        cache["seat_auto_samples"] = []
        return True

    cache["seat_auto_status"] = "CHO VI TRI ON DINH"
    keep_after = current_time - 1.0
    cache["seat_auto_samples"] = [s for s in samples if s["t"] >= keep_after]
    return False



# ============================================================
# CROWD V16: CHON CHE DO THEO SO TRACK HIEN TAI
# ============================================================

def choose_face_only_mode(previous, low_since, person_count, current_time):
    """Vao FACE_ONLY ngay khi >5, tro lai FULL sau 1.5s on dinh <=5.

    Khong dung danh tinh, vi tri ghe, hay ket qua FER de quyet dinh mode.
    """
    t = float(current_time)
    if int(person_count) > CROWD_FACE_ONLY_THRESHOLD:
        return True, None
    if not previous:
        return False, None
    if low_since is None or t < float(low_since):
        return True, t
    if t - float(low_since) >= CROWD_RETURN_TO_FULL_SECONDS:
        return False, None
    return True, low_since


def reset_crowd_transition(cache, face_only):
    """Clear mode-specific interpretations; DO NOT clear Track ID or FER model."""
    cache["crowd_face_only"] = bool(face_only)
    cache["body_found"] = False
    cache["body_status"] = "TU THE CHUA RO"
    cache["body_last_reliable_status"] = None
    cache["body_last_reliable_time"] = -1e9
    cache["body_turn_confirmations"] = 0
    cache["seat_away"] = False
    cache["seat_away_since"] = None
    cache["fusion_last_alert_time"] = None
    cache["fusion_status"] = "BINH THUONG"
    cache["fusion_reason"] = ""
    for key in ("fusion_eye", "fusion_head", "fusion_body_turn", "final_low_eng", "final_anxiety"):
        cache[f"{key}_since"] = None
        cache[f"{key}_duration"] = 0.0
    cache["final_status"] = "BINH THUONG"
    cache["final_reason"] = ""
    for kind in ("head", "eye"):
        cache[f"v162_{kind}_last"] = None
        cache[f"v162_{kind}_start"] = None
        cache[f"v162_{kind}_samples"] = 0
        cache[f"v162_{kind}_seconds"] = 0.0
    # Khi nguon input AI chuyen tu toan than sang mat (hoac nguoc lai),
    # ket qua model cu khong con tuong thich. Bo cache, KHONG sua model.
    cache["eng_probs"] = None
    cache["eng_label"] = None
    cache["eng_conf"] = 0.0
    cache["eng_low_score"] = 0.0
    cache["eng_last_time"] = -1e9
    cache["fer_probs"] = None
    cache["fer_label"] = None
    cache["fer_conf"] = 0.0
    cache["negative_emotion"] = 0.0
    cache["fer_last_time"] = -1e9
    cache["anxiety_score"] = 0.0
    cache["_anxiety_independent_samples"] = []
    cache["_anxiety_last_checked_fer_at"] = -1e9


# V16.2: timestamps are taken ONLY when MediaPipe analyzes a NEW frame.
# Cached head/eye labels cannot silently accumulate durations when the face is lost.
def update_observed_face_durations(cache, head_status, eye_status, observed_at):
    now = float(observed_at)
    for kind, active in (("head", head_status in HEAD_AWAY_STATUSES),
                         ("eye", eye_status == "NHAM MAT")):
        last_key = f"v162_{kind}_last"
        start_key = f"v162_{kind}_start"
        sample_key = f"v162_{kind}_samples"
        duration_key = f"v162_{kind}_seconds"
        old = cache.get(last_key)
        gap = (now - float(old)) if old is not None else None
        if not active:
            cache[start_key] = None
            cache[last_key] = None
            cache[sample_key] = 0
            cache[duration_key] = 0.0
            continue
        if gap is not None and abs(gap) < 1e-4:
            continue  # Do not count the same video frame twice.
        if gap is None or gap < 0 or gap > FACE_OBSERVATION_MAX_GAP_SECONDS:
            cache[start_key] = now
            cache[sample_key] = 1
        else:
            cache[sample_key] = min(100000, int(cache.get(sample_key, 0)) + 1)
        cache[last_key] = now
        cache[duration_key] = max(0.0, now - float(cache[start_key]))


def verified_observation_seconds(cache, kind, now, min_samples=3):
    last = cache.get(f"v162_{kind}_last")
    if last is None or float(now) < float(last) or float(now) - float(last) > FACE_OBSERVATION_MAX_GAP_SECONDS:
        return 0.0
    if int(cache.get(f"v162_{kind}_samples", 0)) < int(min_samples):
        return 0.0
    return max(0.0, float(cache.get(f"v162_{kind}_seconds", 0.0)))


def visual_face_state(head_status, eye_status, verified_eye_seconds, verified_head_seconds):
    """BGR colors only reflect OBSERVED face/eye direction, not FER/Fusion psychology."""
    if eye_status == "NHAM MAT" and verified_eye_seconds >= EYE_CLOSED_ALERT_SECONDS:
        return f"Nhắm mắt kéo dài {verified_eye_seconds:.1f}s", (0, 0, 255)
    if head_status in HEAD_AWAY_STATUSES:
        if verified_head_seconds >= HEAD_AWAY_ESCALATE_SECONDS:
            return f"Lệch hướng {verified_head_seconds:.0f}s • cần chú ý", (0, 165, 255)
        if verified_head_seconds >= HEAD_AWAY_ALERT_SECONDS:
            return f"Lệch hướng {verified_head_seconds:.0f}s", (0, 165, 255)
        return "Lệch hướng", (0, 165, 255)
    if head_status == "NHIN THANG":
        # A brief blink isn't a sustained red warning.
        return "Nhìn về camera", (0, 200, 0)
    if head_status in ("DANG TIM MAT", "CHUA PHAN TICH", "NO FACE"):
        return "Đang tìm mặt", (145, 145, 145)
    if head_status in ("QUA XA / MAT QUA NHO", "TOO SMALL"):
        return "Mặt quá nhỏ", (145, 145, 145)
    return "Chưa rõ hướng mặt", (145, 145, 145)


# ============================================================
# FUSION HANH VI THEO THOI GIAN - TUNG TRACK ID
# ============================================================

def update_signal_timer(cache, key, active, current_time):
    """Cap nhat bo dem lien tuc cua 1 tin hieu va tra ve so giay."""
    start_key = f"{key}_since"
    duration_key = f"{key}_duration"

    if active:
        if cache.get(start_key) is None:
            cache[start_key] = float(current_time)
        duration = max(0.0, float(current_time) - float(cache[start_key]))
    else:
        cache[start_key] = None
        duration = 0.0

    cache[duration_key] = duration
    return duration


def update_behavior_fusion(cache, current_time):
    """
    Fusion hanh vi V11 - on dinh hon theo tung Track ID.

    Nguyen tac:
    - Nham mat: giu luat 1 giay.
    - Lech dau: 3 giay.
    - Quay nguoi: chi tinh khi Body Pose vua co ket qua tin cay,
      da duoc xac nhan lap lai, va phai keo dai 5 giay neu dung mot minh.
    - Quay dau + quay nguoi cung luc: canh bao som hon sau 2 giay.
    - Roi vi tri: canh bao ngay sau khi module vi tri da xac nhan.
    - Canh bao duoc giu them 1.5 giay de tranh nhap nhay.
    """
    eye_status = cache.get("eye_status", "")
    head_status = cache.get("head_status", "")
    body_status = cache.get("body_status", "")
    seat_away = bool(cache.get("seat_away", False))

    crowd_mode = bool(cache.get("crowd_face_only", False))
    last_face = float(cache.get("face_last_observed_time", -1e9))
    face_fresh = (0 <= float(current_time) - last_face <= FACE_SIGNAL_FRESH_SECONDS)
    # A label cached from a previous face detection is NOT a fresh measurement.
    eye_active = face_fresh and eye_status == "NHAM MAT"
    head_active = face_fresh and head_status in HEAD_AWAY_STATUSES

    body_last_reliable_time = float(
        cache.get("body_last_reliable_time", 0.0)
    )
    body_fresh = (
        body_last_reliable_time > 0.0
        and (float(current_time) - body_last_reliable_time)
            <= BODY_TURN_FRESH_SECONDS
    )

    body_turn_confirmations = int(
        cache.get("body_turn_confirmations", 0)
    )

    body_turn_active = (
        not crowd_mode
        and body_status in BODY_TURN_STATUSES
        and body_fresh
        and body_turn_confirmations >= BODY_TURN_START_CONFIRMATIONS
    )

    # Durations depend on independent, recent face observations, never just a
    # previously cached label carried forward across frames or lost tracking.
    eye_seconds = verified_observation_seconds(cache, "eye", current_time) if eye_active else 0.0
    head_seconds = verified_observation_seconds(cache, "head", current_time) if head_active else 0.0
    cache["fusion_eye_duration"] = eye_seconds
    cache["fusion_head_duration"] = head_seconds
    body_seconds = update_signal_timer(
        cache, "fusion_body_turn", body_turn_active, current_time
    )

    reasons = []

    if eye_seconds >= EYE_CLOSED_ALERT_SECONDS:
        reasons.append(f"Nhắm mắt kéo dài {eye_seconds:.1f}s")
    elif head_seconds >= HEAD_AWAY_ALERT_SECONDS:
        # This is an attention CHECK, never an anxiety/mental-health diagnosis.
        qualifier = " • cần giáo viên kiểm tra" if head_seconds >= HEAD_AWAY_ESCALATE_SECONDS else ""
        reasons.append(f"Lệch hướng liên tục {head_seconds:.1f}s{qualifier}")

    # Head direction alone is not proof of inattention (teacher/board may be
    # to the side). At >=5 seconds, emit a REVIEW alert with observable reason.
    # At >=10 seconds do NOT turn this into an anxiety diagnosis.
    combined_head_body = (
        not crowd_mode
        and head_active
        and body_turn_active
        and head_seconds >= HEAD_BODY_COMBINED_SECONDS
        and body_seconds >= HEAD_BODY_COMBINED_SECONDS
    )

    # Hai huong/tu the ket hop van chi la quan sat, KHONG la bang chung
    # chac chan cho mat tap trung. Canh bao do chi tu nham mat lap lai
    # va keo dai (hoac roi vi tri neu sau nay duoc xac minh/bat rieng).
    # seat_away da co bo dem SEAT_EXIT_SECONDS tu V9/V10.
    if not crowd_mode and ENABLE_SEAT_AWAY_ALERTS and seat_away:
        reasons.append("Rời vị trí")

    if reasons:
        fusion_status = "DAU HIEU MAT TAP TRUNG"
        fusion_reason = " + ".join(reasons[:2])
        cache["fusion_last_alert_time"] = float(current_time)

    else:
        # Hysteresis dau ra: giu canh bao cu mot chut de giao dien on dinh.
        last_alert_time = cache.get("fusion_last_alert_time")
        previous_status = cache.get("fusion_status", "BINH THUONG")
        previous_reason = cache.get("fusion_reason", "")

        keep_alert = (
            previous_status == "DAU HIEU MAT TAP TRUNG"
            and last_alert_time is not None
            and (float(current_time) - float(last_alert_time))
                < FUSION_ALERT_HOLD_SECONDS
        )

        if keep_alert:
            fusion_status = "DAU HIEU MAT TAP TRUNG"
            fusion_reason = previous_reason
        elif eye_active or head_active or body_turn_active:
            fusion_status = "THEO DOI"

            if eye_active:
                fusion_reason = "Nhắm mắt"
            elif head_active and body_turn_active:
                fusion_reason = "Lệch đầu + quay người"
            elif head_active:
                fusion_reason = "Lệch hướng"
            else:
                fusion_reason = (
                    f"Quay người ({body_turn_confirmations} lần xác nhận)"
                )
        else:
            fusion_status = "BINH THUONG"
            fusion_reason = ""

    cache["fusion_status"] = fusion_status
    cache["fusion_reason"] = fusion_reason
    cache["fusion_eye_seconds"] = eye_seconds
    cache["fusion_head_seconds"] = head_seconds
    cache["fusion_body_seconds"] = body_seconds

    return fusion_status, fusion_reason


def has_independent_multimodal_evidence(cache, current_time, fer_fresh, eng_fresh):
    """Only repeated NEW FER + DAiSEE inferences can trigger the sensitive alert.

    A single FER result may be cached for ~10 s and reused over hundreds of
    video frames. Frame count / signal timer is *not* independent evidence.
    Keep the separate observation times to prevent a lone sad-face prediction
    being described as a sustained state. This is deliberately conservative.
    """
    now = float(current_time)
    fer_at = float(cache.get("fer_last_time", -1e9))
    eng_at = float(cache.get("eng_last_time", -1e9))
    events = cache.setdefault("_anxiety_independent_samples", [])
    events[:] = [float(t) for t in events if 0 <= now - float(t) <= ANXIETY_SAMPLE_WINDOW_SECONDS]

    # Examine a FER prediction only once when its timestamp CHANGES.
    # If inference timestamps reset (e.g. restarted stream), reset history.
    previously_seen = float(cache.get("_anxiety_last_checked_fer_at", -1e9))
    if fer_at < previously_seen:
        events.clear()
        previously_seen = -1e9
    if fer_at > previously_seen:
        cache["_anxiety_last_checked_fer_at"] = fer_at
        strong_independent_evidence = (
            fer_fresh and eng_fresh
            and abs(fer_at - eng_at) <= ANXIETY_SAMPLE_SYNC_SECONDS
            and str(cache.get("fer_label", "")) in ("sad", "fear", "angry")
            and float(cache.get("fer_conf", 0.0)) >= ANXIETY_MIN_FER_TOP_CONF
            and float(cache.get("negative_emotion", 0.0)) >= ANXIETY_STRONG_NEGATIVE_PROB
            and float(cache.get("eng_low_score", 0.0)) >= ANXIETY_STRONG_LOW_ENG_PROB
        )
        if strong_independent_evidence:
            events.append(fer_at)
        else:
            # Require consecutive independent observations, not cherry-picked ones.
            events.clear()

    return (
        len(events) >= ANXIETY_MIN_INDEPENDENT_SAMPLES
        and events[-1] - events[-ANXIETY_MIN_INDEPENDENT_SAMPLES] >= ANXIETY_MIN_SAMPLE_SPAN_SECONDS
        and 0 <= now - events[-1] <= min(AI_RESULT_FRESH_SECONDS, 5.0)
        and fer_fresh and eng_fresh
    )


def update_final_fusion(cache, current_time, behavior_status, behavior_reason):
    """Escalate OBSERVED prolonged head turns, not inferred psychology.

    5-<10 seconds -> existing attention-review alert from behavior fusion.
    >=10 seconds -> CAN CHU Y. A stale/lost face must never accumulate time.
    FER/DAiSEE continue as descriptive estimates, not diagnoses.
    """
    eng_time = float(cache.get("eng_last_time", -1e9))
    eng_fresh = 0 <= float(current_time) - eng_time <= AI_RESULT_FRESH_SECONDS
    low_score = float(cache.get("eng_low_score", 0.0))
    low_active = eng_fresh and low_score >= LOW_ENGAGEMENT_THRESHOLD

    # update_behavior_fusion() computes this from independent fresh MediaPipe
    # observations; no time is added while the face is missing or uncertain.
    last_face = float(cache.get("face_last_observed_time", -1e9))
    face_fresh = 0 <= float(current_time) - last_face <= FACE_SIGNAL_FRESH_SECONDS
    head_active = face_fresh and cache.get("head_status") in HEAD_AWAY_STATUSES
    head_seconds = float(cache.get("fusion_head_duration", 0.0)) if head_active else 0.0

    # Do not relabel negative emotions as a mental-health warning.
    # Retain legacy cache keys for compatibility with telemetry and reports.
    cache["final_anxiety_since"] = None
    cache["final_anxiety_duration"] = 0.0
    cache["final_anxiety_seconds"] = 0.0
    low_seconds = update_signal_timer(cache, "final_low_eng", low_active, current_time)

    if head_seconds >= HEAD_AWAY_ESCALATE_SECONDS:
        final_status = "CAN CHU Y"
        final_reason = f"Quay mặt liên tục {head_seconds:.1f}s; cần giáo viên xem xét"
    elif behavior_status == "DAU HIEU MAT TAP TRUNG":
        final_status = "DAU HIEU MAT TAP TRUNG"
        final_reason = behavior_reason
    elif behavior_status == "THEO DOI" or low_active:
        final_status = "THEO DOI"
        final_reason = (behavior_reason if behavior_status == "THEO DOI" and behavior_reason
                        else "Engagement thấp đang theo dõi")
    else:
        final_status = "BINH THUONG"
        final_reason = ""

    cache["final_status"] = final_status
    cache["final_reason"] = final_reason
    cache["final_low_seconds"] = low_seconds
    return final_status, final_reason


if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {FACE_MODEL_PATH}"
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("======================================================")
print("WEEK 8 - MULTI PERSON FUSION AI V14 + BACKEND MULTI-TRACK PERFORMANCE")
print("======================================================")
print("YOLO model :", YOLO_MODEL_PATH)
print("Face model :", FACE_MODEL_PATH)
print("Pose model :", POSE_MODEL_PATH)
print("Device     :", device)
print("MediaPipe  :", getattr(mp, "__version__", "unknown"))
print("EAR threshold:", EAR_THRESHOLD)
print("Yaw threshold:", YAW_THRESHOLD)
print("Quay khuat sau:", FACE_MISSING_TURNED_SECONDS, "s")
print("Body status hold:", BODY_STATUS_HOLD_SECONDS, "s")
print("Vi tri: TU DONG hoc theo tung Track ID - khong can bam C.")
print("Fusion V14: frame-skip YOLO/Pose + AI worker nen + cache.")
print("Body turn alert:", BODY_TURN_ALERT_SECONDS, "s | xac nhan:", BODY_TURN_ALERT_CONFIRMATIONS)
print("Head+Body combined:", HEAD_BODY_COMBINED_SECONDS, "s")
print("Performance: YOLO moi", YOLO_EVERY_N_FRAMES, "frame | Pose moi", POSE_EVERY_N_FRAMES, "frame | imgsz", IMAGE_SIZE)
print("Dung mot minh KHONG bi xem la mat tap trung.")
print("Nhan Q: thoat.")
print("======================================================")

parser = argparse.ArgumentParser(
    description="Week 9 - Video Analyzer V14 + Backend"
)
parser.add_argument(
    "--source",
    default=str(CAMERA_INDEX),
    help=r"0 cho webcam hoac duong dan video, vi du D:\Doan\test.mp4"
)
parser.add_argument(
    "--analysis-run-id",
    default=None,
    help="UUID of this processing run (Backend supplies it for video)",
)
parser.add_argument(
    "--video-id",
    type=int,
    default=None,
    help="ID video trong bang videos cua Backend Week 9"
)
parser.add_argument(
    "--output",
    default="",
    help="Duong dan MP4 luu video da ve overlay AI"
)
parser.add_argument(
    "--no-display",
    action="store_true",
    help="Khong mo cua so OpenCV khi Backend phan tich video"
)
parser.add_argument(
    "--no-record-camera",
    action="store_true",
    help="Khong luu video khi dung camera truc tiep (mac dinh CO luu)."
)
args = parser.parse_args()
SOURCE, IS_VIDEO_FILE = parse_source(args.source)
VIDEO_ID = args.video_id
ANALYSIS_RUN_ID = args.analysis_run_id or str(uuid.uuid4())
try:
    ANALYSIS_RUN_ID = str(uuid.UUID(ANALYSIS_RUN_ID))
except ValueError:
    parser.error("--analysis-run-id phai la UUID hop le")
print("Analysis run:", ANALYSIS_RUN_ID)
# Every webcam invocation is recorded by default. Uploaded video analysis is unchanged.
CAMERA_RECORDING_ACTIVE = (
    not IS_VIDEO_FILE and VIDEO_ID is None
    and not args.no_record_camera
    and not args.output
)
OUTPUT_PATH = (
    Path(args.output).resolve()
    if args.output
    else (
        (BASE_DIR / "uploads" / "camera_recordings" /
         f"camera_{ANALYSIS_RUN_ID}.mp4")
        if CAMERA_RECORDING_ACTIVE else None
    )
)
CAMERA_DURATION_SECONDS = None
CAMERA_REGISTERED_WITH_BACKEND = False

RAW_OUTPUT_PATH = None

if OUTPUT_PATH is not None:
    RAW_OUTPUT_PATH = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.stem}_raw{OUTPUT_PATH.suffix}"
    )

print("Nguon       :", SOURCE)
print("Che do      :", "VIDEO" if IS_VIDEO_FILE else "WEBCAM")

yolo_model = YOLO(str(YOLO_MODEL_PATH))
fer_model, engagement_model, ENGAGEMENT_MODEL_PATH = load_ai_models()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.30,
    min_face_presence_confidence=0.30,
    min_tracking_confidence=0.30,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = FaceLandmarker.create_from_options(
    face_options
)


PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.30,
    min_pose_presence_confidence=0.30,
    min_tracking_confidence=0.30,
    output_segmentation_masks=False
)

pose_landmarker = PoseLandmarker.create_from_options(
    pose_options
)

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    face_landmarker.close()
    pose_landmarker.close()
    raise RuntimeError(
        f"Khong mo duoc nguon: {SOURCE}"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

source_fps = float(
    cap.get(cv2.CAP_PROP_FPS) or 0
)
if not math.isfinite(source_fps) or source_fps <= 1:
    source_fps = 25.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
)

# Mot so codec/container tra CAP_PROP_FRAME_COUNT = 0.
# Neu la file video, dem nhanh bang grab() de progress van chinh xac.
if IS_VIDEO_FILE and total_frames <= 0:
    counter_cap = cv2.VideoCapture(SOURCE)
    counted_frames = 0

    if counter_cap.isOpened():
        while True:
            ok = counter_cap.grab()

            if not ok:
                break

            counted_frames += 1

    counter_cap.release()

    if counted_frames > 0:
        total_frames = counted_frames

print(
    f"W9_META|{VIDEO_ID or 0}|TOTAL_FRAMES|{total_frames}",
    flush=True
)

# UI khong dung o 0% trong luc model dang khoi tao.
print(
    f"W9_PROGRESS|{VIDEO_ID or 0}|1",
    flush=True
)

video_writer = None
last_progress_report = 1

if OUTPUT_PATH is not None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    print(
        "Video AI raw:",
        RAW_OUTPUT_PATH
    )
    print(
        "Video AI web:",
        OUTPUT_PATH
    )

track_cache = {}
frame_index = 0
round_robin_start = 0
pose_round_robin_start = 0
ai_round_robin_start = 0
last_global_ai_inference_time = -1e9
crowd_face_only = False
crowd_low_since = None

# V14: 1 worker AI nen, khong block vong camera.
ai_job_queue = queue.Queue(maxsize=AI_QUEUE_MAXSIZE)
ai_stop_event = threading.Event()
ai_worker_busy = threading.Event()
ai_worker = threading.Thread(
    target=ai_worker_loop,
    args=(
        ai_job_queue,
        ai_stop_event,
        ai_worker_busy,
        fer_model,
        engagement_model,
    ),
    daemon=True,
    name="V14-AI-Worker",
)
ai_worker.start()

# Week 8 - Khoi: sender Backend multi-track chay thread rieng.
backend_sender = MultiTrackBackendSender()
backend_sender.start()

if CAMERA_RECORDING_ACTIVE:
    print("[CAMERA] DANG GHI VIDEO cho phien camera nay.")
    print("[CAMERA] Thu muc luu:", OUTPUT_PATH.parent)
    if backend_sender.enabled and REQUESTS_AVAILABLE:
        try:
            registration = requests.post(
                f"{BACKEND_URL.rstrip('/')}/api/camera-recordings/start",
                headers=backend_sender._headers(),
                json={"analysisRunId": ANALYSIS_RUN_ID},
                timeout=8,
            )
            registration.raise_for_status()
            CAMERA_REGISTERED_WITH_BACKEND = True
            print("[CAMERA] Da lien ket voi Lop/Buoi hoc tren Dashboard.")
        except Exception as error:
            print("[CAMERA] Canh bao: khong dang ky duoc video len Dashboard:", error)
            print("[CAMERA] Hay chon Lop/Buoi hoc truoc, file van duoc luu tren may.")
    else:
        print("[CAMERA] Backend chua san sang; chi luu video trong thu muc may.")

# V14: giu box/Track ID gan nhat o frame xen ke de giam tai YOLO.
last_detections = []

fps_smooth = 0.0
prev_time = time.time()
camera_record_started_monotonic = time.monotonic()

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Khong doc duoc frame camera.")
            break

        frame_index += 1

        if (
            OUTPUT_PATH is not None
            and video_writer is None
        ):
            frame_h, frame_w = frame.shape[:2]

            video_writer = cv2.VideoWriter(
                str(RAW_OUTPUT_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (frame_w, frame_h)
            )

            if not video_writer.isOpened():
                raise RuntimeError(
                    f"Khong tao duoc video raw: {RAW_OUTPUT_PATH}"
                )

        current_time = get_stream_time(cap, IS_VIDEO_FILE)

        # V14: YOLO chi chay moi N frame. Frame xen ke dung box/ID gan nhat.
        # Phu hop lop hoc vi hoc sinh thuong di chuyen cham, giam tai CPU dang ke.
        run_yolo = (
            frame_index == 1
            or (frame_index % YOLO_EVERY_N_FRAMES) == 0
            or not last_detections
        )

        if run_yolo:
            results = yolo_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[PERSON_CLASS],
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False
            )

            result = results[0]
            detections = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.boxes.id is not None
            ):

                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = map(int, box)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1] - 1, x2)
                    y2 = min(frame.shape[0] - 1, y2)

                    track_id = int(ids[i])

                    detections.append(
                        {
                            "id": track_id,
                            "box": (x1, y1, x2, y2),
                            "conf": float(confidences[i])
                        }
                    )

                    if track_id not in track_cache:
                        track_cache[track_id] = {
                            "ear": None,
                            "eye_status": "CHUA PHAN TICH",
                            "face_found": False,
                            "ever_seen_face": False,
                            "head_status": "CHUA PHAN TICH",
                            "yaw": None,
                            "pitch": None,
                            "roll": None,
                            "last_seen": frame_index,
                            "missing_since": None,
                            "body_found": False,
                            "body_status": "TU THE CHUA RO",
                            "body_turn_ratio": None,
                            "mean_knee_angle": None,
                            "body_last_reliable_status": None,
                            "body_last_reliable_time": 0.0,
                            "body_last_analyzed_time": 0.0,
                            "body_turn_confirmations": 0,
                            "body_turn_last_confirm_time": 0.0,
                            "seat_anchor": None,
                            "seat_away_since": None,
                            "seat_away": False,
                            "seat_dx_norm": None,
                            "seat_dy_norm": None,
                            "seat_auto_samples": [],
                            "seat_auto_status": "DANG HOC VI TRI",
                            "seat_auto_x_spread": None,
                            "seat_auto_y_spread": None,
                            "fusion_status": "BINH THUONG",
                            "fusion_reason": "",
                            "fusion_eye_since": None,
                            "fusion_head_since": None,
                            "fusion_body_turn_since": None,
                            "fusion_eye_duration": 0.0,
                            "fusion_head_duration": 0.0,
                            "fusion_body_turn_duration": 0.0,
                            "fusion_last_alert_time": None,
                            "face_bbox_norm": None,
                            "face_bbox_time": -1e9,
                            "face_last_observed_time": -1e9,
                            "eye_closed_distinct_samples": 0,
                            "crowd_face_only": False,
                            "ai_last_inference_time": -1e9,
                            "ai_pending": False,
                            "fer_probs": None,
                            "fer_label": None,
                            "fer_conf": 0.0,
                            "fer_last_time": -1e9,
                            "negative_emotion": 0.0,
                            "eng_probs": None,
                            "eng_label": None,
                            "eng_conf": 0.0,
                            "eng_low_score": 0.0,
                            "eng_last_time": -1e9,
                            "anxiety_score": 0.0,
                            "final_status": "BINH THUONG",
                            "final_reason": "",
                            "final_low_eng_since": None,
                            "final_anxiety_since": None,
                            "final_low_eng_duration": 0.0,
                            "final_anxiety_duration": 0.0
                        }

                    track_cache[track_id]["last_seen"] = frame_index

            last_detections = [
                {
                    "id": int(d["id"]),
                    "box": tuple(d["box"]),
                    "conf": float(d["conf"]),
                }
                for d in detections
            ]
        else:
            detections = [
                {
                    "id": int(d["id"]),
                    "box": tuple(d["box"]),
                    "conf": float(d["conf"]),
                }
                for d in last_detections
            ]

            # Giu Track ID dang thay o frame xen ke, tranh cache bi coi la cu.
            for detection in detections:
                tid = detection["id"]
                if tid in track_cache:
                    track_cache[tid]["last_seen"] = frame_index


        visible_ids = [
            d["id"]
            for d in detections
        ]

        previous_crowd_mode = crowd_face_only
        crowd_face_only, crowd_low_since = choose_face_only_mode(
            crowd_face_only, crowd_low_since, len(visible_ids), current_time
        )
        if previous_crowd_mode != crowd_face_only:
            print("[CROWD V16]", "FACE_ONLY" if crowd_face_only else "FULL",
                  "| persons:", len(visible_ids), flush=True)
        for tid in visible_ids:
            c = track_cache[tid]
            if c.get("crowd_face_only") != crowd_face_only:
                reset_crowd_transition(c, crowd_face_only)

        analyze_ids = []

        if visible_ids:
            count = min(
                MAX_FACE_ANALYSES_PER_FRAME,
                len(visible_ids)
            )

            for k in range(count):
                idx = (
                    round_robin_start + k
                ) % len(visible_ids)

                analyze_ids.append(
                    visible_ids[idx]
                )

            round_robin_start = (
                round_robin_start + count
            ) % len(visible_ids)


        pose_analyze_ids = []

        if (
            visible_ids
            and not crowd_face_only
            and (frame_index % POSE_EVERY_N_FRAMES) == 0
        ):
            pose_count = min(
                MAX_POSE_ANALYSES_PER_FRAME,
                len(visible_ids)
            )

            for k in range(pose_count):
                idx = (
                    pose_round_robin_start + k
                ) % len(visible_ids)
                pose_analyze_ids.append(visible_ids[idx])

            pose_round_robin_start = (
                pose_round_robin_start + pose_count
            ) % len(visible_ids)


        # Schedule FER only for a FRESH, sizable face. The old queue eagerly
        # spent its one worker on tracks with no face ROI, so visible faces
        # stayed at "Đang phân tích" longer than necessary. Both camera and
        # uploaded video use this same branch.
        ai_analyze_id = None
        if (
            visible_ids
            and not ai_worker_busy.is_set()
            and ai_job_queue.empty()
            and (current_time - last_global_ai_inference_time)
                >= AI_GLOBAL_INTERVAL_SECONDS
        ):
            for offset in range(len(visible_ids)):
                idx = (ai_round_robin_start + offset) % len(visible_ids)
                candidate_id = visible_ids[idx]
                candidate_cache = track_cache[candidate_id]
                if not ready_for_face_ai(candidate_cache, current_time):
                    continue
                if (
                    current_time
                    - float(candidate_cache.get("ai_last_inference_time", -1e9))
                    >= AI_PER_TRACK_INTERVAL_SECONDS
                ):
                    ai_analyze_id = candidate_id
                    ai_round_robin_start = (idx + 1) % len(visible_ids)
                    break


        face_count = 0
        head_not_forward_count = 0
        eyes_closed_count = 0
        body_count = 0
        seat_away_count = 0
        calibrated_seat_count = 0
        distracted_count = 0
        anxiety_count = 0
        ai_ready_count = 0

        person_labels = []

        for detection in detections:

            track_id = detection["id"]
            x1, y1, x2, y2 = detection["box"]

            person_w = x2 - x1
            person_h = y2 - y1

            person_crop = frame[
                y1:y2,
                x1:x2
            ]

            left_points = []
            right_points = []

            cache = track_cache[track_id]

            if track_id in pose_analyze_ids:
                body_found_now, body_status_now, body_metrics = analyze_body_pose(
                    pose_landmarker,
                    person_crop
                )

                cache["body_last_analyzed_time"] = current_time
                cache["body_turn_ratio"] = body_metrics.get("body_turn_ratio")
                cache["mean_knee_angle"] = body_metrics.get("mean_knee_angle")

                if (
                    body_found_now
                    and body_status_now in RELIABLE_BODY_STATUSES
                ):
                    # Kết quả đáng tin: cập nhật ngay cho đúng Track ID.
                    cache["body_found"] = True
                    cache["body_status"] = body_status_now
                    cache["body_last_reliable_status"] = body_status_now
                    cache["body_last_reliable_time"] = current_time

                    # V11: dem so lan Pose thuc su xac nhan quay nguoi.
                    # Cache giu nhan de giao dien on dinh KHONG duoc tinh la
                    # mot lan xac nhan moi.
                    if body_status_now in BODY_TURN_STATUSES:
                        last_confirm = float(
                            cache.get("body_turn_last_confirm_time", 0.0)
                        )

                        if (
                            last_confirm > 0.0
                            and (current_time - last_confirm) <= 2.0
                        ):
                            cache["body_turn_confirmations"] = min(
                                int(cache.get("body_turn_confirmations", 0)) + 1,
                                20
                            )
                        else:
                            cache["body_turn_confirmations"] = 1

                        cache["body_turn_last_confirm_time"] = current_time
                    else:
                        # Mot ket qua Body Pose ro rang khong quay nguoi se reset.
                        cache["body_turn_confirmations"] = 0
                        cache["body_turn_last_confirm_time"] = current_time

                else:
                    # Nếu frame hiện tại bị che/chưa rõ, giữ kết quả tốt cuối cùng
                    # trong vài giây thay vì làm nhãn biến mất/chớp liên tục.
                    last_status = cache.get("body_last_reliable_status")
                    last_time = cache.get("body_last_reliable_time", 0.0)

                    if (
                        last_status in RELIABLE_BODY_STATUSES
                        and (current_time - last_time) <= BODY_STATUS_HOLD_SECONDS
                    ):
                        cache["body_found"] = True
                        cache["body_status"] = last_status
                    else:
                        cache["body_found"] = False
                        cache["body_status"] = body_status_now

            if track_id in analyze_ids:

                (
                    face_found,
                    ear,
                    eye_status,
                    head_pose,
                    left_points,
                    right_points,
                    face_bbox_norm
                ) = analyze_face(
                    face_landmarker,
                    person_crop,
                    upper_body_ratio=0.65 if crowd_face_only else UPPER_BODY_RATIO,
                )

                cache["face_found"] = face_found

                if face_found:

                    cache["ever_seen_face"] = True
                    cache["missing_since"] = None

                    prev_face_t = float(cache.get("face_last_observed_time", -1e9))
                    if eye_status == "NHAM MAT":
                        if 0 <= current_time - prev_face_t <= FACE_SIGNAL_FRESH_SECONDS                                 and int(cache.get("eye_closed_distinct_samples", 0)):
                            cache["eye_closed_distinct_samples"] += 1
                        else:
                            cache["eye_closed_distinct_samples"] = 1
                    else:
                        cache["eye_closed_distinct_samples"] = 0
                    cache["face_last_observed_time"] = float(current_time)
                    cache["ear"] = ear
                    cache["eye_status"] = eye_status
                    cache["face_bbox_norm"] = face_bbox_norm
                    cache["face_bbox_time"] = current_time

                    if head_pose is not None:
                        yaw = head_pose["yaw"]
                        pitch = head_pose["pitch"]
                        roll = head_pose["roll"]

                        cache["yaw"] = yaw
                        cache["pitch"] = pitch
                        cache["roll"] = roll

                        cache["head_status"] = classify_head_pose(
                            yaw, pitch,
                            head_pose.get("landmarks"),
                            head_pose.get("image_size")
                        )

                    else:
                        cache["yaw"] = None
                        cache["pitch"] = None
                        cache["roll"] = None
                        cache["head_status"] = "POSE UNKNOWN"

                    # One fresh facial landmark observation; both webcam and
                    # uploaded video execute this identical pathway.
                    update_observed_face_durations(
                        cache, cache["head_status"], cache["eye_status"], current_time
                    )

                else:

                    # Xoa ket qua cu de tranh hien stale yaw/pitch/ear
                    cache["ear"] = None
                    cache["eye_status"] = eye_status
                    cache["eye_closed_distinct_samples"] = 0
                    cache["face_bbox_norm"] = None
                    cache["face_bbox_time"] = -1e9
                    cache["yaw"] = None
                    cache["pitch"] = None
                    cache["roll"] = None

                    if (
                        person_w < MIN_PERSON_WIDTH
                        or
                        person_h < MIN_PERSON_HEIGHT
                    ):

                        cache["missing_since"] = None
                        cache["head_status"] = (
                            "QUA XA / MAT QUA NHO"
                        )

                    else:

                        if cache["missing_since"] is None:
                            cache["missing_since"] = current_time

                        missing_duration = (
                            current_time
                            - cache["missing_since"]
                        )

                        # Missing face is ambiguous; do not invent head direction.
                        cache["head_status"] = "DANG TIM MAT"


            # =================================================
            # VI TRI TU DONG - theo đúng Track ID
            # =================================================
            if not crowd_face_only:
                auto_calibrated_now = update_auto_seat_anchor(
                    cache,
                    (x1, y1, x2, y2),
                    current_time
                )

                if auto_calibrated_now:
                    print(
                        f"[AUTO VI TRI] Track ID {track_id}: da tu dong khoa vi tri chuan."
                    )

                anchor = cache.get("seat_anchor")

                if anchor is not None:
                    calibrated_seat_count += 1

                    current_cx = (x1 + x2) / 2.0
                    current_bottom_y = float(y2)

                    dx_norm = abs(current_cx - anchor["cx"]) / max(anchor["w"], 1.0)
                    dy_norm = abs(current_bottom_y - anchor["bottom_y"]) / max(anchor["h"], 1.0)

                    cache["seat_dx_norm"] = dx_norm
                    cache["seat_dy_norm"] = dy_norm

                    if cache.get("seat_away", False):
                        # Hysteresis: phải trở vào vùng nhỏ hơn mới reset.
                        back_inside = (
                            dx_norm <= SEAT_RETURN_X_RATIO
                            and dy_norm <= SEAT_RETURN_Y_RATIO
                        )

                        if back_inside:
                            cache["seat_away"] = False
                            cache["seat_away_since"] = None
                    else:
                        outside = (
                            dx_norm >= SEAT_EXIT_X_RATIO
                            or dy_norm >= SEAT_EXIT_Y_RATIO
                        )

                        if outside:
                            if cache.get("seat_away_since") is None:
                                cache["seat_away_since"] = current_time

                            if (
                                current_time - cache["seat_away_since"]
                                >= SEAT_EXIT_SECONDS
                            ):
                                cache["seat_away"] = True
                        else:
                            cache["seat_away_since"] = None
                            cache["seat_away"] = False
                else:
                    cache["seat_away_since"] = None
                    cache["seat_away"] = False

            else:
                # FACE_ONLY never infers a seat or uses body movement.
                cache["seat_away"] = False
                cache["seat_away_since"] = None

            # =================================================
            # V13: FER + DAiSEE ENGAGEMENT CHAY NEN
            # =================================================
            if ai_analyze_id == track_id:
                # ROI qua nho thi bo qua AI de tranh ton CPU ma ket qua kem tin cay.
                if (
                    person_w >= AI_MIN_PERSON_WIDTH
                    and person_h >= AI_MIN_PERSON_HEIGHT
                    and not cache.get("ai_pending", False)
                    and ready_for_face_ai(cache, current_time)
                ):
                    try:
                        cache["ai_pending"] = True
                        # Dat moc ngay khi xep job de scheduler khong xep trung ID.
                        cache["ai_last_inference_time"] = float(current_time)
                        ai_job_queue.put_nowait(
                            (
                                track_id,
                                person_crop.copy(),
                                cache,
                                float(current_time),
                            )
                        )
                        last_global_ai_inference_time = float(current_time)
                    except queue.Full:
                        cache["ai_pending"] = False
                else:
                    # Cho ID qua nho nghi mot nhip truoc khi thu lai.
                    if person_w < AI_MIN_PERSON_WIDTH or person_h < AI_MIN_PERSON_HEIGHT:
                        cache["ai_last_inference_time"] = float(current_time)

            # In crowded scenes, a Track may not get a fresh face observation
            # for several frames. Old face state must not accumulate alert time.
            fresh_face = (
                0 <= current_time - float(cache.get("face_last_observed_time", -1e9))
                <= FACE_SIGNAL_FRESH_SECONDS
            )
            if not fresh_face:
                cache["eye_status"] = "CHUA PHAN TICH"
                cache["ear"] = None
                cache["eye_closed_distinct_samples"] = 0
                cache["head_status"] = "DANG TIM MAT"
                cache["yaw"] = None
                cache["pitch"] = None
                cache["roll"] = None
                cache["face_found"] = False
                cache["face_bbox_norm"] = None
                cache["face_bbox_time"] = -1e9
            if crowd_face_only:
                cache["seat_away"] = False
                cache["body_status"] = "TU THE CHUA RO"

            ear = cache["ear"]
            eye_status = cache["eye_status"]
            head_status = cache["head_status"]
            yaw = cache["yaw"]
            pitch = cache["pitch"]
            body_status = cache["body_status"]
            seat_away = cache.get("seat_away", False)

            if seat_away:
                seat_away_count += 1

            if cache["body_status"] in RELIABLE_BODY_STATUSES:
                body_count += 1

            if cache["face_found"]:
                face_count += 1

            if eye_status == "NHAM MAT":
                eyes_closed_count += 1

            if head_status in (
                "QUAY TRAI",
                "QUAY PHAI",
                "CUI DAU",
                "NGUOC LEN",
                "QUAY KHUAT / KHONG NHIN PHIA TRUOC"
            ):
                head_not_forward_count += 1


            # =================================================
            # V12: FUSION HANH VI + FER + ENGAGEMENT RIENG ID
            # =================================================
            behavior_status, behavior_reason = update_behavior_fusion(
                cache,
                current_time
            )

            final_status, final_reason = update_final_fusion(
                cache,
                current_time,
                behavior_status,
                behavior_reason
            )

            # =============================================
            # WEEK 8 - KHOI: GUI TUNG TRACK ID SANG BACKEND
            # =============================================
            student_id = TRACK_TO_STUDENT_ID.get(track_id)

            backend_payload = {
                "trackId": int(track_id),
                "analysisRunId": ANALYSIS_RUN_ID,
                "positionX": round(((x1 + x2) / 2.0) / frame.shape[1], 5),
                "positionY": round(float(y2) / frame.shape[0], 5),
                "videoId": int(VIDEO_ID) if VIDEO_ID is not None else None,
                "ear": float(ear) if ear is not None else None,
                "eyeStatus": eye_status,
                "headStatus": head_status,
                "bodyStatus": body_status,
                "seatAway": bool(seat_away),
                "emotion": cache.get("fer_label"),
                "emotionConfidence": float(cache.get("fer_conf", 0.0))
                    if cache.get("fer_label") else None,
                "engagement": cache.get("eng_label"),
                "engagementConfidence": float(cache.get("eng_conf", 0.0))
                    if cache.get("eng_label") else None,
                "lowScore": float(cache.get("eng_low_score", 0.0)),
                "anxietyScore": float(cache.get("anxiety_score", 0.0)),
                "fusionStatus": final_status,
                "fusionReason": final_reason or "",
                "sourceVersion": BACKEND_SOURCE_VERSION,
                "capturedAt": datetime.now(timezone.utc).isoformat(),
            }

            if student_id is not None:
                backend_payload["studentId"] = int(student_id)

            backend_sender.maybe_send(
                track_id,
                backend_payload,
                final_status,
            )

            base_display, base_color = get_person_display(
                head_status,
                eye_status,
                body_status,
                seat_away=seat_away
            )

            fer_fresh = (
                current_time - float(cache.get("fer_last_time", -1e9))
                <= AI_RESULT_FRESH_SECONDS
            )
            eng_fresh = (
                current_time - float(cache.get("eng_last_time", -1e9))
                <= AI_RESULT_FRESH_SECONDS
            )
            if fer_fresh or eng_fresh:
                ai_ready_count += 1

            if final_status == "CAN CHU Y":
                anxiety_count += 1
                box_color = (0, 165, 255)  # Head turn stays amber/yellow.
                display_status = f"Cần chú ý • {final_reason}"

            elif final_status == "DAU HIEU MAT TAP TRUNG":
                distracted_count += 1
                box_color = (0, 0, 255)
                if final_reason:
                    display_status = f"Mất tập trung • {final_reason}"
                else:
                    display_status = "Mất tập trung"

            elif final_status == "THEO DOI":
                box_color = (0, 165, 255)
                if final_reason:
                    display_status = f"Theo dõi • {final_reason}"
                else:
                    display_status = f"Theo dõi • {base_display}"

            else:
                box_color = base_color
                display_status = base_display

            # When 6+ people, the outline is a FACE DIRECTION indicator.
            # Do not let a separate "THEO DOI" status paint a forward-facing
            # face amber while its label says "Nhìn về camera".
            face_eye_seconds = verified_observation_seconds(cache, "eye", current_time)
            face_head_seconds = verified_observation_seconds(cache, "head", current_time)
            direction_label, box_color, fer_label_line = face_direction_and_fer_labels(
                head_status, cache.get("fer_label"), cache.get("fer_conf"), fer_fresh,
                eye_status, face_eye_seconds, face_head_seconds,
            )
            # Color is ALWAYS about visible face/eye behavior (green/yellow/red/gray).
            # Fusion alerts remain separate and still reach Backend; no purple
            # or red anxiety labels are painted over a forward-looking face.
            if crowd_face_only:
                display_status = direction_label
            else:
                display_status = direction_label
                if body_status in RELIABLE_BODY_STATUSES:
                    display_status += " • " + DISPLAY_TEXT.get(body_status, body_status)
            crowd_direction = direction_label
            crowd_fer = fer_label_line

            # In FACE_ONLY mode draw ONLY reliable face rectangles, not body boxes.
            display_box = (x1, y1, x2, y2)
            if crowd_face_only:
                norm = cache.get("face_bbox_norm")
                if norm is None or not fresh_face:
                    display_box = None
                else:
                    fx1, fy1, fx2, fy2 = norm
                    face_coords = (
                        max(0, min(frame.shape[1] - 1, x1 + int(fx1 * person_w))),
                        max(0, min(frame.shape[0] - 1, y1 + int(fy1 * person_h))),
                        max(0, min(frame.shape[1] - 1, x1 + int(fx2 * person_w))),
                        max(0, min(frame.shape[0] - 1, y1 + int(fy2 * person_h))),
                    )
                    if (face_coords[2] - face_coords[0] >= 16
                            and face_coords[3] - face_coords[1] >= 16):
                        display_box = face_coords
                    else:
                        display_box = None

            if display_box is not None:
                bx1, by1, bx2, by2 = display_box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                # Keep head-direction and FER separate, readable and honest.
                # Crowd: line 1 is colored only by head direction; line 2
                # shows FER if a fresh inference exists (else waiting).
                person_labels.append((
                    f"ID {track_id} • {crowd_direction}" if crowd_face_only
                    else f"ID {track_id} • {display_status}",
                    bx1, max(24, by1 - 5), box_color,
                ))
                if crowd_face_only:
                    person_labels.append((
                        crowd_fer, bx1, min(frame.shape[0] - 8, by2 + 24),
                        (200, 200, 200),
                    ))
                else:
                    # Small-group mode also shows FER status, even when the
                    # asynchronous model hasn't produced a FRESH result yet.
                    person_labels.append((
                        fer_label_line, bx1, min(frame.shape[0] - 8, by1 + 28),
                        (200, 200, 200),
                    ))


            if (
                SHOW_DEBUG_EYE_LANDMARKS
                and track_id in analyze_ids
                and not crowd_face_only
                and cache["face_found"]
            ):

                for px, py in (
                    left_points
                    + right_points
                ):
                    cv2.circle(
                        frame,
                        (x1 + px, y1 + py),
                        2,
                        (255, 0, 255),
                        -1
                    )


        old_ids = [
            track_id
            for track_id, data
            in track_cache.items()
            if (
                frame_index
                - data["last_seen"]
                > CACHE_MAX_AGE_FRAMES
            )
        ]

        for old_id in old_ids:
            del track_cache[old_id]


        now = time.time()

        dt = max(
            now - prev_time,
            1e-6
        )

        fps_now = 1.0 / dt
        prev_time = now

        if fps_smooth == 0.0:
            fps_smooth = fps_now
        else:
            fps_smooth = (
                0.90 * fps_smooth
                + 0.10 * fps_now
            )


        hud_lines = [
            (
                f"Người: {len(detections)}  |  "
                f"Mặt: {face_count}  |  "
                f"Tư thế rõ: {body_count}  |  "
                f"Chế độ: {'MẶT (>5)' if crowd_face_only else 'ĐẦY ĐỦ (≤5)'}"
            ),
            (
                f"Lệch hướng: {head_not_forward_count}  |  "
                f"Nhắm mắt: {eyes_closed_count}  |  "
                f"Rời chỗ: {seat_away_count}  |  "
                f"AI: {ai_ready_count}  |  "
                f"Mất tập trung: {distracted_count}  |  "
                f"Cần chú ý: {anxiety_count}  |  "
                f"FPS: {fps_smooth:.1f}"
            ),
        ]

        # Han che cac nhan 1 dong bi de len nhau khi nhieu hoc sinh dung gan nhau.
        person_labels.sort(key=lambda item: (item[2], item[1]))
        adjusted_labels = []
        # Keep labels near each face; DO NOT stack all seven across same column.
        # Move a label only if its horizontal span overlaps another label.
        occupied = []
        for text_label, lx, ly, lcolor in person_labels:
            width_estimate = min(frame.shape[1] - 6, max(110, len(text_label) * 9))
            adjusted_y = min(max(24, ly), frame.shape[0] - 8)
            for _ in range(12):
                conflict = any(
                    abs(adjusted_y - oy) < 22
                    and lx < ox + ow and ox < lx + width_estimate
                    for ox, oy, ow in occupied
                )
                if not conflict:
                    break
                adjusted_y = min(frame.shape[0] - 8, adjusted_y + 23)
            adjusted_labels.append((text_label, lx, adjusted_y, lcolor))
            occupied.append((lx, adjusted_y, width_estimate))

        frame = draw_unicode_labels(
            frame,
            adjusted_labels,
            hud_lines
        )


        if video_writer is not None:
            video_writer.write(frame)

        if total_frames > 0:
            progress = int(
                min(
                    99,
                    max(
                        1,
                        round(
                            frame_index
                            * 100
                            / total_frames
                        )
                    )
                )
            )
        else:
            # Fallback hiem gap: van cho nguoi dung thay analyzer dang tien len.
            # Chi toi da 95%; 100% chi set khi process ket thuc thanh cong.
            progress = int(
                min(
                    95,
                    max(
                        1,
                        frame_index // 2
                    )
                )
            )

        if progress > last_progress_report:
            last_progress_report = progress

            print(
                f"W9_PROGRESS|{VIDEO_ID or 0}|{progress}",
                flush=True
            )

        if not args.no_display:
            cv2.imshow(
                "Week 9 - Video Analyzer V14",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

        # V9: khong can C/R. Moi Track ID tu hoc vi tri khi on dinh.


finally:
    if CAMERA_RECORDING_ACTIVE:
        CAMERA_DURATION_SECONDS = max(0.0, time.monotonic() - camera_record_started_monotonic)

    # Dung Backend sender truoc, camera/AI khong bi block boi HTTP.
    try:
        backend_sender.close()
    except Exception:
        pass

    # Dung worker AI gon gang.
    ai_stop_event.set()
    try:
        if ai_worker.is_alive():
            ai_worker.join(timeout=1.0)
    except Exception:
        pass

    cap.release()

    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass

    face_landmarker.close()
    pose_landmarker.close()
    cv2.destroyAllWindows()


# ============================================================
# WEEK 9 - CHUYEN VIDEO AI SANG H.264 DE TRINH DUYET PHAT DUOC
# ============================================================
if (
    OUTPUT_PATH is not None
    and RAW_OUTPUT_PATH is not None
    and RAW_OUTPUT_PATH.exists()
):
    ffmpeg_exe = shutil.which("ffmpeg")

    if not ffmpeg_exe and imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

    if not ffmpeg_exe:
        raise RuntimeError(
            "Khong tim thay FFmpeg. Hay chay: "
            "python -m pip install imageio-ffmpeg"
        )

    if OUTPUT_PATH.exists():
        try:
            OUTPUT_PATH.unlink()
        except Exception:
            pass

    print(
        "Dang chuyen video AI sang H.264 de Web phat duoc...",
        flush=True
    )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(RAW_OUTPUT_PATH),
        "-an",
    ]
    if CAMERA_RECORDING_ACTIVE and CAMERA_DURATION_SECONDS and frame_index > 0:
        nominal_seconds = frame_index / max(1.0, source_fps)
        ratio = min(100.0, max(0.01, CAMERA_DURATION_SECONDS / max(nominal_seconds, 0.001)))
        # The AI processes fewer frames than realtime; stretch timestamps instead
        # of playing 20 minutes of camera capture in just 3 minutes.
        cmd.extend(["-vf", f"setpts={ratio:.6f}*(PTS-STARTPTS)", "-r", "25"])
    cmd.extend([
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT_PATH),
    ])

    conversion = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if conversion.returncode != 0:
        raise RuntimeError(
            "FFmpeg chuyen H.264 that bai: "
            + conversion.stderr[-1200:]
        )

    if (
        not OUTPUT_PATH.exists()
        or OUTPUT_PATH.stat().st_size <= 0
    ):
        raise RuntimeError(
            "FFmpeg khong tao duoc video H.264 hop le."
        )

    try:
        RAW_OUTPUT_PATH.unlink()
    except Exception:
        pass

    print(
        f"W9_VIDEO_READY|{VIDEO_ID or 0}|{OUTPUT_PATH}",
        flush=True
    )

    if CAMERA_RECORDING_ACTIVE and CAMERA_REGISTERED_WITH_BACKEND:
        try:
            response = requests.post(
                f"{BACKEND_URL.rstrip('/')}/api/camera-recordings/finish",
                headers=backend_sender._headers(),
                json={
                    "analysisRunId": ANALYSIS_RUN_ID,
                    "durationSeconds": CAMERA_DURATION_SECONDS,
                },
                timeout=10,
            )
            response.raise_for_status()
            print("[CAMERA] DA LUU VIDEO. Mo Dashboard > Camera > Video da quay.")
        except Exception as error:
            print("[CAMERA] KHONG CAP NHAT DUOC Dashboard:", error)
            print("[CAMERA] File MP4 van con tren may:", OUTPUT_PATH)


print("")
print("======================================================")
print("KET THUC WEEK 9 - VIDEO ANALYZER V14 + BACKEND")
print("Backend sent  :", backend_sender.sent_count)
print("Backend failed:", backend_sender.failed_count)
print("Alerts created:", backend_sender.alert_count)
print("======================================================")
