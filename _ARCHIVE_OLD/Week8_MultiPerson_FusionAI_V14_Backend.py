import os
import cv2
import time
import math
import urllib.request
import argparse
import threading
import queue
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
MAX_FACE_ANALYSES_PER_FRAME = 1
MAX_POSE_ANALYSES_PER_FRAME = 1
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
EYE_CLOSED_ALERT_SECONDS = 1.0
HEAD_AWAY_ALERT_SECONDS = 3.0

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
AI_GLOBAL_INTERVAL_SECONDS = 1.30
AI_PER_TRACK_INTERVAL_SECONDS = 3.50
AI_RESULT_FRESH_SECONDS = 10.0
FACE_BBOX_FRESH_SECONDS = 4.5
AI_EMA_ALPHA = 0.35
PRINT_AI_RESULTS = False
AI_QUEUE_MAXSIZE = 1
AI_MIN_PERSON_WIDTH = 110
AI_MIN_PERSON_HEIGHT = 160

# Threshold Low Engagement da chon tren Validation o Tuan 6/7.
LOW_ENGAGEMENT_THRESHOLD = 0.27
LOW_ENGAGEMENT_ALERT_SECONDS = 2.0

# Fusion dau hieu lo au: chi la canh bao ho tro, KHONG phai chan doan tam ly.
NEGATIVE_EMOTION_WEIGHT = 0.75
ENGAGEMENT_WEIGHT = 0.25
ANXIETY_FUSION_THRESHOLD = 0.45
ANXIETY_ALERT_SECONDS = 3.0

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
            self.email = env.get("ADMIN_EMAIL")
            self.password = env.get("ADMIN_PASSWORD")

            if not self.email or not self.password:
                raise RuntimeError(
                    "Thieu ADMIN_EMAIL hoac ADMIN_PASSWORD trong D:\\Doan\\.env"
                )

            self.login()

            self.worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="V14-Backend-MultiTrack",
            )
            self.worker.start()

            print("[BACKEND] Week 8 Multi-Track ket noi OK:", self.base_url)
            print(
                "[BACKEND] Moi Track gui toi da 1 lan/",
                f"{self.interval_per_track:.1f}s | global <= {BACKEND_MAX_REQUESTS_PER_SECOND:.1f} req/s"
            )
            print("[BACKEND] Track ID duoc luu rieng, khong gia lam studentId.")

        except Exception as error:
            self.enabled = False
            print("[BACKEND] Khong khoi tao duoc:", error)
            print("[BACKEND] V14 van tiep tuc camera/AI binh thuong.")

    def login(self):
        response = self.session.post(
            f"{self.base_url}/api/auth/login",
            json={
                "email": self.email,
                "password": self.password,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        token = response.json().get("token")
        if not token:
            raise RuntimeError("Backend khong tra JWT token.")
        self.token = token

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

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
            self.login()
            response = self.session.post(
                f"{self.base_url}/api/ai/results",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
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


def classify_head_pose(yaw, pitch):
    # Ket qua test thuc te tren camera cua ban:
    # quay trai -> yaw goc duong, quay phai -> yaw goc am.
    # Dao dau yaw mot lan de chuan hoa theo huong nguoi dung.
    yaw_display = -yaw

    if yaw_display <= -YAW_THRESHOLD:
        return "QUAY TRAI"

    if yaw_display >= YAW_THRESHOLD:
        return "QUAY PHAI"

    if pitch >= PITCH_DOWN_THRESHOLD:
        return "CUI DAU"

    if pitch <= PITCH_UP_THRESHOLD:
        return "NGUOC LEN"

    return "NHIN THANG"


def analyze_face(face_landmarker, person_crop):
    if person_crop is None or person_crop.size == 0:
        return False, None, "NO FACE", None, [], [], None

    h, w = person_crop.shape[:2]

    if h < MIN_PERSON_HEIGHT or w < MIN_PERSON_WIDTH:
        return False, None, "TOO SMALL", None, [], [], None

    upper_h = max(1, int(h * UPPER_BODY_RATIO))
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

    left_ear = eye_aspect_ratio(left_points_up)
    right_ear = eye_aspect_ratio(right_points_up)
    ear = (left_ear + right_ear) / 2.0

    eye_status = (
        "NHAM MAT"
        if ear < EAR_THRESHOLD
        else "MO MAT"
    )

    head_pose = estimate_head_pose(
        landmarks,
        upscaled.shape[1],
        upscaled.shape[0]
    )

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

    # Engagement: dung person ROI vi V12 can ket qua RIENG tung hoc sinh.
    eng_new = predict_probs(engagement_model, person_crop)
    if eng_new is not None and len(eng_new) >= 4:
        eng_probs = ema_probs(cache.get("eng_probs"), eng_new[:4])
        cache["eng_probs"] = eng_probs
        cache["eng_last_time"] = float(current_time)

        eng_idx = int(np.argmax(eng_probs))
        cache["eng_label"] = ENGAGEMENT_CLASSES[eng_idx]
        cache["eng_conf"] = float(eng_probs[eng_idx])
        cache["eng_low_score"] = float(eng_probs[0] + eng_probs[1])

    # FER: chi chay neu bbox khuon mat con moi.
    face_bbox_time = float(cache.get("face_bbox_time", -1e9))
    face_bbox_fresh = (
        cache.get("face_bbox_norm") is not None
        and (float(current_time) - face_bbox_time) <= FACE_BBOX_FRESH_SECONDS
    )

    if face_bbox_fresh:
        face_crop = crop_face_from_norm(
            person_crop,
            cache.get("face_bbox_norm")
        )

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

    eye_active = (eye_status == "NHAM MAT")
    head_active = (head_status in HEAD_AWAY_STATUSES)

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
        body_status in BODY_TURN_STATUSES
        and body_fresh
        and body_turn_confirmations >= BODY_TURN_START_CONFIRMATIONS
    )

    eye_seconds = update_signal_timer(
        cache, "fusion_eye", eye_active, current_time
    )
    head_seconds = update_signal_timer(
        cache, "fusion_head", head_active, current_time
    )
    body_seconds = update_signal_timer(
        cache, "fusion_body_turn", body_turn_active, current_time
    )

    reasons = []

    if eye_seconds >= EYE_CLOSED_ALERT_SECONDS:
        reasons.append("Nhắm mắt kéo dài")

    if head_seconds >= HEAD_AWAY_ALERT_SECONDS:
        if head_status == "CUI DAU":
            reasons.append("Cúi đầu kéo dài")
        elif head_status == "QUAY KHUAT / KHONG NHIN PHIA TRUOC":
            reasons.append("Không nhìn phía trước")
        else:
            reasons.append("Quay đầu kéo dài")

    # Hai tin hieu doc lap cung keo dai -> tin cay hon Body Pose don le.
    combined_head_body = (
        head_active
        and body_turn_active
        and head_seconds >= HEAD_BODY_COMBINED_SECONDS
        and body_seconds >= HEAD_BODY_COMBINED_SECONDS
    )

    if combined_head_body:
        reasons.append("Quay đầu + quay người kéo dài")
    elif (
        body_seconds >= BODY_TURN_ALERT_SECONDS
        and body_turn_confirmations >= BODY_TURN_ALERT_CONFIRMATIONS
    ):
        reasons.append("Quay người kéo dài")

    # seat_away da co bo dem SEAT_EXIT_SECONDS tu V9/V10.
    if seat_away:
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


def update_final_fusion(cache, current_time, behavior_status, behavior_reason):
    """
    V13 Fusion cuoi cung theo tung Track ID:
    - Hanh vi V11 la nhanh chinh cho mat tap trung.
    - DAiSEE Low Engagement la tin hieu ho tro.
    - FER + Engagement tao diem CANH BAO DAU HIEU LO AU.
    - Khong duoc hieu la chan doan tam ly.
    """
    fer_time = float(cache.get("fer_last_time", -1e9))
    eng_time = float(cache.get("eng_last_time", -1e9))

    fer_fresh = (float(current_time) - fer_time) <= AI_RESULT_FRESH_SECONDS
    eng_fresh = (float(current_time) - eng_time) <= AI_RESULT_FRESH_SECONDS

    low_score = float(cache.get("eng_low_score", 0.0))
    anxiety_score = float(cache.get("anxiety_score", 0.0))

    low_active = (
        eng_fresh
        and low_score >= LOW_ENGAGEMENT_THRESHOLD
    )

    # Can co FER con moi moi cho phep kich hoat dau hieu lo au.
    # Engagement co the ho tro diem, nhung khong tu no tao canh bao lo au.
    anxiety_active = (
        fer_fresh
        and anxiety_score >= ANXIETY_FUSION_THRESHOLD
    )

    low_seconds = update_signal_timer(
        cache, "final_low_eng", low_active, current_time
    )
    anxiety_seconds = update_signal_timer(
        cache, "final_anxiety", anxiety_active, current_time
    )

    if anxiety_seconds >= ANXIETY_ALERT_SECONDS:
        final_status = "DAU HIEU LO AU"

        fer_label = cache.get("fer_label")
        fer_text = FER_DISPLAY.get(fer_label, fer_label) if fer_label else "Cảm xúc tiêu cực"
        final_reason = f"{fer_text} + tín hiệu AI kéo dài"

    elif behavior_status == "DAU HIEU MAT TAP TRUNG":
        final_status = "DAU HIEU MAT TAP TRUNG"
        final_reason = behavior_reason

    elif (
        low_seconds >= LOW_ENGAGEMENT_ALERT_SECONDS
        and behavior_status == "THEO DOI"
    ):
        # V13: DAiSEE Engagement chi la tin hieu HO TRO.
        # Khong de Low Engagement mot minh ket luan mat tap trung.
        final_status = "DAU HIEU MAT TAP TRUNG"
        if behavior_reason:
            final_reason = f"{behavior_reason} + Engagement thấp"
        else:
            final_reason = "Tín hiệu hành vi + Engagement thấp"

    elif (
        behavior_status == "THEO DOI"
        or low_active
        or anxiety_active
    ):
        final_status = "THEO DOI"

        if behavior_status == "THEO DOI" and behavior_reason:
            final_reason = behavior_reason
        elif anxiety_active:
            final_reason = "Tín hiệu cảm xúc đang theo dõi"
        else:
            final_reason = "Engagement thấp đang theo dõi"

    else:
        final_status = "BINH THUONG"
        final_reason = ""

    cache["final_status"] = final_status
    cache["final_reason"] = final_reason
    cache["final_low_seconds"] = low_seconds
    cache["final_anxiety_seconds"] = anxiety_seconds

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
    description="Week 8 V14 - Multi-person Fusion AI performance cho webcam/video"
)
parser.add_argument(
    "--source",
    default=str(CAMERA_INDEX),
    help=r"0 cho webcam hoac duong dan video, vi du D:\Doan\test.mp4"
)
args = parser.parse_args()
SOURCE, IS_VIDEO_FILE = parse_source(args.source)

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

track_cache = {}
frame_index = 0
round_robin_start = 0
pose_round_robin_start = 0
ai_round_robin_start = 0
last_global_ai_inference_time = -1e9

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

# V14: giu box/Track ID gan nhat o frame xen ke de giam tai YOLO.
last_detections = []

fps_smooth = 0.0
prev_time = time.time()

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Khong doc duoc frame camera.")
            break

        frame_index += 1
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


        # V14: chi xep AI job khi worker dang ranh.
        # Khong cho hang doi day de tranh TensorFlow chiem CPU lien tuc.
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

                if candidate_cache.get("ai_pending", False):
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
                    person_crop
                )

                cache["face_found"] = face_found

                if face_found:

                    cache["ever_seen_face"] = True
                    cache["missing_since"] = None

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
                            yaw,
                            pitch
                        )

                    else:
                        cache["yaw"] = None
                        cache["pitch"] = None
                        cache["roll"] = None
                        cache["head_status"] = "POSE UNKNOWN"

                else:

                    # Xoa ket qua cu de tranh hien stale yaw/pitch/ear
                    cache["ear"] = None
                    cache["eye_status"] = eye_status
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

                        if (
                            cache["ever_seen_face"]
                            and
                            missing_duration
                            >= FACE_MISSING_TURNED_SECONDS
                        ):
                            cache["head_status"] = (
                                "QUAY KHUAT / KHONG NHIN PHIA TRUOC"
                            )
                        else:
                            cache["head_status"] = "DANG TIM MAT"


            # =================================================
            # VI TRI TU DONG - theo đúng Track ID
            # =================================================
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

            # =================================================
            # V13: FER + DAiSEE ENGAGEMENT CHAY NEN
            # =================================================
            if ai_analyze_id == track_id:
                # ROI qua nho thi bo qua AI de tranh ton CPU ma ket qua kem tin cay.
                if (
                    person_w >= AI_MIN_PERSON_WIDTH
                    and person_h >= AI_MIN_PERSON_HEIGHT
                    and not cache.get("ai_pending", False)
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

            if final_status == "DAU HIEU LO AU":
                anxiety_count += 1
                box_color = (180, 0, 180)
                display_status = f"Dấu hiệu lo âu • {final_reason}"

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

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            person_labels.append(
                (
                    f"ID {track_id} • {display_status}",
                    x1,
                    max(24, y1 - 5),
                    box_color,
                )
            )


            if (
                track_id in analyze_ids
                and
                cache["face_found"]
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
                f"Tư thế: {body_count}  |  "
                f"Đã tự học chỗ: {calibrated_seat_count}"
            ),
            (
                f"Lệch hướng: {head_not_forward_count}  |  "
                f"Nhắm mắt: {eyes_closed_count}  |  "
                f"Rời chỗ: {seat_away_count}  |  "
                f"AI: {ai_ready_count}  |  "
                f"Mất tập trung: {distracted_count}  |  "
                f"Lo âu: {anxiety_count}  |  "
                f"FPS: {fps_smooth:.1f}"
            ),
        ]

        # Han che cac nhan 1 dong bi de len nhau khi nhieu hoc sinh dung gan nhau.
        person_labels.sort(key=lambda item: item[2])
        adjusted_labels = []
        last_y = -999

        for text_label, lx, ly, lcolor in person_labels:
            adjusted_y = ly
            if adjusted_y - last_y < 24:
                adjusted_y = last_y + 24
            adjusted_y = min(adjusted_y, frame.shape[0] - 8)
            adjusted_labels.append((text_label, lx, adjusted_y, lcolor))
            last_y = adjusted_y

        frame = draw_unicode_labels(
            frame,
            adjusted_labels,
            hud_lines
        )


        cv2.imshow(
            "Week 8 - Multi Person FUSION AI V14 + Backend",
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
    face_landmarker.close()
    pose_landmarker.close()
    cv2.destroyAllWindows()


print("")
print("======================================================")
print("KET THUC WEEK 8 - FUSION AI V14 + BACKEND MULTI-TRACK")
print("Backend sent  :", backend_sender.sent_count)
print("Backend failed:", backend_sender.failed_count)
print("Alerts created:", backend_sender.alert_count)
print("======================================================")
