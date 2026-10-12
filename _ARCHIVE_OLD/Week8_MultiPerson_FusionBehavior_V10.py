import cv2
import time
import math
import urllib.request
import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

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

CAMERA_INDEX = 0
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
PERSON_CLASS = 0

EAR_THRESHOLD = 0.21
MAX_FACE_ANALYSES_PER_FRAME = 2
MAX_POSE_ANALYSES_PER_FRAME = 1
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
BODY_TURN_ALERT_SECONDS = 3.0

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
        return False, None, "NO FACE", None, [], []

    h, w = person_crop.shape[:2]

    if h < MIN_PERSON_HEIGHT or w < MIN_PERSON_WIDTH:
        return False, None, "TOO SMALL", None, [], []

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
        return False, None, "NO FACE", None, [], []

    landmarks = result.face_landmarks[0]

    if len(landmarks) < 468:
        return False, None, "NO FACE", None, [], []

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
        right_points
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
    Fusion hanh vi rieng tung Track ID.
    CHUA gom FER/Engagement o buoc nay.
    """
    eye_status = cache.get("eye_status", "")
    head_status = cache.get("head_status", "")
    body_status = cache.get("body_status", "")
    seat_away = bool(cache.get("seat_away", False))

    eye_active = (eye_status == "NHAM MAT")
    head_active = (head_status in HEAD_AWAY_STATUSES)
    body_turn_active = (body_status in BODY_TURN_STATUSES)

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

    if body_seconds >= BODY_TURN_ALERT_SECONDS:
        reasons.append("Quay người kéo dài")

    # seat_away da co bo dem SEAT_EXIT_SECONDS tu V9
    if seat_away:
        reasons.append("Rời vị trí")

    if reasons:
        fusion_status = "DAU HIEU MAT TAP TRUNG"
        fusion_reason = " + ".join(reasons[:2])
    elif eye_active or head_active or body_turn_active:
        # Tin hieu moi xuat hien, chua du thoi gian canh bao.
        fusion_status = "THEO DOI"
        if eye_active:
            fusion_reason = "Nhắm mắt"
        elif head_active:
            fusion_reason = "Lệch hướng"
        else:
            fusion_reason = "Quay người"
    else:
        fusion_status = "BINH THUONG"
        fusion_reason = ""

    cache["fusion_status"] = fusion_status
    cache["fusion_reason"] = fusion_reason
    cache["fusion_eye_seconds"] = eye_seconds
    cache["fusion_head_seconds"] = head_seconds
    cache["fusion_body_seconds"] = body_seconds

    return fusion_status, fusion_reason


if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {FACE_MODEL_PATH}"
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("======================================================")
print("WEEK 8 - MULTI PERSON FUSION HANH VI V10")
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
print("Fusion hanh vi: EAR + Head Pose + Body Pose + Roi vi tri.")
print("Dung mot minh KHONG bi xem la mat tap trung.")
print("Nhan Q: thoat.")
print("======================================================")

parser = argparse.ArgumentParser(
    description="Week 8 V10 - Multi-person Fusion hanh vi cho webcam/video"
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
                        "fusion_body_turn_duration": 0.0
                    }

                track_cache[track_id]["last_seen"] = frame_index


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

        if visible_ids:
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


        face_count = 0
        head_not_forward_count = 0
        eyes_closed_count = 0
        body_count = 0
        seat_away_count = 0
        calibrated_seat_count = 0
        distracted_count = 0

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
                    right_points
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
            # FUSION HANH VI RIENG CHO TRACK ID NAY
            # =================================================
            fusion_status, fusion_reason = update_behavior_fusion(
                cache,
                current_time
            )

            base_display, base_color = get_person_display(
                head_status,
                eye_status,
                body_status,
                seat_away=seat_away
            )

            if fusion_status == "DAU HIEU MAT TAP TRUNG":
                distracted_count += 1
                box_color = (0, 0, 255)
                if fusion_reason:
                    display_status = f"Mất tập trung • {fusion_reason}"
                else:
                    display_status = "Mất tập trung"
            elif fusion_status == "THEO DOI":
                box_color = (0, 165, 255)
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
                f"Mất tập trung: {distracted_count}  |  "
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
            "Week 8 - Multi Person FUSION HANH VI V10",
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

    cap.release()
    face_landmarker.close()
    pose_landmarker.close()
    cv2.destroyAllWindows()


print("")
print("======================================================")
print("KET THUC WEEK 8 - FUSION HANH VI V10")
print("======================================================")
