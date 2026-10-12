import cv2
import time
import math
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

CAMERA_INDEX = 0
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
PERSON_CLASS = 0

EAR_THRESHOLD = 0.21
MAX_FACE_ANALYSES_PER_FRAME = 2
UPPER_BODY_RATIO = 0.50
CACHE_MAX_AGE_FRAMES = 90

MIN_PERSON_WIDTH = 70
MIN_PERSON_HEIGHT = 120

YAW_THRESHOLD = 25.0
PITCH_DOWN_THRESHOLD = 18.0
PITCH_UP_THRESHOLD = -18.0

FACE_MISSING_TURNED_SECONDS = 1.5

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
}


def get_person_display(head_status, eye_status):
    # Mau: xanh = binh thuong, cam = lech huong, do = nham mat,
    # vang = dang tim mat, xam = mat qua nho/qua xa.
    # Nhắm mắt ưu tiên hiển thị đỏ khi đầu vẫn nhìn gần phía trước.
    if (
        eye_status == "NHAM MAT"
        and head_status in (
            "NHIN THANG",
            "CHUA PHAN TICH",
            "POSE UNKNOWN",
        )
    ):
        return "Nhắm mắt", (0, 0, 255)

    if head_status in (
        "QUAY TRAI",
        "QUAY PHAI",
        "CUI DAU",
        "NGUOC LEN",
        "QUAY KHUAT / KHONG NHIN PHIA TRUOC",
    ):
        # Cam: tín hiệu lệch hướng / tư thế đầu
        return DISPLAY_TEXT.get(head_status, head_status), (0, 165, 255)

    if head_status == "NHIN THANG":
        # Xanh lá: bình thường
        return "Nhìn thẳng", (0, 200, 0)

    if head_status == "DANG TIM MAT":
        # Vàng: chưa đủ dữ liệu
        return "Đang tìm mặt", (0, 215, 255)

    if head_status == "QUA XA / MAT QUA NHO":
        # Xám: không đủ chất lượng ảnh
        return "Mặt quá nhỏ / quá xa", (150, 150, 150)

    return DISPLAY_TEXT.get(head_status, head_status), (160, 160, 160)


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



if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {YOLO_MODEL_PATH}"
    )

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {FACE_MODEL_PATH}"
    )

device = 0 if torch.cuda.is_available() else "cpu"

print("======================================================")
print("WEEK 8 - MULTI PERSON + EAR + HEAD POSE V5")
print("======================================================")
print("YOLO model :", YOLO_MODEL_PATH)
print("Face model :", FACE_MODEL_PATH)
print("Device     :", device)
print("MediaPipe  :", getattr(mp, "__version__", "unknown"))
print("EAR threshold:", EAR_THRESHOLD)
print("Yaw threshold:", YAW_THRESHOLD)
print("Quay khuat sau:", FACE_MISSING_TURNED_SECONDS, "s")
print("Nhan Q de thoat.")
print("======================================================")

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

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    face_landmarker.close()
    raise RuntimeError(
        f"Khong mo duoc camera index {CAMERA_INDEX}"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

track_cache = {}
frame_index = 0
round_robin_start = 0

fps_smooth = 0.0
prev_time = time.time()

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Khong doc duoc frame camera.")
            break

        frame_index += 1
        current_time = time.time()

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
                        "missing_since": None
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


        face_count = 0
        head_not_forward_count = 0
        eyes_closed_count = 0

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


            ear = cache["ear"]
            eye_status = cache["eye_status"]
            head_status = cache["head_status"]
            yaw = cache["yaw"]
            pitch = cache["pitch"]

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


            display_status, box_color = get_person_display(
                head_status,
                eye_status
            )


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
                f"Lệch hướng: {head_not_forward_count}"
            ),
            (
                f"Nhắm mắt: {eyes_closed_count}  |  "
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
            "Week 8 - Multi Person EAR + Head Pose V5",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break


finally:

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()


print("")
print("======================================================")
print("KET THUC WEEK 8 - EAR + HEAD POSE V5")
print("======================================================")
