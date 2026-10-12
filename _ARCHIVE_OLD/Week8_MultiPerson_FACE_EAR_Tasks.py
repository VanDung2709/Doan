import cv2
import time
import math
import urllib.request
from pathlib import Path

import torch
from ultralytics import YOLO
import mediapipe as mp

# ============================================================
# TUAN 8 - BUOC 2 (MEDIAPIPE TASKS API)
# MULTI-PERSON TRACKING + FACE LANDMARKER + EAR
# Tuong thich MediaPipe 1.0.x
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = BASE_DIR / "yolov8n.pt"
FACE_MODEL_PATH = BASE_DIR / "face_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

CAMERA_INDEX = 0

CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

PERSON_CLASS = 0

EAR_THRESHOLD = 0.21

# Toi da bao nhieu nguoi duoc chay Face Landmarker moi frame.
# Giu nho de CPU khong bi qua tai khi 5-8 nguoi.
MAX_FACE_ANALYSES_PER_FRAME = 2

# Chi lay phan tren bounding box nguoi de tim khuon mat.
UPPER_BODY_RATIO = 0.48

CACHE_MAX_AGE_FRAMES = 90

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# DOWNLOAD FACE LANDMARKER MODEL
# ============================================================

def download_face_model_if_needed():

    if FACE_MODEL_PATH.exists():
        if FACE_MODEL_PATH.stat().st_size > 1_000_000:
            print(
                "[FACE MODEL] Da co:",
                FACE_MODEL_PATH
            )
            return

        print(
            "[FACE MODEL] File cu khong hop le, tai lai..."
        )

        try:
            FACE_MODEL_PATH.unlink()
        except Exception:
            pass

    print("======================================================")
    print("[FACE MODEL] Dang tai face_landmarker.task")
    print("[FACE MODEL] Nguon: Google MediaPipe")
    print("======================================================")

    request = urllib.request.Request(
        FACE_MODEL_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    temp_path = FACE_MODEL_PATH.with_suffix(
        ".task.part"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            total = response.headers.get(
                "Content-Length"
            )

            total = int(total) if total else None

            downloaded = 0

            with open(
                temp_path,
                "wb"
            ) as f:

                while True:

                    chunk = response.read(
                        256 * 1024
                    )

                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:

                        percent = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r[FACE MODEL] "
                            f"{downloaded / 1024 / 1024:.2f} MB / "
                            f"{total / 1024 / 1024:.2f} MB "
                            f"({percent:.1f}%)",
                            end=""
                        )

                    else:

                        print(
                            f"\r[FACE MODEL] "
                            f"Da tai "
                            f"{downloaded / 1024 / 1024:.2f} MB",
                            end=""
                        )

        print()

        if (
            not temp_path.exists()
            or temp_path.stat().st_size < 1_000_000
        ):
            raise RuntimeError(
                "File face_landmarker.task tai ve qua nho."
            )

        temp_path.replace(
            FACE_MODEL_PATH
        )

        print(
            "[FACE MODEL] Tai thanh cong:",
            FACE_MODEL_PATH
        )

    except Exception as e:

        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        print("")
        print(
            "[LOI] Khong tai duoc face_landmarker.task"
        )
        print(
            "[LOI]",
            e
        )

        raise SystemExit(1)


# ============================================================
# EAR
# ============================================================

def dist(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def eye_aspect_ratio(points):

    a = dist(
        points[1],
        points[5]
    )

    b = dist(
        points[2],
        points[4]
    )

    c = dist(
        points[0],
        points[3]
    )

    if c <= 1e-6:
        return 0.0

    return (
        (a + b)
        / (2.0 * c)
    )


def landmarks_to_points(
    landmarks,
    indices,
    width,
    height
):

    points = []

    for idx in indices:

        lm = landmarks[idx]

        points.append(
            (
                int(lm.x * width),
                int(lm.y * height)
            )
        )

    return points


# ============================================================
# FACE + EAR
# ============================================================

def analyze_face_ear(
    face_landmarker,
    person_crop
):

    if (
        person_crop is None
        or person_crop.size == 0
    ):
        return (
            False,
            None,
            "NO FACE",
            [],
            []
        )

    h, w = person_crop.shape[:2]

    if h < 60 or w < 50:
        return (
            False,
            None,
            "TOO SMALL",
            [],
            []
        )

    upper_h = max(
        1,
        int(h * UPPER_BODY_RATIO)
    )

    upper_crop = person_crop[
        0:upper_h,
        :
    ]

    rgb = cv2.cvtColor(
        upper_crop,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.copy()

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:

        return (
            False,
            None,
            "NO FACE",
            [],
            []
        )

    landmarks = result.face_landmarks[0]

    if len(landmarks) < 468:

        return (
            False,
            None,
            "NO FACE",
            [],
            []
        )

    left_points = landmarks_to_points(
        landmarks,
        LEFT_EYE,
        upper_crop.shape[1],
        upper_crop.shape[0]
    )

    right_points = landmarks_to_points(
        landmarks,
        RIGHT_EYE,
        upper_crop.shape[1],
        upper_crop.shape[0]
    )

    left_ear = eye_aspect_ratio(
        left_points
    )

    right_ear = eye_aspect_ratio(
        right_points
    )

    ear = (
        left_ear
        + right_ear
    ) / 2.0

    eye_status = (
        "NHAM MAT"
        if ear < EAR_THRESHOLD
        else "MO MAT"
    )

    return (
        True,
        ear,
        eye_status,
        left_points,
        right_points
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not YOLO_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Khong tim thay YOLO model: "
        f"{YOLO_MODEL_PATH}"
    )


download_face_model_if_needed()


# ============================================================
# LOAD YOLO
# ============================================================

device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


print("======================================================")
print("WEEK 8 - MULTI PERSON + FACE + EAR (TASKS API)")
print("======================================================")
print("YOLO model :", YOLO_MODEL_PATH)
print("Face model :", FACE_MODEL_PATH)
print("Device     :", device)
print("MediaPipe  :", getattr(mp, "__version__", "unknown"))
print("EAR threshold:", EAR_THRESHOLD)
print(
    "Face analyses/frame:",
    MAX_FACE_ANALYSES_PER_FRAME
)
print("Nhan Q de thoat.")
print("======================================================")


yolo_model = YOLO(
    str(YOLO_MODEL_PATH)
)


# ============================================================
# MEDIAPIPE TASKS FACE LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            FACE_MODEL_PATH
        )
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.40,
    min_face_presence_confidence=0.40,
    min_tracking_confidence=0.40,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)


face_landmarker = (
    FaceLandmarker.create_from_options(
        face_options
    )
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    face_landmarker.close()

    raise RuntimeError(
        f"Khong mo duoc camera index "
        f"{CAMERA_INDEX}"
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# TRACK CACHE
# ============================================================

track_cache = {}

frame_index = 0
round_robin_start = 0

fps_smooth = 0.0
prev_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ok, frame = cap.read()

        if not ok:
            print(
                "Khong doc duoc frame camera."
            )
            break

        frame_index += 1


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

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )


            for i, box in enumerate(
                boxes
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1] - 1,
                    x2
                )

                y2 = min(
                    frame.shape[0] - 1,
                    y2
                )


                track_id = int(
                    ids[i]
                )


                detections.append(
                    {
                        "id": track_id,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "conf": float(
                            confidences[i]
                        )
                    }
                )


                if track_id not in track_cache:

                    track_cache[
                        track_id
                    ] = {
                        "ear": None,
                        "eye_status": "CHUA PHAN TICH",
                        "face_found": False,
                        "last_seen": frame_index,
                        "last_analyzed": -1
                    }


                track_cache[
                    track_id
                ]["last_seen"] = frame_index


        # ====================================================
        # ROUND ROBIN
        # ====================================================

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
                    round_robin_start
                    + k
                ) % len(visible_ids)

                analyze_ids.append(
                    visible_ids[idx]
                )


            round_robin_start = (
                round_robin_start
                + count
            ) % len(visible_ids)


        # ====================================================
        # ANALYZE + DRAW
        # ====================================================

        face_count = 0
        eyes_closed_count = 0


        for detection in detections:

            track_id = detection[
                "id"
            ]

            x1, y1, x2, y2 = detection[
                "box"
            ]


            person_crop = frame[
                y1:y2,
                x1:x2
            ]


            left_points = []
            right_points = []


            if track_id in analyze_ids:

                (
                    face_found,
                    ear,
                    eye_status,
                    left_points,
                    right_points
                ) = analyze_face_ear(
                    face_landmarker,
                    person_crop
                )


                track_cache[
                    track_id
                ]["face_found"] = face_found

                track_cache[
                    track_id
                ]["ear"] = ear

                track_cache[
                    track_id
                ]["eye_status"] = eye_status

                track_cache[
                    track_id
                ]["last_analyzed"] = frame_index


            cache = track_cache[
                track_id
            ]

            ear = cache["ear"]

            eye_status = cache[
                "eye_status"
            ]


            if cache["face_found"]:
                face_count += 1


            if eye_status == "NHAM MAT":
                eyes_closed_count += 1


            if eye_status == "NHAM MAT":

                box_color = (
                    0,
                    0,
                    255
                )

            elif cache["face_found"]:

                box_color = (
                    0,
                    255,
                    0
                )

            else:

                box_color = (
                    0,
                    255,
                    255
                )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            if ear is None:

                label = (
                    f"ID {track_id} | "
                    f"{eye_status}"
                )

            else:

                label = (
                    f"ID {track_id} | "
                    f"EAR {ear:.3f} | "
                    f"{eye_status}"
                )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA
            )


            if (
                track_id in analyze_ids
                and cache["face_found"]
            ):

                for px, py in (
                    left_points
                    + right_points
                ):

                    cv2.circle(
                        frame,
                        (
                            x1 + px,
                            y1 + py
                        ),
                        2,
                        (
                            255,
                            0,
                            255
                        ),
                        -1
                    )


        # ====================================================
        # CLEAR OLD CACHE
        # ====================================================

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


        # ====================================================
        # FPS
        # ====================================================

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


        # ====================================================
        # HUD
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (540, 145),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"People visible: {len(detections)}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Faces with EAR: {face_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Eyes closed: {eyes_closed_count}",
            (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f}",
            (20, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            "Week 8 - Multi Person Face EAR - Tasks API",
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
print("KET THUC WEEK 8 - MULTI PERSON FACE EAR")
print("======================================================")
