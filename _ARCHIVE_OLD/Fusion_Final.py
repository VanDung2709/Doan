"""
FUSION FINAL - TUAN 6
MediaPipe + EAR + FER-2013 + DAiSEE Engagement + Rule-based Fusion

Muc tieu:
- EAR: tin hieu mat nham/mo
- FER: cam xuc 7 lop tren FACE ROI
- DAiSEE Engagement: 4 muc tren TOAN FRAME (giong pipeline frame DAiSEE)
- Fusion:
    + BINH THUONG
    + DAU HIEU MAT TAP TRUNG
    + DAU HIEU LO AU
- Anxiety fusion = 75% negative emotion + 25% Low Engagement cue

Luu y:
- He thong chi CANH BAO DAU HIEU, khong chan doan tam ly.
- Threshold fusion la heuristic/demo, co the tinh chinh sau khi test thuc te.
- Nhan 'q' de thoat.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
import queue
import requests
from pathlib import Path
from collections import deque


# ============================================================
# 1. MODEL PATH
# ============================================================

FER_MODEL_PATH = r"D:\Doan\mobilenetv2_fer2013_final.keras"

ENGAGEMENT_MODEL_PATH = (
    r"D:\Doan\mobilenetv2_daisee_engagement_official_final.keras"
)


# ============================================================
# 2. LABELS
# ============================================================

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

ENGAGEMENT_LABELS = [
    "Very Low",
    "Low",
    "High",
    "Very High",
]


# ============================================================
# 3. THRESHOLDS
# ============================================================

# EAR da dung tu truoc
EAR_THRESHOLD = 0.21

# Threshold Low Engagement da chon tren Validation DAiSEE
LOW_ENGAGEMENT_THRESHOLD = 0.27

# Fusion lo au:
# negative_emotion = P(angry) + P(fear) + P(sad)
# engagement_factor = min(LowScore / 0.27, 1.0)
# anxiety_fusion = 0.75 * negative_emotion + 0.25 * engagement_factor
#
# Day chi la heuristic/cue ho tro, KHONG phai chan doan.
ANXIETY_FUSION_THRESHOLD = 0.45
ANXIETY_EMOTION_WEIGHT = 0.75
ANXIETY_ENGAGEMENT_WEIGHT = 0.25

# Thoi gian duy tri de tranh canh bao nhay lien tuc
EYE_CLOSED_ALERT_SECONDS = 1.0
LOW_ENGAGEMENT_ALERT_SECONDS = 2.0
ANXIETY_ALERT_SECONDS = 2.0


# ============================================================
# 4. PERFORMANCE
# ============================================================

# FER + Engagement chi chay 1 lan / 3 frame
AI_INTERVAL = 3

# Smoothing tren 5 lan inference gan nhat
AI_SMOOTHING_SIZE = 5

# Tat luoi day dac de tang FPS
DRAW_FACE_CONTOURS = False

# Hien cua so ROI rieng hay khong
SHOW_FACE_ROI = False

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# BACKEND INTEGRATION
# ============================================================
# Chỉ dùng để GỬI kết quả Fusion hiện có sang Backend.
# KHÔNG thay đổi EAR / FER / Engagement / Fusion.

BACKEND_ENABLED = True
BACKEND_URL = "http://localhost:3001"
BACKEND_ENV_PATH = Path(r"D:\Doan\.env")
BACKEND_STUDENT_ID = 1

# Gửi tối đa 1 lần/giây để Backend không ảnh hưởng FPS/Fusion.
BACKEND_SEND_INTERVAL_SECONDS = 1.0
BACKEND_REQUEST_TIMEOUT_SECONDS = 2.0


# ============================================================
# 5. EYE LANDMARKS
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# 6. MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# 7. EAR
# ============================================================

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(
            np.array(
                [
                    lm.x * frame_w,
                    lm.y * frame_h,
                ],
                dtype=np.float32,
            )
        )

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ============================================================
# 8. FACE ROI FOR FER
# ============================================================

def extract_face_roi(
    frame,
    landmarks,
    frame_w,
    frame_h,
    padding=20,
):
    x_coords = [lm.x * frame_w for lm in landmarks]
    y_coords = [lm.y * frame_h for lm in landmarks]

    x1 = max(0, int(min(x_coords)) - padding)
    y1 = max(0, int(min(y_coords)) - padding)
    x2 = min(frame_w, int(max(x_coords)) + padding)
    y2 = min(frame_h, int(max(y_coords)) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return None, None

    face_224 = cv2.resize(
        face_roi,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )

    return face_224, (x1, y1, x2, y2)


# ============================================================
# 9. MODEL INPUT
# ============================================================

def prepare_rgb_224(image_bgr):
    """
    2 model deu co preprocessing/rescaling ben trong.
    Chi BGR -> RGB, resize 224x224, giu pixel 0-255.
    """
    image_224 = cv2.resize(
        image_bgr,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )

    image_rgb = cv2.cvtColor(
        image_224,
        cv2.COLOR_BGR2RGB,
    )

    image_input = image_rgb.astype(
        np.float32,
        copy=False,
    )

    return np.expand_dims(
        image_input,
        axis=0,
    )


def predict_model(model, image_bgr):
    image_input = prepare_rgb_224(
        image_bgr
    )

    return model(
        image_input,
        training=False,
    ).numpy()[0]


# ============================================================
# BACKEND SENDER - CHẠY THREAD RIÊNG ĐỂ KHÔNG LÀM GIẢM FPS
# ============================================================

def load_env_file(path):
    values = {}

    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file .env: {path}"
        )

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
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


class BackendSender:
    """
    Chỉ gửi kết quả hiện tại sang Backend.
    Không can thiệp vào logic Fusion.
    """

    def __init__(self):
        self.enabled = BACKEND_ENABLED
        self.base_url = BACKEND_URL.rstrip("/")
        self.student_id = BACKEND_STUDENT_ID
        self.interval = BACKEND_SEND_INTERVAL_SECONDS
        self.timeout = BACKEND_REQUEST_TIMEOUT_SECONDS

        self.session = requests.Session()
        self.token = None

        self.last_enqueue_time = 0.0
        self.last_status = None

        self.sent_count = 0
        self.failed_count = 0
        self.alert_count = 0

        self.work_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = None

        self.email = None
        self.password = None

    def start(self):
        if not self.enabled:
            print("[BACKEND] Tat gui Backend.")
            return

        try:
            env = load_env_file(
                BACKEND_ENV_PATH
            )

            self.email = env.get(
                "ADMIN_EMAIL"
            )

            self.password = env.get(
                "ADMIN_PASSWORD"
            )

            if not self.email or not self.password:
                raise RuntimeError(
                    "Thieu ADMIN_EMAIL hoac ADMIN_PASSWORD trong .env"
                )

            self.login()

            self.worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
            )

            self.worker.start()

            print(
                "[BACKEND] Ket noi OK:",
                self.base_url,
            )

            print(
                "[BACKEND] Student ID:",
                self.student_id,
            )

        except Exception as error:
            self.enabled = False

            print(
                "[BACKEND] Khong khoi tao duoc:",
                error,
            )

            print(
                "[BACKEND] Fusion van tiep tuc chay binh thuong."
            )

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

        data = response.json()

        token = data.get("token")

        if not token:
            raise RuntimeError(
                "Backend khong tra ve JWT token."
            )

        self.token = token

    def _headers(self):
        return {
            "Authorization":
                f"Bearer {self.token}"
        }

    def maybe_send(self, payload, fusion_status):
        """
        Gui toi da 1 request / giay.

        Quan trong:
        - Backend chi NHAN ket qua Fusion da co.
        - Khong gui them moi khi status dao dong tung frame.
        - Neu queue dang ban, chi giu payload moi nhat.
        - Do tre realtime toi da xap xi 1 giay.
        """

        if not self.enabled:
            return

        now = time.perf_counter()

        interval_reached = (
            now - self.last_enqueue_time
            >= self.interval
        )

        if not interval_reached:
            return

        self.last_enqueue_time = now
        self.last_status = fusion_status

        try:
            self.work_queue.put_nowait(
                payload
            )

        except queue.Full:
            # Bo payload cu, chi giu ket qua moi nhat.
            try:
                self.work_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.work_queue.put_nowait(
                    payload
                )
            except queue.Full:
                pass

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
                alert.get("alert_type"),
                "confidence=",
                alert.get("confidence"),
            )

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.work_queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                continue

            try:
                self._post_once(
                    payload
                )

            except Exception as error:
                self.failed_count += 1

                print(
                    "[BACKEND] Gui that bai:",
                    error,
                )

    def close(self):
        if not self.enabled:
            return

        self.stop_event.set()

        if (
            self.worker is not None
            and self.worker.is_alive()
        ):
            self.worker.join(
                timeout=1.0
            )

        self.session.close()


# ============================================================
# 10. TIMER HELPER
# ============================================================

def update_timer(active, start_time, now):
    if active:
        if start_time is None:
            start_time = now

        duration = now - start_time
    else:
        start_time = None
        duration = 0.0

    return start_time, duration


# ============================================================
# 11. FUSION RULE
# ============================================================

def fusion_decision(
    eye_closed_duration,
    low_engagement_duration,
    anxiety_duration,
):
    """
    Rule-based fusion.

    Uu tien:
    1) Dau hieu lo au:
       anxiety fusion score duy tri du lau.
       Score nay da ket hop negative emotion + Low Engagement Score.
    2) Dau hieu mat tap trung:
       engagement thap keo dai HOAC nham mat keo dai
    3) Binh thuong

    Day la heuristic phuc vu prototype/khóa luận,
    khong phai mo hinh chan doan y khoa.
    """

    anxiety_alert = (
        anxiety_duration >= ANXIETY_ALERT_SECONDS
    )

    distraction_alert = (
        low_engagement_duration
        >= LOW_ENGAGEMENT_ALERT_SECONDS
        or eye_closed_duration
        >= EYE_CLOSED_ALERT_SECONDS
    )

    if anxiety_alert:
        return (
            "DAU HIEU LO AU",
            (0, 165, 255),
        )

    if distraction_alert:
        return (
            "DAU HIEU MAT TAP TRUNG",
            (0, 0, 255),
        )

    return (
        "BINH THUONG",
        (0, 255, 0),
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    backend_sender = BackendSender()
    backend_sender.start()


    # --------------------------------------------------------
    # LOAD FER
    # --------------------------------------------------------

    print("===== LOAD FER MODEL =====")

    fer_model = tf.keras.models.load_model(
        FER_MODEL_PATH
    )

    print("FER model loaded OK")
    print("FER Input :", fer_model.input_shape)
    print("FER Output:", fer_model.output_shape)


    # --------------------------------------------------------
    # LOAD ENGAGEMENT
    # --------------------------------------------------------

    print("\n===== LOAD ENGAGEMENT MODEL =====")

    engagement_model = tf.keras.models.load_model(
        ENGAGEMENT_MODEL_PATH
    )

    print("Engagement model loaded OK")
    print(
        "Engagement Input :",
        engagement_model.input_shape,
    )
    print(
        "Engagement Output:",
        engagement_model.output_shape,
    )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return


    # --------------------------------------------------------
    # BUFFERS
    # --------------------------------------------------------

    fps_buffer = deque(maxlen=30)
    fps_log = []

    emotion_buffer = deque(
        maxlen=AI_SMOOTHING_SIZE
    )

    engagement_buffer = deque(
        maxlen=AI_SMOOTHING_SIZE
    )

    prev_time = time.perf_counter()

    frame_index = 0
    ai_inference_count = 0


    # --------------------------------------------------------
    # LAST AI RESULT
    # --------------------------------------------------------

    last_emotion_label = "..."
    last_emotion_confidence = 0.0

    last_engagement_label = "..."
    last_engagement_confidence = 0.0

    last_low_score = 0.0
    last_engagement_state = "..."

    last_negative_emotion_score = 0.0
    last_anxiety_score = 0.0


    # --------------------------------------------------------
    # TIMERS
    # --------------------------------------------------------

    eye_closed_start = None
    low_engagement_start = None
    anxiety_start = None

    eye_closed_duration = 0.0
    low_engagement_duration = 0.0
    anxiety_duration = 0.0


    # --------------------------------------------------------
    # STATUS COUNTERS
    # --------------------------------------------------------

    status_counts = {
        "BINH THUONG": 0,
        "DAU HIEU MAT TAP TRUNG": 0,
        "DAU HIEU LO AU": 0,
    }


    # --------------------------------------------------------
    # FACE MESH
    # --------------------------------------------------------

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Khong doc duoc frame.")
                break

            frame_index += 1

            frame = cv2.flip(
                frame,
                1,
            )

            h, w, _ = frame.shape

            # Frame sach cho FER + Engagement
            clean_frame = frame.copy()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb_frame.flags.writeable = False

            results = face_mesh.process(
                rgb_frame
            )

            rgb_frame.flags.writeable = True

            now = time.perf_counter()


            # =================================================
            # CO FACE
            # =================================================

            if results.multi_face_landmarks:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = face_landmarks.landmark


                # ---------------------------------------------
                # OPTIONAL FACE CONTOURS
                # ---------------------------------------------

                if DRAW_FACE_CONTOURS:

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=(
                            mp_face_mesh
                            .FACEMESH_CONTOURS
                        ),
                        landmark_drawing_spec=None,
                        connection_drawing_spec=(
                            mp_drawing_styles
                            .get_default_face_mesh_contours_style()
                        ),
                    )


                # ---------------------------------------------
                # EAR - EVERY FRAME
                # ---------------------------------------------

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE,
                    w,
                    h,
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE,
                    w,
                    h,
                )

                avg_ear = (
                    left_ear + right_ear
                ) / 2.0

                eye_closed = (
                    avg_ear < EAR_THRESHOLD
                )

                if eye_closed:
                    eye_status = "NHAM MAT"
                    ear_color = (0, 0, 255)
                else:
                    eye_status = "MO MAT"
                    ear_color = (0, 255, 0)


                # ---------------------------------------------
                # FACE ROI FOR FER
                # ---------------------------------------------

                face_224, face_box = extract_face_roi(
                    clean_frame,
                    landmarks,
                    w,
                    h,
                    padding=20,
                )


                if (
                    face_224 is not None
                    and face_box is not None
                ):

                    x1, y1, x2, y2 = face_box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2,
                    )


                    # =========================================
                    # AI EVERY N FRAMES
                    # =========================================

                    should_run_ai = (
                        frame_index == 1
                        or frame_index
                        % AI_INTERVAL == 0
                    )

                    if should_run_ai:

                        # -------------------------------------
                        # FER: FACE ROI
                        # -------------------------------------

                        emotion_probs = predict_model(
                            fer_model,
                            face_224,
                        )

                        emotion_buffer.append(
                            emotion_probs
                        )

                        smooth_emotion_probs = np.mean(
                            np.asarray(
                                emotion_buffer
                            ),
                            axis=0,
                        )

                        emotion_id = int(
                            np.argmax(
                                smooth_emotion_probs
                            )
                        )

                        last_emotion_label = (
                            EMOTION_LABELS[
                                emotion_id
                            ]
                        )

                        last_emotion_confidence = float(
                            smooth_emotion_probs[
                                emotion_id
                            ]
                        )


                        # -------------------------------------
                        # ENGAGEMENT: TOAN FRAME
                        # -------------------------------------
                        #
                        # DAiSEE pipeline da trich frame video
                        # va resize 224x224, nen realtime
                        # Engagement dung full frame thay vi
                        # face ROI de giam domain mismatch.
                        # -------------------------------------

                        engagement_probs = predict_model(
                            engagement_model,
                            clean_frame,
                        )

                        engagement_buffer.append(
                            engagement_probs
                        )

                        smooth_engagement_probs = np.mean(
                            np.asarray(
                                engagement_buffer
                            ),
                            axis=0,
                        )

                        engagement_id = int(
                            np.argmax(
                                smooth_engagement_probs
                            )
                        )

                        last_engagement_label = (
                            ENGAGEMENT_LABELS[
                                engagement_id
                            ]
                        )

                        last_engagement_confidence = float(
                            smooth_engagement_probs[
                                engagement_id
                            ]
                        )

                        last_low_score = float(
                            smooth_engagement_probs[0]
                            + smooth_engagement_probs[1]
                        )

                        if (
                            last_low_score
                            >= LOW_ENGAGEMENT_THRESHOLD
                        ):
                            last_engagement_state = "LOW"
                        else:
                            last_engagement_state = "HIGH"


                        # -------------------------------------
                        # ANXIETY FUSION SCORE
                        #
                        # Negative emotion cue:
                        # angry + fear + sad
                        #
                        # Engagement cue:
                        # Low Score duoc chuan hoa theo threshold 0.27.
                        #
                        # Weighted fusion:
                        # 75% negative emotion + 25% engagement.
                        # -------------------------------------

                        angry_id = EMOTION_LABELS.index(
                            "angry"
                        )

                        fear_id = EMOTION_LABELS.index(
                            "fear"
                        )

                        sad_id = EMOTION_LABELS.index(
                            "sad"
                        )

                        last_negative_emotion_score = float(
                            smooth_emotion_probs[
                                angry_id
                            ]
                            + smooth_emotion_probs[
                                fear_id
                            ]
                            + smooth_emotion_probs[
                                sad_id
                            ]
                        )

                        engagement_factor = min(
                            last_low_score
                            / LOW_ENGAGEMENT_THRESHOLD,
                            1.0,
                        )

                        last_anxiety_score = float(
                            ANXIETY_EMOTION_WEIGHT
                            * last_negative_emotion_score
                            + ANXIETY_ENGAGEMENT_WEIGHT
                            * engagement_factor
                        )

                        ai_inference_count += 1


                # ---------------------------------------------
                # TIMER STATES
                # ---------------------------------------------

                low_engagement_active = (
                    last_engagement_state == "LOW"
                )

                anxiety_active = (
                    last_anxiety_score
                    >= ANXIETY_FUSION_THRESHOLD
                )

                (
                    eye_closed_start,
                    eye_closed_duration,
                ) = update_timer(
                    eye_closed,
                    eye_closed_start,
                    now,
                )

                (
                    low_engagement_start,
                    low_engagement_duration,
                ) = update_timer(
                    low_engagement_active,
                    low_engagement_start,
                    now,
                )

                (
                    anxiety_start,
                    anxiety_duration,
                ) = update_timer(
                    anxiety_active,
                    anxiety_start,
                    now,
                )


                # ---------------------------------------------
                # FUSION
                # ---------------------------------------------

                fusion_status, fusion_color = (
                    fusion_decision(
                        eye_closed_duration,
                        low_engagement_duration,
                        anxiety_duration,
                    )
                )

                status_counts[
                    fusion_status
                ] += 1


                # ---------------------------------------------
                # GUI KET QUA HIEN TAI SANG BACKEND
                # ---------------------------------------------
                # Chỉ gửi dữ liệu đã có sẵn.
                # Không sửa bất kỳ quyết định Fusion nào.

                backend_payload = {
                    "studentId": BACKEND_STUDENT_ID,

                    "ear": round(
                        float(avg_ear),
                        4,
                    ),

                    "eyeStatus": eye_status.replace(
                        " ",
                        "_",
                    ),

                    "emotion": last_emotion_label,

                    "emotionConfidence": round(
                        float(last_emotion_confidence),
                        4,
                    ),

                    "engagement": last_engagement_label,

                    "engagementConfidence": round(
                        float(last_engagement_confidence),
                        4,
                    ),

                    "lowScore": round(
                        float(last_low_score),
                        4,
                    ),

                    "anxietyScore": round(
                        float(last_anxiety_score),
                        4,
                    ),

                    "fusionStatus": fusion_status,
                }

                backend_sender.maybe_send(
                    backend_payload,
                    fusion_status,
                )


                # ---------------------------------------------
                # DISPLAY
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.3f} ({eye_status})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    ear_color,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Emotion: "
                        f"{last_emotion_label} "
                        f"{last_emotion_confidence:.2f}"
                    ),
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Engagement: "
                        f"{last_engagement_label} "
                        f"{last_engagement_confidence:.2f}"
                    ),
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 0),
                    2,
                )

                if last_engagement_state == "LOW":
                    low_color = (0, 0, 255)
                else:
                    low_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    (
                        f"Low score: "
                        f"{last_low_score:.2f} "
                        f"-> {last_engagement_state}"
                    ),
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    low_color,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Neg emotion: "
                        f"{last_negative_emotion_score:.2f}"
                    ),
                    (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Anxiety fusion: "
                        f"{last_anxiety_score:.2f}"
                    ),
                    (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Fusion: "
                        f"{fusion_status}"
                    ),
                    (20, 265),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    fusion_color,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Low {low_engagement_duration:.1f}s | "
                        f"Eyes {eye_closed_duration:.1f}s | "
                        f"Anx {anxiety_duration:.1f}s"
                    ),
                    (20, 295),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )

                cv2.putText(
                    frame,
                    f"AI: 1/{AI_INTERVAL} frame",
                    (20, 322),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )

                if SHOW_FACE_ROI and face_224 is not None:
                    cv2.imshow(
                        "Face ROI 224x224",
                        face_224,
                    )


            # =================================================
            # NO FACE
            # =================================================

            else:

                emotion_buffer.clear()
                engagement_buffer.clear()

                last_emotion_label = "..."
                last_emotion_confidence = 0.0

                last_engagement_label = "..."
                last_engagement_confidence = 0.0

                last_low_score = 0.0
                last_engagement_state = "..."

                last_negative_emotion_score = 0.0
                last_anxiety_score = 0.0

                eye_closed_start = None
                low_engagement_start = None
                anxiety_start = None

                eye_closed_duration = 0.0
                low_engagement_duration = 0.0
                anxiety_duration = 0.0

                cv2.putText(
                    frame,
                    "NO FACE",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )


            # =================================================
            # FPS
            # =================================================

            curr_time = time.perf_counter()
            delta = curr_time - prev_time

            if delta > 0:
                fps = 1.0 / delta
            else:
                fps = 0.0

            prev_time = curr_time

            fps_buffer.append(
                fps
            )

            avg_fps = (
                sum(fps_buffer)
                / len(fps_buffer)
            )

            fps_log.append(
                avg_fps
            )

            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
            )


            # =================================================
            # SHOW
            # =================================================

            cv2.imshow(
                "FUSION FINAL - EAR + FER + Engagement",
                frame,
            )

            if (
                cv2.waitKey(1)
                & 0xFF
            ) == ord("q"):
                break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()

    backend_sender.close()


    # ========================================================
    # REPORT
    # ========================================================

    if fps_log:

        print(
            "\n===== FUSION FINAL REPORT ====="
        )

        print(
            f"AI_INTERVAL: "
            f"{AI_INTERVAL}"
        )

        print(
            f"FPS trung binh: "
            f"{sum(fps_log)/len(fps_log):.2f}"
        )

        print(
            f"FPS thap nhat: "
            f"{min(fps_log):.2f}"
        )

        print(
            f"FPS cao nhat: "
            f"{max(fps_log):.2f}"
        )

        print(
            f"So frame da xu ly: "
            f"{len(fps_log)}"
        )

        print(
            f"So lan AI inference: "
            f"{ai_inference_count}"
        )

        print(
            "\n===== FUSION STATUS COUNTS ====="
        )

        for status, count in status_counts.items():
            print(
                f"{status}: {count}"
            )

        print(
            "\nLuu y: "
            "DAU HIEU LO AU chi la canh bao ho tro, "
            "khong phai chan doan tam ly."
        )

        print(
            "\n===== BACKEND REPORT ====="
        )

        print(
            "Da gui Backend:",
            backend_sender.sent_count,
        )

        print(
            "Gui Backend that bai:",
            backend_sender.failed_count,
        )

        print(
            "Alert Backend da tao:",
            backend_sender.alert_count,
        )


if __name__ == "__main__":
    main()
