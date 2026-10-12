"""
TUAN 6 - BAN TOI UU TOC DO
MEDIAPIPE + EAR + FER + DAISEE ENGAGEMENT

Toi uu:
- MediaPipe + EAR chay moi frame.
- FER + Engagement chi inference moi AI_INTERVAL frame.
- Cac frame con lai dung lai ket qua AI gan nhat.
- Mac dinh KHONG ve toan bo FACEMESH_TESSELATION de tang FPS.
- Camera dat 640x480 va buffer size = 1 neu driver ho tro.
- ROI van resize 224x224 cho 2 MobileNetV2.

Nhan 'q' de thoat.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque


# ============================================================
# 1. CAU HINH MODEL
# ============================================================

FER_MODEL_PATH = r"D:\Doan\mobilenetv2_fer2013_final.keras"

ENGAGEMENT_MODEL_PATH = (
    r"D:\Doan\mobilenetv2_daisee_engagement_official_final.keras"
)

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
# 2. THRESHOLD / TOI UU
# ============================================================

EAR_THRESHOLD = 0.21

# Threshold da chon tren Validation DAiSEE
LOW_ENGAGEMENT_THRESHOLD = 0.27

# AI chi chay 1 lan moi 3 frame.
# Neu van cham co the doi thanh 4 hoac 5.
AI_INTERVAL = 3

# Trung binh 5 LAN INFERENCE AI gan nhat.
AI_SMOOTHING_SIZE = 5

# False = nhanh hon.
# True = ve cac duong vien mat/mat/mieng, khong ve luoi day dac.
DRAW_FACE_CONTOURS = False

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# 3. LANDMARK MAT
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# 4. MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# 5. EAR
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
# 6. FACE ROI
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
# 7. MODEL INPUT
# ============================================================

def prepare_face_input(face_bgr):
    """
    Ca FER va Engagement model deu co preprocessing/rescaling
    ben trong model. Chi doi BGR -> RGB, giu pixel 0-255.
    """
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB,
    )

    face_input = face_rgb.astype(
        np.float32,
        copy=False,
    )

    return np.expand_dims(
        face_input,
        axis=0,
    )


# ============================================================
# 8. PREDICTION
# ============================================================

def predict_model(model, face_bgr):
    face_input = prepare_face_input(
        face_bgr
    )

    return model(
        face_input,
        training=False,
    ).numpy()[0]


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("===== LOAD FER MODEL =====")

    fer_model = tf.keras.models.load_model(
        FER_MODEL_PATH
    )

    print("FER model loaded OK")
    print("FER Input :", fer_model.input_shape)
    print("FER Output:", fer_model.output_shape)


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

    # Khong phai driver nao cung ho tro, nhung neu co se giam tre.
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return


    # --------------------------------------------------------
    # BUFFER
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

    # Ket qua AI gan nhat de dung lai o cac frame bi skip.
    last_emotion_label = "..."
    last_emotion_confidence = 0.0

    last_engagement_label = "..."
    last_engagement_confidence = 0.0

    last_low_score = 0.0
    last_engagement_state = "..."

    # Dem so lan AI thuc su inference.
    ai_inference_count = 0


    # --------------------------------------------------------
    # FACE MESH
    # refine_landmarks=False van co 468 landmark va nhanh hon.
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

            clean_frame = frame.copy()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Bao MediaPipe khong can ghi vao RGB frame.
            rgb_frame.flags.writeable = False

            results = face_mesh.process(
                rgb_frame
            )

            rgb_frame.flags.writeable = True


            # =================================================
            # CO KHUON MAT
            # =================================================

            if results.multi_face_landmarks:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = face_landmarks.landmark


                # ---------------------------------------------
                # CHI VE CONTOURS NEU MUON
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
                # EAR - CHAY MOI FRAME
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

                if avg_ear < EAR_THRESHOLD:
                    eye_status = "NHAM MAT"
                    ear_color = (0, 0, 255)
                else:
                    eye_status = "MO MAT"
                    ear_color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.3f} ({eye_status})",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    ear_color,
                    2,
                )


                # ---------------------------------------------
                # FACE ROI
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
                    # AI CHI CHAY MOI AI_INTERVAL FRAME
                    # =========================================

                    should_run_ai = (
                        frame_index == 1
                        or frame_index % AI_INTERVAL == 0
                    )

                    if should_run_ai:

                        # ---------------- FER -----------------

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


                        # ------------ ENGAGEMENT -------------

                        engagement_probs = predict_model(
                            engagement_model,
                            face_224,
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

                        ai_inference_count += 1


                    # =========================================
                    # DISPLAY LAST AI RESULT
                    # =========================================

                    if last_engagement_state == "LOW":
                        engagement_color = (
                            0,
                            0,
                            255,
                        )
                    else:
                        engagement_color = (
                            0,
                            255,
                            0,
                        )

                    text_x = max(
                        10,
                        x1,
                    )

                    text_y = max(
                        110,
                        y1 - 70,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Emotion: "
                            f"{last_emotion_label} "
                            f"{last_emotion_confidence:.2f}"
                        ),
                        (
                            text_x,
                            text_y,
                        ),
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
                        (
                            text_x,
                            text_y + 27,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Low score: "
                            f"{last_low_score:.2f} "
                            f"-> {last_engagement_state}"
                        ),
                        (
                            text_x,
                            text_y + 54,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        engagement_color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"AI: 1/{AI_INTERVAL} frame"
                        ),
                        (
                            20,
                            112,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )

                    cv2.imshow(
                        "Face ROI 224x224",
                        face_224,
                    )


            # =================================================
            # KHONG CO KHUON MAT
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
                0.8,
                (255, 255, 0),
                2,
            )


            # =================================================
            # SHOW
            # =================================================

            cv2.imshow(
                "Optimized - MediaPipe + EAR + FER + Engagement",
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


    # ========================================================
    # REPORT
    # ========================================================

    if fps_log:

        print(
            "\n===== THONG KE FPS - OPTIMIZED ====="
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


if __name__ == "__main__":
    main()
