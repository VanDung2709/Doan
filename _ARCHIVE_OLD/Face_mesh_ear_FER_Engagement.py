"""
TUAN 6 - MEDIAPIPE + EAR + FER + DAISEE ENGAGEMENT

Chuc nang:
- Webcam realtime
- MediaPipe Face Mesh
- Tinh EAR
- Face ROI 224x224
- MobileNetV2 FER-2013: 7 cam xuc
- MobileNetV2 DAiSEE: Engagement 4 muc
- Low Engagement Score = P(Very Low) + P(Low)
- Threshold Low Engagement = 0.27
- Smoothing 5 frame
- FPS rolling average

Nhan 'q' de thoat.
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque


# ============================================================
# CAU HINH
# ============================================================

FER_MODEL_PATH = r"D:\Doan\mobilenetv2_fer2013_final.keras"
ENGAGEMENT_MODEL_PATH = r"D:\Doan\mobilenetv2_daisee_engagement_official_final.keras"

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

EAR_THRESHOLD = 0.21

# Threshold nay da duoc chon tren Validation DAiSEE
LOW_ENGAGEMENT_THRESHOLD = 0.27

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# EAR
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
                ]
            )
        )

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ============================================================
# FACE ROI
# ============================================================

def extract_face_roi(frame, landmarks, frame_w, frame_h, padding=20):
    x_coords = [lm.x * frame_w for lm in landmarks]
    y_coords = [lm.y * frame_h for lm in landmarks]

    x1 = int(min(x_coords)) - padding
    y1 = int(min(y_coords)) - padding
    x2 = int(max(x_coords)) + padding
    y2 = int(max(y_coords)) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return None, None

    face_224 = cv2.resize(face_roi, (224, 224))

    return face_224, (x1, y1, x2, y2)


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_face_input(face_bgr):
    """
    Ca 2 model da co preprocessing/rescaling ben trong.
    O day chi doi BGR -> RGB va giu pixel 0-255.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_input = face_rgb.astype(np.float32)
    face_input = np.expand_dims(face_input, axis=0)

    return face_input


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(model, face_bgr):
    face_input = prepare_face_input(face_bgr)

    probabilities = model(
        face_input,
        training=False,
    ).numpy()[0]

    return probabilities


def predict_engagement(model, face_bgr):
    face_input = prepare_face_input(face_bgr)

    probabilities = model(
        face_input,
        training=False,
    ).numpy()[0]

    return probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD FER MODEL
    # --------------------------------------------------------

    print("===== LOAD FER MODEL =====")

    fer_model = tf.keras.models.load_model(
        FER_MODEL_PATH
    )

    print("FER model loaded OK")
    print("FER Input :", fer_model.input_shape)
    print("FER Output:", fer_model.output_shape)


    # --------------------------------------------------------
    # LOAD ENGAGEMENT MODEL
    # --------------------------------------------------------

    print("\n===== LOAD ENGAGEMENT MODEL =====")

    engagement_model = tf.keras.models.load_model(
        ENGAGEMENT_MODEL_PATH
    )

    print("Engagement model loaded OK")
    print("Engagement Input :", engagement_model.input_shape)
    print("Engagement Output:", engagement_model.output_shape)


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return


    # --------------------------------------------------------
    # BUFFER
    # --------------------------------------------------------

    fps_buffer = deque(maxlen=30)
    emotion_buffer = deque(maxlen=5)
    engagement_buffer = deque(maxlen=5)

    prev_time = time.time()
    fps_log = []


    # --------------------------------------------------------
    # FACE MESH
    # --------------------------------------------------------

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Khong doc duoc frame.")
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # Frame sach de crop ROI, khong co landmark ve tren mat
            clean_frame = frame.copy()

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results = face_mesh.process(
                rgb_frame
            )


            # =================================================
            # CO KHUON MAT
            # =================================================

            if results.multi_face_landmarks:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                landmarks = face_landmarks.landmark


                # ---------------------------------------------
                # FACE MESH
                # ---------------------------------------------

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    ),
                )


                # ---------------------------------------------
                # EAR
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
                    0.75,
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


                    # -----------------------------------------
                    # FER
                    # -----------------------------------------

                    emotion_probs = predict_emotion(
                        fer_model,
                        face_224,
                    )

                    emotion_buffer.append(
                        emotion_probs
                    )

                    smooth_emotion_probs = np.mean(
                        np.array(emotion_buffer),
                        axis=0,
                    )

                    emotion_id = int(
                        np.argmax(
                            smooth_emotion_probs
                        )
                    )

                    emotion_label = (
                        EMOTION_LABELS[
                            emotion_id
                        ]
                    )

                    emotion_confidence = float(
                        smooth_emotion_probs[
                            emotion_id
                        ]
                    )


                    # -----------------------------------------
                    # ENGAGEMENT
                    # -----------------------------------------

                    engagement_probs = (
                        predict_engagement(
                            engagement_model,
                            face_224,
                        )
                    )

                    engagement_buffer.append(
                        engagement_probs
                    )

                    smooth_engagement_probs = np.mean(
                        np.array(
                            engagement_buffer
                        ),
                        axis=0,
                    )

                    engagement_id = int(
                        np.argmax(
                            smooth_engagement_probs
                        )
                    )

                    engagement_label = (
                        ENGAGEMENT_LABELS[
                            engagement_id
                        ]
                    )

                    engagement_confidence = float(
                        smooth_engagement_probs[
                            engagement_id
                        ]
                    )


                    # -----------------------------------------
                    # LOW ENGAGEMENT SCORE
                    # -----------------------------------------

                    low_score = float(
                        smooth_engagement_probs[0]
                        + smooth_engagement_probs[1]
                    )

                    if (
                        low_score
                        >= LOW_ENGAGEMENT_THRESHOLD
                    ):
                        engagement_state = "LOW"
                        engagement_color = (
                            0,
                            0,
                            255,
                        )
                    else:
                        engagement_state = "HIGH"
                        engagement_color = (
                            0,
                            255,
                            0,
                        )


                    # -----------------------------------------
                    # DISPLAY
                    # -----------------------------------------

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
                            f"{emotion_label} "
                            f"{emotion_confidence:.2f}"
                        ),
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Engagement: "
                            f"{engagement_label} "
                            f"{engagement_confidence:.2f}"
                        ),
                        (
                            text_x,
                            text_y + 28,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            f"Low score: "
                            f"{low_score:.2f} "
                            f"-> {engagement_state}"
                        ),
                        (
                            text_x,
                            text_y + 56,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        engagement_color,
                        2,
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

            curr_time = time.time()
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
                "MediaPipe + EAR + FER + Engagement",
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
    # FPS REPORT
    # ========================================================

    if fps_log:

        print(
            "\n===== THONG KE FPS ====="
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


if __name__ == "__main__":
    main()
