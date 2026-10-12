import cv2
import time
import torch
from ultralytics import YOLO

# ============================================================
# TUAN 8 - BUOC 1
# MULTI-PERSON DETECTION + BYTETRACK
# ============================================================

MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 0

CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

PERSON_CLASS = 0  # COCO class 0 = person

# ============================================================
# LOAD MODEL
# ============================================================

device = 0 if torch.cuda.is_available() else "cpu"

print("==============================================")
print("WEEK 8 - MULTI PERSON TRACKING")
print("==============================================")
print(f"Device: {device}")
print(f"Model : {MODEL_PATH}")
print("Nhan Q de thoat.")
print("==============================================")

model = YOLO(MODEL_PATH)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Khong mo duoc camera index {CAMERA_INDEX}"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()
seen_ids = set()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ok, frame = cap.read()

    if not ok:
        print("Khong doc duoc frame camera.")
        break

    results = model.track(
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
    visible_count = 0

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        ids = None

        if result.boxes.id is not None:
            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

        visible_count = len(boxes)

        for index, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box)
            confidence = float(confidences[index])

            if ids is not None:
                track_id = int(ids[index])
                seen_ids.add(track_id)

                label = (
                    f"ID {track_id} | "
                    f"{confidence:.2f}"
                )
            else:
                label = (
                    f"PERSON | "
                    f"{confidence:.2f}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # FPS
    # ========================================================

    now = time.time()
    delta = max(now - prev_time, 1e-6)
    fps = 1.0 / delta
    prev_time = now

    # ========================================================
    # HUD
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (400, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"People visible: {visible_count}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Unique Track IDs: {len(seen_ids)}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(
        "Week 8 - Multi Person Tracking",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("")
print("==============================================")
print("KET THUC MULTI-PERSON TRACKING")
print("Tong Track ID da thay:", len(seen_ids))
print("==============================================")
