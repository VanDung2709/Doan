import cv2
import time
import torch
import hashlib
import urllib.request
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# TUAN 8 - BUOC 1
# MULTI-PERSON DETECTION + BYTETRACK
# TU DONG TAI YOLOV8N NEU CHUA CO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"

MODEL_URL = (
    "https://huggingface.co/Ultralytics/YOLOv8/"
    "resolve/main/yolov8n.pt?download=true"
)

# SHA256 chinh thuc cua yolov8n.pt tren kho Ultralytics/YOLOv8
MODEL_SHA256 = (
    "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
)

CAMERA_INDEX = 0

CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 640

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

PERSON_CLASS = 0  # COCO class 0 = person


# ============================================================
# HAM KIEM TRA SHA256
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# TU DONG TAI MODEL
# ============================================================

def download_model_if_needed():

    if MODEL_PATH.exists():

        print(f"[MODEL] Da co san: {MODEL_PATH}")

        current_hash = sha256_file(MODEL_PATH)

        if current_hash == MODEL_SHA256:
            print("[MODEL] SHA256 hop le.")
            return

        print("[MODEL] File ton tai nhung SHA256 khong khop.")
        print("[MODEL] Se tai lai model.")

        try:
            MODEL_PATH.unlink()
        except Exception:
            pass

    print("==============================================")
    print("[MODEL] Chua co yolov8n.pt")
    print("[MODEL] Dang tai tu Hugging Face...")
    print("==============================================")

    request = urllib.request.Request(
        MODEL_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    temp_path = MODEL_PATH.with_suffix(".pt.part")

    try:

        with urllib.request.urlopen(
            request,
            timeout=60
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
                        1024 * 256
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
                            f"\r[MODEL] "
                            f"{downloaded / 1024 / 1024:.2f} MB "
                            f"/ "
                            f"{total / 1024 / 1024:.2f} MB "
                            f"({percent:.1f}%)",
                            end=""
                        )

                    else:

                        print(
                            f"\r[MODEL] "
                            f"Da tai "
                            f"{downloaded / 1024 / 1024:.2f} MB",
                            end=""
                        )

        print()

        downloaded_hash = sha256_file(
            temp_path
        )

        if downloaded_hash != MODEL_SHA256:

            raise RuntimeError(
                "SHA256 cua file tai ve khong dung. "
                "Khong su dung model nay."
            )

        temp_path.replace(
            MODEL_PATH
        )

        print(
            f"[MODEL] Tai thanh cong: "
            f"{MODEL_PATH}"
        )

        print(
            "[MODEL] SHA256 hop le."
        )

    except Exception as e:

        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

        print("")
        print(
            "[LOI] Khong tai duoc yolov8n.pt."
        )

        print(
            "[LOI]",
            e
        )

        print(
            "[GOI Y] Neu Hugging Face cung bi chan, "
            "bao lai cho minh."
        )

        raise SystemExit(1)


# ============================================================
# TAI / KIEM TRA MODEL
# ============================================================

download_model_if_needed()


# ============================================================
# LOAD MODEL
# ============================================================

device = (
    0
    if torch.cuda.is_available()
    else "cpu"
)

print("==============================================")
print("WEEK 8 - MULTI PERSON TRACKING")
print("==============================================")
print(f"Device: {device}")
print(f"Model : {MODEL_PATH}")
print("Nhan Q de thoat.")
print("==============================================")

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

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


prev_time = time.time()

seen_ids = set()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ok, frame = cap.read()

    if not ok:

        print(
            "Khong doc duoc frame camera."
        )

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


    if (
        result.boxes is not None
        and len(result.boxes) > 0
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


        ids = None


        if result.boxes.id is not None:

            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )


        visible_count = len(
            boxes
        )


        for index, box in enumerate(
            boxes
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            confidence = float(
                confidences[index]
            )


            if ids is not None:

                track_id = int(
                    ids[index]
                )

                seen_ids.add(
                    track_id
                )

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
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
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

    delta = max(
        now - prev_time,
        1e-6
    )

    fps = 1.0 / delta

    prev_time = now


    # ========================================================
    # HUD
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (420, 110),
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
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 102),
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


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


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
print(
    "Tong Track ID da thay:",
    len(seen_ids)
)
print("==============================================")
