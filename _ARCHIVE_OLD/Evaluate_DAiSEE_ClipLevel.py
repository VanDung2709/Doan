from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

MODEL_PATH = Path(
    r"D:\Doan\mobilenetv2_daisee_engagement_official_final.keras"
)

TEST_DIR = Path(
    r"D:\Doan\datasets\DAiSEE_Frames\test"
)

OUTPUT_PATH = Path(
    r"D:\Doan\evaluation_daisee_engagement_cliplevel.txt"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "Very Low",
    "Low",
    "High",
    "Very High"
]


print("===== DAISEE CLIP-LEVEL EVALUATION =====")


# ============================================================
# 1. DOC DANH SACH FRAME
# ============================================================

image_paths = []
frame_labels = []
clip_ids = []

for class_id in range(4):

    class_dir = TEST_DIR / str(class_id)

    files = sorted(
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png"))
    )

    for path in files:

        # VD:
        # 1100131017_frame_1.jpg
        # -> clip_id = 1100131017

        clip_id = path.stem.split("_frame_")[0]

        image_paths.append(str(path))
        frame_labels.append(class_id)

        # Them class_id vao key de tranh trung ID neu co
        clip_ids.append(f"{class_id}_{clip_id}")


print("Tong frame:", len(image_paths))
print("Ky vong   : 8920")


# ============================================================
# 2. KIEM TRA SO CLIP
# ============================================================

unique_clips = sorted(set(clip_ids))

print("Tong clip :", len(unique_clips))
print("Ky vong   : 1784")

clip_count_per_class = []

for class_id in range(4):

    clips = {
        clip_id
        for clip_id, label in zip(clip_ids, frame_labels)
        if label == class_id
    }

    clip_count_per_class.append(len(clips))

    print(
        f"Class {class_id} - {CLASS_NAMES[class_id]:10s}: "
        f"{len(clips)} clips"
    )


# ============================================================
# 3. LOAD IMAGE
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Giu gia tri pixel 0-255.
    # Model da co preprocessing ben trong.
    image = tf.cast(
        image,
        tf.float32
    )

    return image


dataset = tf.data.Dataset.from_tensor_slices(
    image_paths
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(
    BATCH_SIZE
)

dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n===== LOAD MODEL =====")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded OK")
print("Input :", model.input_shape)
print("Output:", model.output_shape)


# ============================================================
# 5. PREDICT TUNG FRAME
# ============================================================

print("\n===== FRAME PREDICTION =====")

frame_probs = model.predict(
    dataset,
    verbose=1
)

print("Prediction shape:", frame_probs.shape)


# ============================================================
# 6. GOP 5 FRAME CUNG CLIP
# ============================================================

clip_probabilities = defaultdict(list)
clip_true_labels = {}

for clip_id, label, probs in zip(
    clip_ids,
    frame_labels,
    frame_probs
):

    clip_probabilities[clip_id].append(
        probs
    )

    clip_true_labels[clip_id] = label


y_true_clip = []
y_pred_clip = []
clip_mean_probs = []

frames_per_clip = []
for clip_id in sorted(clip_probabilities.keys()):

    probs_list = np.array(
        clip_probabilities[clip_id]
    )

    frames_per_clip.append(
        len(probs_list)
    )

    # Trung binh xac suat cua 5 frame
    mean_probs = np.mean(
        probs_list,
        axis=0
    )
    clip_mean_probs.append(mean_probs)

    predicted_class = int(
        np.argmax(mean_probs)
    )

    true_class = int(
        clip_true_labels[clip_id]
    )

    y_true_clip.append(
        true_class
    )

    y_pred_clip.append(
        predicted_class
    )


y_true_clip = np.array(
    y_true_clip
)

y_pred_clip = np.array(
    y_pred_clip
)


# ============================================================
# 7. KIEM TRA SO FRAME MOI CLIP
# ============================================================

print("\n===== FRAME PER CLIP CHECK =====")

unique_frame_counts, frequency = np.unique(
    frames_per_clip,
    return_counts=True
)

for frame_count, number_of_clips in zip(
    unique_frame_counts,
    frequency
):

    print(
        f"{frame_count} frames/clip : "
        f"{number_of_clips} clips"
    )


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_true_clip,
    y_pred_clip
)

cm = confusion_matrix(
    y_true_clip,
    y_pred_clip,
    labels=[0, 1, 2, 3]
)

report = classification_report(
    y_true_clip,
    y_pred_clip,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print("\n===== CLIP-LEVEL RESULT =====")

print(
    f"Total Clips   : {len(y_true_clip)}"
)

print(
    f"Clip Accuracy : {accuracy:.4f}"
)

print(
    f"Clip Accuracy : {accuracy * 100:.2f}%"
)


print("\n===== CLIP CONFUSION MATRIX =====")

print(cm)


print("\n===== CLIP CLASSIFICATION REPORT =====")

print(report)

# ============================================================
# 8B. BINARY ENGAGEMENT EVALUATION
# 0 + 1 = LOW ENGAGEMENT
# 2 + 3 = HIGH ENGAGEMENT
# ============================================================

clip_mean_probs = np.array(clip_mean_probs)

# Ground truth:
# class 0,1 -> 0 = Low Engagement
# class 2,3 -> 1 = High Engagement
y_true_binary = np.where(
    y_true_clip <= 1,
    0,
    1
)

# Probability gom tu model 4 class
low_score = (
    clip_mean_probs[:, 0]
    + clip_mean_probs[:, 1]
)

high_score = (
    clip_mean_probs[:, 2]
    + clip_mean_probs[:, 3]
)

# Threshold tu nhien 0.5
y_pred_binary = np.where(
    low_score >= 0.27,
    0,
    1
)

binary_cm = confusion_matrix(
    y_true_binary,
    y_pred_binary,
    labels=[0, 1]
)

binary_report = classification_report(
    y_true_binary,
    y_pred_binary,
    labels=[0, 1],
    target_names=[
        "Low Engagement",
        "High Engagement"
    ],
    digits=4,
    zero_division=0
)

binary_accuracy = accuracy_score(
    y_true_binary,
    y_pred_binary
)

print("\n===== BINARY ENGAGEMENT RESULT =====")

print(
    f"Binary Accuracy : "
    f"{binary_accuracy * 100:.2f}%"
)

print("\n===== BINARY CONFUSION MATRIX =====")
print(binary_cm)

print("\n===== BINARY CLASSIFICATION REPORT =====")
print(binary_report)


# ============================================================
# 9. SAVE RESULT
# ============================================================

result_text = f"""
===== DAISEE ENGAGEMENT CLIP-LEVEL EVALUATION =====

Model:
{MODEL_PATH}

Test Frames:
{len(image_paths)}

Test Clips:
{len(y_true_clip)}

Clip distribution:

0 - Very Low:
{clip_count_per_class[0]}

1 - Low:
{clip_count_per_class[1]}

2 - High:
{clip_count_per_class[2]}

3 - Very High:
{clip_count_per_class[3]}

Method:

5 frames of one clip
-> MobileNetV2 probabilities
-> mean probability
-> argmax
-> engagement class

Clip Accuracy:
{accuracy:.4f}

Clip Accuracy Percent:
{accuracy * 100:.2f}%

Confusion Matrix:

{cm}

Classification Report:

{report}
"""

OUTPUT_PATH.write_text(
    result_text,
    encoding="utf-8"
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\n===== DONE =====")