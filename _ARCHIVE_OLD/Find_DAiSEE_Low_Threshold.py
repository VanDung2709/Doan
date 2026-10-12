from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

MODEL_PATH = Path(
    r"D:\Doan\mobilenetv2_daisee_engagement_official_final.keras"
)

VAL_DIR = Path(
    r"D:\Doan\datasets\DAiSEE_Frames\validation"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("===== FIND LOW-ENGAGEMENT THRESHOLD ON VALIDATION =====")


# ============================================================
# 1. LOAD FRAME LIST
# ============================================================

image_paths = []
frame_labels = []
clip_ids = []

for class_id in range(4):

    class_dir = VAL_DIR / str(class_id)

    files = sorted(
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png"))
    )

    for path in files:

        clip_id = path.stem.split("_frame_")[0]

        image_paths.append(str(path))
        frame_labels.append(class_id)

        clip_ids.append(
            f"{class_id}_{clip_id}"
        )


print("Validation frames:", len(image_paths))
print("Expected         : 7145")


# ============================================================
# 2. LOAD IMAGES
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
).prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\n===== LOAD MODEL =====")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded OK")


# ============================================================
# 4. PREDICT
# ============================================================

print("\n===== PREDICT VALIDATION =====")

frame_probs = model.predict(
    dataset,
    verbose=1
)


# ============================================================
# 5. GROUP 5 FRAMES PER CLIP
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


y_true = []
low_scores = []

for clip_id in sorted(
    clip_probabilities.keys()
):

    probs_list = np.array(
        clip_probabilities[clip_id]
    )

    mean_probs = np.mean(
        probs_list,
        axis=0
    )

    true_class = clip_true_labels[
        clip_id
    ]

    # Binary ground truth:
    # 0/1 -> Low Engagement = 0
    # 2/3 -> High Engagement = 1
    binary_true = (
        0 if true_class <= 1 else 1
    )

    # Low engagement probability
    low_score = (
        mean_probs[0]
        + mean_probs[1]
    )

    y_true.append(
        binary_true
    )

    low_scores.append(
        low_score
    )


y_true = np.array(y_true)
low_scores = np.array(low_scores)


print("\nValidation clips:", len(y_true))

print(
    "Low clips :",
    np.sum(y_true == 0)
)

print(
    "High clips:",
    np.sum(y_true == 1)
)


# ============================================================
# 6. SWEEP THRESHOLD
# ============================================================

print("\n===== THRESHOLD SEARCH =====")

best_threshold = None
best_macro_f1 = -1

best_result = None


for threshold in np.arange(
    0.05,
    0.951,
    0.01
):

    # low_score >= threshold
    # -> predict Low = 0
    # otherwise High = 1

    y_pred = np.where(
        low_scores >= threshold,
        0,
        1
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    low_precision = precision_score(
        y_true == 0,
        y_pred == 0,
        zero_division=0
    )

    low_recall = recall_score(
        y_true == 0,
        y_pred == 0,
        zero_division=0
    )

    low_f1 = f1_score(
        y_true == 0,
        y_pred == 0,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_threshold = threshold

        best_result = (
            accuracy,
            low_precision,
            low_recall,
            low_f1,
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )
        )


# ============================================================
# 7. RESULT
# ============================================================

accuracy, precision, recall, low_f1, cm = best_result


print("\n===== BEST VALIDATION THRESHOLD =====")

print(
    f"Threshold       : {best_threshold:.2f}"
)

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1        : {best_macro_f1:.4f}"
)

print(
    f"Low Precision   : {precision:.4f}"
)

print(
    f"Low Recall      : {recall:.4f}"
)

print(
    f"Low F1          : {low_f1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\n===== DONE =====")