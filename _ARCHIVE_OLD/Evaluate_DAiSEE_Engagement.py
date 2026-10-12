import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

MODEL_PATH = r"D:\Doan\mobilenetv2_daisee_engagement_final.keras"
TEST_DIR = r"D:\Doan\datasets\DAiSEE_Frames\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "Very Low",
    "Low",
    "High",
    "Very High"
]

print("===== LOAD TEST DATA =====")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

print("Class folders:", test_ds.class_names)

print("\n===== LOAD MODEL =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded OK")
print("Input :", model.input_shape)
print("Output:", model.output_shape)

print("\n===== PREDICT TEST SET =====")

y_true_all = []
y_prob_all = []

total_batches = tf.data.experimental.cardinality(test_ds).numpy()

for i, (images, labels) in enumerate(test_ds, start=1):

    probs = model(images, training=False).numpy()

    y_true_all.append(labels.numpy())
    y_prob_all.append(probs)

    if i % 25 == 0 or i == total_batches:
        print(f"Processed {i}/{total_batches} batches")

y_true = np.concatenate(y_true_all)
y_prob = np.concatenate(y_prob_all)

y_pred = np.argmax(y_prob, axis=1)

loss_values = tf.keras.losses.sparse_categorical_crossentropy(
    y_true,
    y_prob
).numpy()

test_loss = float(np.mean(loss_values))
test_acc = accuracy_score(y_true, y_pred)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3]
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n===== TEST RESULT =====")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(report)

result_text = f"""
===== DAISEE ENGAGEMENT EVALUATION =====

Model:
{MODEL_PATH}

Test images:
{len(y_true)}

Class mapping:
0 = Very Low
1 = Low
2 = High
3 = Very High

Test Loss:
{test_loss:.4f}

Test Accuracy:
{test_acc:.4f}

Test Accuracy Percent:
{test_acc * 100:.2f}%

Confusion Matrix:
{cm}

Classification Report:
{report}
"""

Path(
    r"D:\Doan\evaluation_daisee_engagement.txt"
).write_text(
    result_text,
    encoding="utf-8"
)

print("\nSaved:")
print(r"D:\Doan\evaluation_daisee_engagement.txt")
print("\n===== DONE =====")