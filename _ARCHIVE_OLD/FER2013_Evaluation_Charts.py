import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ==============================
# DOC CONFUSION MATRIX
# ==============================

cm = np.loadtxt(
    "confusion_matrix_fer2013.txt",
    dtype=int
)


# ==============================
# CONFUSION MATRIX - SO LUONG
# ==============================

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("Confusion Matrix - FER-2013")
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "fer2013_confusion_matrix.png",
    dpi=200
)

plt.close()


# ==============================
# TINH PRECISION / RECALL / F1
# ==============================

precisions = []
recalls = []
f1_scores = []

for i in range(len(CLASS_NAMES)):

    tp = cm[i, i]

    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


# ==============================
# BIEU DO PRECISION / RECALL / F1
# ==============================

x = np.arange(len(CLASS_NAMES))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    x - width,
    precisions,
    width,
    label="Precision"
)

plt.bar(
    x,
    recalls,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1_scores,
    width,
    label="F1-score"
)

plt.xticks(
    x,
    CLASS_NAMES
)

plt.ylim(0, 1)

plt.xlabel("Lop cam xuc")
plt.ylabel("Gia tri")
plt.title(
    "Precision, Recall va F1-score tren FER-2013"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "fer2013_precision_recall_f1.png",
    dpi=200
)

plt.close()


print("===== TAO BIEU DO DANH GIA THANH CONG =====")
print("- fer2013_confusion_matrix.png")
print("- fer2013_precision_recall_f1.png")