
import re
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path

VAL_DIR = Path(r"D:\Doan\datasets\DAiSEE_Frames\validation")
TEST_DIR = Path(r"D:\Doan\datasets\DAiSEE_Frames\test")

OUTPUT_DIR = Path(r"D:\Doan\Week7_Binary_Comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Baseline": Path(
        r"D:\Doan\Week7_Baseline\mobilenetv2_daisee_engagement_baseline.keras"
    ),
    "Config1_Unfreeze40": Path(
        r"D:\Doan\Week7_Config1\mobilenetv2_daisee_engagement_config1.keras"
    ),
    "Config2_ClassWeight10": Path(
        r"D:\Doan\Week7_Config2\mobilenetv2_daisee_engagement_config2.keras"
    ),
    "Config3_FocalLoss": Path(
        r"D:\Doan\Week7_Config3\mobilenetv2_daisee_engagement_config3.keras"
    ),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 64


def clip_id_from_filename(path: Path):
    stem = path.stem
    return re.sub(r"_frame_\d+$", "", stem)


def load_dataset_files(root: Path):
    items = []

    for class_id in range(4):
        class_dir = root / str(class_id)

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc: {class_dir}"
            )

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in sorted(class_dir.glob(ext)):
                items.append(
                    (
                        p,
                        class_id,
                        clip_id_from_filename(p),
                    )
                )

    if not items:
        raise RuntimeError(
            f"Khong co anh trong: {root}"
        )

    return items


def preprocess_path(path_str):
    image_bytes = tf.io.read_file(path_str)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image


def predict_frames(model, items):
    paths = [
        str(x[0])
        for x in items
    ]

    ds = tf.data.Dataset.from_tensor_slices(
        paths
    )

    ds = ds.map(
        preprocess_path,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(
        BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    )

    probs = model.predict(
        ds,
        verbose=1,
    )

    return np.asarray(
        probs,
        dtype=np.float32,
    )


def aggregate_clip_level(items, probs):
    clips = {}

    for (
        path,
        class_id,
        clip_id,
    ), prob in zip(
        items,
        probs,
    ):

        if clip_id not in clips:
            clips[clip_id] = {
                "label": class_id,
                "probs": [],
            }

        clips[clip_id]["probs"].append(
            prob
        )

    y_true = []
    low_scores = []

    for clip_id in sorted(
        clips.keys()
    ):
        info = clips[clip_id]

        mean_prob = np.mean(
            np.asarray(
                info["probs"],
                dtype=np.float32,
            ),
            axis=0,
        )

        class_id = int(
            info["label"]
        )

        binary_label = (
            0 if class_id in (0, 1)
            else 1
        )

        low_score = float(
            mean_prob[0]
            + mean_prob[1]
        )

        y_true.append(
            binary_label
        )

        low_scores.append(
            low_score
        )

    return (
        np.asarray(
            y_true,
            dtype=np.int32,
        ),
        np.asarray(
            low_scores,
            dtype=np.float32,
        ),
    )


def confusion_binary(
    y_true,
    y_pred,
):
    cm = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    for yt, yp in zip(
        y_true,
        y_pred,
    ):
        cm[yt, yp] += 1

    return cm


def metrics_from_cm(cm):
    metrics = []

    for i in range(2):
        tp = cm[i, i]
        fp = (
            np.sum(cm[:, i])
            - tp
        )
        fn = (
            np.sum(cm[i, :])
            - tp
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.append(
            (
                float(precision),
                float(recall),
                float(f1),
            )
        )

    accuracy = float(
        np.trace(cm)
        / np.sum(cm)
    )

    macro_f1 = float(
        np.mean(
            [
                metrics[0][2],
                metrics[1][2],
            ]
        )
    )

    return (
        accuracy,
        macro_f1,
        metrics,
    )


def predict_binary(
    low_scores,
    threshold,
):
    return np.where(
        low_scores >= threshold,
        0,
        1,
    ).astype(
        np.int32
    )


def find_best_threshold(
    y_true,
    low_scores,
):
    best = None

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):
        threshold = round(
            float(threshold),
            2,
        )

        y_pred = predict_binary(
            low_scores,
            threshold,
        )

        cm = confusion_binary(
            y_true,
            y_pred,
        )

        (
            accuracy,
            macro_f1,
            metrics,
        ) = metrics_from_cm(
            cm
        )

        low_f1 = metrics[0][2]

        candidate = {
            "threshold": threshold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "low_f1": low_f1,
            "cm": cm,
            "metrics": metrics,
        }

        if best is None:
            best = candidate
        elif (
            candidate["macro_f1"]
            > best["macro_f1"] + 1e-12
        ):
            best = candidate
        elif (
            abs(
                candidate["macro_f1"]
                - best["macro_f1"]
            ) <= 1e-12
            and candidate["low_f1"]
            > best["low_f1"]
        ):
            best = candidate

    return best


def print_result(
    title,
    threshold,
    y_true,
    low_scores,
):
    y_pred = predict_binary(
        low_scores,
        threshold,
    )

    cm = confusion_binary(
        y_true,
        y_pred,
    )

    (
        accuracy,
        macro_f1,
        metrics,
    ) = metrics_from_cm(
        cm
    )

    low_p, low_r, low_f1 = metrics[0]
    high_p, high_r, high_f1 = metrics[1]

    print(
        f"\n===== {title} ====="
    )

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f} "
        f"({accuracy*100:.2f}%)"
    )

    print(
        f"Macro F1  : {macro_f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    print(
        f"\nLOW  | Precision: {low_p:.4f} "
        f"| Recall: {low_r:.4f} "
        f"| F1: {low_f1:.4f}"
    )

    print(
        f"HIGH | Precision: {high_p:.4f} "
        f"| Recall: {high_r:.4f} "
        f"| F1: {high_f1:.4f}"
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "low_precision": low_p,
        "low_recall": low_r,
        "low_f1": low_f1,
        "high_precision": high_p,
        "high_recall": high_r,
        "high_f1": high_f1,
    }


def main():
    print(
        "===== WEEK 7 - BINARY LOW/HIGH COMPARISON ====="
    )

    val_items = load_dataset_files(
        VAL_DIR
    )

    test_items = load_dataset_files(
        TEST_DIR
    )

    print(
        f"Validation frames: {len(val_items)}"
    )

    print(
        f"Test frames      : {len(test_items)}"
    )

    rows = []

    for model_name, model_path in MODELS.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            "MODEL:",
            model_name
        )

        print(
            "PATH :",
            model_path
        )

        if not model_path.exists():
            print(
                "SKIP - khong tim thay model."
            )
            continue

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        print(
            "\n--- Predict Validation ---"
        )

        val_probs = predict_frames(
            model,
            val_items,
        )

        (
            val_y,
            val_low_scores,
        ) = aggregate_clip_level(
            val_items,
            val_probs,
        )

        print(
            f"Validation clips: {len(val_y)} "
            f"| LOW={(val_y == 0).sum()} "
            f"| HIGH={(val_y == 1).sum()}"
        )

        best = find_best_threshold(
            val_y,
            val_low_scores,
        )

        print(
            f"\n>>> BEST VALIDATION THRESHOLD: "
            f"{best['threshold']:.2f}"
        )

        val_result = print_result(
            f"{model_name} - VALIDATION",
            best["threshold"],
            val_y,
            val_low_scores,
        )

        print(
            "\n--- Predict Test ---"
        )

        test_probs = predict_frames(
            model,
            test_items,
        )

        (
            test_y,
            test_low_scores,
        ) = aggregate_clip_level(
            test_items,
            test_probs,
        )

        print(
            f"Test clips: {len(test_y)} "
            f"| LOW={(test_y == 0).sum()} "
            f"| HIGH={(test_y == 1).sum()}"
        )

        test_result = print_result(
            f"{model_name} - TEST",
            best["threshold"],
            test_y,
            test_low_scores,
        )

        rows.append({
            "model": model_name,
            "threshold_from_validation":
                best["threshold"],

            "val_accuracy":
                val_result["accuracy"],
            "val_macro_f1":
                val_result["macro_f1"],
            "val_low_precision":
                val_result["low_precision"],
            "val_low_recall":
                val_result["low_recall"],
            "val_low_f1":
                val_result["low_f1"],

            "test_accuracy":
                test_result["accuracy"],
            "test_macro_f1":
                test_result["macro_f1"],
            "test_low_precision":
                test_result["low_precision"],
            "test_low_recall":
                test_result["low_recall"],
            "test_low_f1":
                test_result["low_f1"],
            "test_high_f1":
                test_result["high_f1"],
        })

        del model
        tf.keras.backend.clear_session()

    if not rows:
        print(
            "Khong co model nao de danh gia."
        )
        return

    csv_path = (
        OUTPUT_DIR
        / "binary_comparison_week7.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "===== FINAL COMPARISON ====="
    )

    print(
        f"{'Model':24s} "
        f"{'Thr':>5s} "
        f"{'ValMacro':>9s} "
        f"{'TestMacro':>10s} "
        f"{'TestLowF1':>10s} "
        f"{'TestAcc':>8s}"
    )

    for row in rows:
        print(
            f"{row['model']:24s} "
            f"{row['threshold_from_validation']:5.2f} "
            f"{row['val_macro_f1']:9.4f} "
            f"{row['test_macro_f1']:10.4f} "
            f"{row['test_low_f1']:10.4f} "
            f"{row['test_accuracy']*100:7.2f}%"
        )

    best_model = max(
        rows,
        key=lambda x: (
            x["val_macro_f1"],
            x["val_low_f1"],
        ),
    )

    print(
        "\n===== MODEL CHON THEO VALIDATION ====="
    )

    print(
        "Model:",
        best_model["model"]
    )

    print(
        "Threshold:",
        f"{best_model['threshold_from_validation']:.2f}"
    )

    print(
        "Validation Macro F1:",
        f"{best_model['val_macro_f1']:.4f}"
    )

    print(
        "Test Macro F1:",
        f"{best_model['test_macro_f1']:.4f}"
    )

    print(
        "Test LOW F1:",
        f"{best_model['test_low_f1']:.4f}"
    )

    print(
        "\nDa luu bang so sanh:"
    )

    print(
        csv_path
    )


if __name__ == "__main__":
    main()
