import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_LABELS = ["Very Low", "Low", "High", "Very High"]

def load_ds(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=42 if shuffle else None
    )

def count_images(train_dir):
    counts = []
    for i in range(4):
        folder = os.path.join(train_dir, str(i))
        counts.append(sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ))
    return counts

def make_class_weights(train_dir, cap=5.0):
    counts = count_images(train_dir)
    total = sum(counts)
    weights = {}
    print("\n===== CLASS WEIGHT DAISEE =====")
    for i, count in enumerate(counts):
        raw = total / (4 * count) if count > 0 else 1.0
        used = min(raw, cap)
        weights[i] = used
        print(f"{i} - {CLASS_LABELS[i]:10s} | {count:6d} frames | raw={raw:.4f} | used={used:.4f}")
    return weights

def build_model():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomBrightness(0.10, value_range=(0, 255)),
    ])

    inp = keras.Input((224, 224, 3))
    x = aug(inp)
    x = layers.Rescaling(1/127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(4, activation="softmax")(x)

    return keras.Model(inp, out), base

def sparse_focal_loss(y_true, y_pred, gamma=2.0):
    """
    Sparse categorical focal loss.

    Chi thay doi loss trong luc TRAIN de tap trung hon vao
    cac mau kho / lop thieu so. Ham tra ve loss theo tung mau
    de class_weight cua Keras van hoat dong dung.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    row_ids = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
    indices = tf.stack([row_ids, y_true], axis=1)
    p_t = tf.gather_nd(y_pred, indices)

    return -tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)


def compile_model(model, lr, use_focal=True):
    loss_fn = sparse_focal_loss if use_focal else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )

def evaluate(model, test_ds, output_dir):
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss     : {loss:.4f}")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Accuracy : {acc*100:.2f}%")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        pred = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(pred, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=4
    ).numpy()

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    lines = [
        f"Test Loss: {loss:.4f}",
        f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)",
        "",
        "Class\tPrecision\tRecall\tF1-score"
    ]

    class_f1 = []
    supports = []

    print("\n===== PRECISION / RECALL / F1 =====")

    for i in range(4):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        support = int(np.sum(cm[i, :]))

        class_f1.append(f1)
        supports.append(support)

        print(
            f"{i} - {CLASS_LABELS[i]:10s} | "
            f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}"
        )

        lines.append(
            f"{i}-{CLASS_LABELS[i]}\t{p:.4f}\t{r:.4f}\t{f1:.4f}"
        )

    macro_f1 = float(np.mean(class_f1))
    weighted_f1 = float(
        np.average(
            class_f1,
            weights=supports
        )
    )

    print(f"\nMacro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    lines.extend([
        "",
        f"Macro F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
    ])

    np.savetxt(
        output_dir / "confusion_matrix_config3.txt",
        cm,
        fmt="%d"
    )

    (
        output_dir / "evaluation_config3.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--head_epochs", type=int, default=1)
    ap.add_argument("--ft_epochs", type=int, default=1)
    ap.add_argument("--unfreeze", type=int, default=20)
    ap.add_argument("--class_weight_cap", type=float, default=5.0)
    ap.add_argument(
        "--output_dir",
        default=r"D:\Doan\Week7_Config3"
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("Config 3: Focal Loss gamma=2.0")
    print("Output dir:", output_dir)

    train_ds = load_ds(args.train, True).prefetch(tf.data.AUTOTUNE)
    val_ds = load_ds(args.val, False).prefetch(tf.data.AUTOTUNE)
    test_ds = load_ds(args.test, False).prefetch(tf.data.AUTOTUNE)

    weights = make_class_weights(args.train, args.class_weight_cap)
    model, base = build_model()
    model.summary()

    print("\n===== GIAI DOAN 1 - TRAIN HEAD =====")
    compile_model(model, 1e-3, use_focal=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "mobilenetv2_daisee_engagement_config3_head.keras"),
                monitor="val_accuracy",
                save_best_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    print("\n===== GIAI DOAN 2 - FINE-TUNE =====")
    base.trainable = True

    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False

    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, 1e-5, use_focal=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.ft_epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "mobilenetv2_daisee_engagement_config3_finetuned.keras"),
                monitor="val_accuracy",
                save_best_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    # Recompile bang Cross-Entropy TRUOC KHI SAVE.
    # Viec nay KHONG thay doi trong so da hoc bang Focal Loss.
    # Muc dich: model final load duoc binh thuong trong Fusion
    # ma khong can custom_objects.
    compile_model(
        model,
        1e-5,
        use_focal=False
    )

    final_path = (
        output_dir
        / "mobilenetv2_daisee_engagement_config3.keras"
    )

    model.save(
        str(final_path)
    )

    print(
        "\nSaved final Config 3:",
        final_path
    )

    evaluate(
        model,
        test_ds,
        output_dir
    )

if __name__ == "__main__":
    main()
