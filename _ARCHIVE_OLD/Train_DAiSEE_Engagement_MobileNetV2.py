import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

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

def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def evaluate(model, test_ds):
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

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=4).numpy()
    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    lines = [
        f"Test Loss: {loss:.4f}",
        f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)",
        "",
        "Class\tPrecision\tRecall\tF1-score"
    ]

    print("\n===== PRECISION / RECALL / F1 =====")
    for i in range(4):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0

        print(f"{i} - {CLASS_LABELS[i]:10s} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        lines.append(f"{i}-{CLASS_LABELS[i]}\t{p:.4f}\t{r:.4f}\t{f1:.4f}")

    np.savetxt("confusion_matrix_daisee_engagement.txt", cm, fmt="%d")
    Path("evaluation_daisee_engagement.txt").write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--head_epochs", type=int, default=1)
    ap.add_argument("--ft_epochs", type=int, default=1)
    ap.add_argument("--unfreeze", type=int, default=20)
    ap.add_argument("--class_weight_cap", type=float, default=5.0)
    args = ap.parse_args()

    print("TensorFlow:", tf.__version__)

    train_ds = load_ds(args.train, True).prefetch(tf.data.AUTOTUNE)
    val_ds = load_ds(args.val, False).prefetch(tf.data.AUTOTUNE)
    test_ds = load_ds(args.test, False).prefetch(tf.data.AUTOTUNE)

    weights = make_class_weights(args.train, args.class_weight_cap)
    model, base = build_model()
    model.summary()

    print("\n===== GIAI DOAN 1 - TRAIN HEAD =====")
    compile_model(model, 1e-3)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "mobilenetv2_daisee_engagement_head.keras",
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

    compile_model(model, 1e-5)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.ft_epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "mobilenetv2_daisee_engagement_finetuned.keras",
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

    model.save("mobilenetv2_daisee_engagement_final.keras")
    evaluate(model, test_ds)

if __name__ == "__main__":
    main()
