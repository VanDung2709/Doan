import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES_EXPECTED = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def count_images_by_class(train_dir, class_names):
    counts = {}

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)

        count = 0

        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):
                    count += 1

        counts[class_name] = count

    return counts


def build_class_weights(train_dir, class_names):
    """
    Tính class weight để giảm ảnh hưởng mất cân bằng dữ liệu FER-2013.
    """

    counts = count_images_by_class(train_dir, class_names)

    total = sum(counts.values())
    num_classes = len(class_names)

    class_weights = {}

    print("\n===== PHAN BO DU LIEU TRAIN =====")

    for i, class_name in enumerate(class_names):

        count = counts[class_name]

        if count == 0:
            weight = 1.0
        else:
            weight = total / (num_classes * count)

        class_weights[i] = weight

        print(
            f"{i} - {class_name:10s}: "
            f"{count:5d} anh | weight = {weight:.4f}"
        )

    return class_weights


def create_datasets(train_dir, test_dir):

    print("\n===== TAO TRAIN DATASET =====")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True
    )

    print("\n===== TAO VALIDATION DATASET =====")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    print("\n===== TAO TEST DATASET =====")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    print("\n===== CLASS NAMES =====")

    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):

    print("\n===== KHOI TAO MOBILENETV2 =====")

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Giai đoạn đầu: đóng băng backbone
    base_model.trainable = False

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(15 / 360),
            layers.RandomBrightness(
                factor=0.15,
                value_range=(0, 255)
            )
        ],
        name="data_augmentation"
    )

    inputs = keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    x = data_augmentation(inputs)

    # Tương đương MobileNetV2 preprocess_input:
    # pixel từ [0,255] -> [-1,1]
    x = layers.Rescaling(
        1.0 / 127.5,
        offset=-1,
        name="mobilenet_preprocess"
    )(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_output"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="FER2013_MobileNetV2"
    )

    return model, base_model


def compile_model(model, learning_rate):

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def evaluate_model(
    model,
    test_ds,
    class_names
):

    print("\n===== DANH GIA TAP TEST =====")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    y_true = []
    y_pred = []

    print("\nDang du doan tap test...")

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predicted_classes
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = len(class_names)

    cm = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=num_classes
    ).numpy()

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    report_lines = []

    report_lines.append(
        f"Test Loss: {test_loss:.4f}"
    )

    report_lines.append(
        f"Test Accuracy: "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    report_lines.append("")
    report_lines.append(
        "Class\tPrecision\tRecall\tF1-score"
    )

    print(
        "\n===== PRECISION / RECALL / F1 ====="
    )

    for i, class_name in enumerate(class_names):

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

        print(
            f"{class_name:10s} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        report_lines.append(
            f"{class_name}\t"
            f"{precision:.4f}\t"
            f"{recall:.4f}\t"
            f"{f1:.4f}"
        )

    np.savetxt(
        "confusion_matrix_fer2013.txt",
        cm,
        fmt="%d"
    )

    with open(
        "evaluation_fer2013.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    print(
        "\nDa luu:"
        "\n- confusion_matrix_fer2013.txt"
        "\n- evaluation_fer2013.txt"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Thu muc FER2013/train"
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Thu muc FER2013/test"
    )

    parser.add_argument(
        "--head_epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--unfreeze",
        type=int,
        default=30
    )

    args = parser.parse_args()

    print(
        "TensorFlow version:",
        tf.__version__
    )

    train_ds, val_ds, test_ds, class_names = (
        create_datasets(
            args.train,
            args.test
        )
    )

    class_weights = build_class_weights(
        args.train,
        class_names
    )

    model, base_model = build_model(
        len(class_names)
    )

    model.summary()

    # =====================================
    # GIAI DOAN 1: TRAIN CLASSIFICATION HEAD
    # =====================================

    print(
        "\n====================================="
    )
    print(
        "GIAI DOAN 1 - TRAIN CLASSIFICATION HEAD"
    )
    print(
        "====================================="
    )

    compile_model(
        model,
        learning_rate=1e-3
    )

    callbacks_head = [
        keras.callbacks.ModelCheckpoint(
            "mobilenetv2_fer2013_head.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks_head
    )

    # =====================================
    # GIAI DOAN 2: FINE-TUNE MOBILENETV2
    # =====================================

    print(
        "\n====================================="
    )
    print(
        "GIAI DOAN 2 - FINE-TUNE MOBILENETV2"
    )
    print(
        "====================================="
    )

    base_model.trainable = True

    # Chỉ mở một số layer cuối
    for layer in base_model.layers[
        :-args.unfreeze
    ]:
        layer.trainable = False

    # Giữ BatchNormalization đóng băng
    for layer in base_model.layers:
        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

    trainable_count = sum(
        1 for layer in base_model.layers
        if layer.trainable
    )

    print(
        f"So layer backbone duoc fine-tune: "
        f"{trainable_count}"
    )

    compile_model(
        model,
        learning_rate=1e-5
    )

    callbacks_ft = [
        keras.callbacks.ModelCheckpoint(
            "mobilenetv2_fer2013_finetuned.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.ft_epochs,
        class_weight=class_weights,
        callbacks=callbacks_ft
    )

    # Lưu model cuối
    model.save(
        "mobilenetv2_fer2013_final.keras"
    )

    print(
        "\nDa luu model:"
        "\n- mobilenetv2_fer2013_head.keras"
        "\n- mobilenetv2_fer2013_finetuned.keras"
        "\n- mobilenetv2_fer2013_final.keras"
    )

    evaluate_model(
        model,
        test_ds,
        class_names
    )


if __name__ == "__main__":
    main()