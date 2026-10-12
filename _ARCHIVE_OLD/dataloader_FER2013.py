import os
import argparse
import tensorflow as tf

# ==================================================
# CAU HINH
# ==================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ==================================================
# DATA AUGMENTATION
# ==================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(15 / 360.0),
    tf.keras.layers.RandomBrightness(0.15),
], name="data_augmentation")


# ==================================================
# PREPROCESS CHO MOBILENETV2
# ==================================================
def preprocess_train(images, labels):
    """
    Train:
    Augmentation -> MobileNetV2 preprocess_input
    """
    images = data_augmentation(images, training=True)

    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


def preprocess_eval(images, labels):
    """
    Validation/Test:
    Khong augmentation.
    Chi preprocess theo MobileNetV2.
    """
    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


# ==================================================
# TAO DATALOADER
# ==================================================
def create_dataloaders(train_dir, test_dir):
    print("\n===== TAO DATALOADER FER-2013 =====")

    # --------------------------------------------------
    # TRAIN
    # 85% cua thu muc train
    # --------------------------------------------------
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

    # Luu ten class truoc khi map()
    class_names = train_ds.class_names

    # --------------------------------------------------
    # VALIDATION
    # 15% cua thu muc train
    # --------------------------------------------------
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

    # --------------------------------------------------
    # TEST
    # Su dung thu muc test rieng cua FER-2013
    # --------------------------------------------------
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    print("\n===== DANH SACH CLASS =====")

    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------
    train_ds = train_ds.map(
        preprocess_train,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess_eval,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_ds = test_ds.map(
        preprocess_eval,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------
    # PREFETCH
    # --------------------------------------------------
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# ==================================================
# KIEM TRA 1 BATCH
# ==================================================
def test_dataloader(dataset, name):
    print(f"\n===== KIEM TRA {name} =====")

    for images, labels in dataset.take(1):

        print("Shape images :", images.shape)
        print("Shape labels :", labels.shape)

        print(
            "Min pixel:",
            float(tf.reduce_min(images))
        )

        print(
            "Max pixel:",
            float(tf.reduce_max(images))
        )

        print(
            "Labels dau batch:",
            labels[:10].numpy()
        )


# ==================================================
# MAIN
# ==================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Duong dan FER2013/train"
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Duong dan FER2013/test"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.train):
        print("Khong tim thay train:", args.train)
        return

    if not os.path.isdir(args.test):
        print("Khong tim thay test:", args.test)
        return

    train_ds, val_ds, test_ds, class_names = create_dataloaders(
        args.train,
        args.test
    )

    test_dataloader(train_ds, "TRAIN")
    test_dataloader(val_ds, "VALIDATION")
    test_dataloader(test_ds, "TEST")

    print("\n=====================================")
    print("DATALOADER HOAT DONG THANH CONG")
    print("=====================================")

    print("Image size :", IMG_SIZE)
    print("Batch size :", BATCH_SIZE)
    print("So class   :", len(class_names))

    print("\nTrain:")
    print("- Shuffle: Co")
    print("- Augmentation: Co")
    print("- MobileNetV2 preprocessing: Co")

    print("\nValidation:")
    print("- Shuffle: Khong")
    print("- Augmentation: Khong")
    print("- MobileNetV2 preprocessing: Co")

    print("\nTest:")
    print("- Shuffle: Khong")
    print("- Augmentation: Khong")
    print("- MobileNetV2 preprocessing: Co")


if __name__ == "__main__":
    main()