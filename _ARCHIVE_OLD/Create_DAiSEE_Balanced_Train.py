from pathlib import Path
import shutil
import random
import tensorflow as tf
import numpy as np

SRC = Path(r"D:\Doan\datasets\DAiSEE_Frames\train")
DST = Path(r"D:\Doan\datasets\DAiSEE_Frames\train_balanced")

TARGET = {
    "0": 1000,
    "1": 2000,
    "2": 13085,
    "3": 12465
}

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomContrast(0.10),
])

print("===== CREATE BALANCED DAISEE TRAIN =====")

if DST.exists():
    print("Xoa thu muc cu:", DST)
    shutil.rmtree(DST)

DST.mkdir(parents=True)

for cls in ["0", "1", "2", "3"]:

    src_cls = SRC / cls
    dst_cls = DST / cls
    dst_cls.mkdir()

    files = sorted(
        list(src_cls.glob("*.jpg")) +
        list(src_cls.glob("*.jpeg")) +
        list(src_cls.glob("*.png"))
    )

    print(f"\nClass {cls}")
    print("Original:", len(files))

    # Copy toàn bộ ảnh gốc
    for f in files:
        shutil.copy2(f, dst_cls / f.name)

    current = len(files)
    target = TARGET[cls]

    # Class 2,3 giữ nguyên
    if current >= target:
        print("Final:", current)
        continue

    need = target - current
    print("Need augmented:", need)

    for i in range(need):

        f = random.choice(files)

        img = tf.io.read_file(str(f))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        img = augmenter(
            tf.expand_dims(img, axis=0),
            training=True
        )[0]

        img = tf.clip_by_value(img, 0.0, 1.0)
        img = tf.image.convert_image_dtype(img, tf.uint8)

        encoded = tf.io.encode_jpeg(img, quality=95)

        out_name = f"{f.stem}_aug_{i+1:04d}.jpg"

        tf.io.write_file(
            str(dst_cls / out_name),
            encoded
        )

    final_count = len(list(dst_cls.glob("*.jpg")))
    print("Final:", final_count)

print("\n===== DONE =====")
print("Output:")
print(DST)