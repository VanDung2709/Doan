""

import os
import sys
import argparse
import random
import numpy as np
import cv2
import tensorflow as tf

# ============ CAU HINH ============
TARGET_SIZE = (224, 224)

# Chuan ImageNet (scale 0-1, kieu PyTorch/torchvision)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_ROTATION_DEG = 15        # xoay toi da +-15 do
BRIGHTNESS_RANGE = (0.7, 1.3)  # he so nhan do sang, 1.0 = giu nguyen


# ============ BUOC 1: RESIZE ============
def resize_image(img, size=TARGET_SIZE):
    """Resize anh ve kich thuoc chuan (mac dinh 224x224)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ============ BUOC 2: CHUAN HOA THEO IMAGENET MEAN/STD ============
def normalize_imagenet(img):
    """
    Chuan hoa anh theo dung preprocess_input cua
    TensorFlow/Keras MobileNetV2.

    Input:
        RGB uint8, gia tri 0-255

    Output:
        float32, gia tri trong khoang gan [-1, 1]
    """

    img = img.astype(np.float32)

    img_norm = tf.keras.applications.mobilenet_v2.preprocess_input(
        img
    )

    return img_norm


# ============ BUOC 3: CAC HAM AUGMENTATION ============
def random_flip(img, p=0.5):
    """
    Lat ngang ngau nhien (KHONG lat doc - lat doc lam mat vo nghia doi voi khuon mat).
    """
    if random.random() < p:
        return cv2.flip(img, 1)  # 1 = lat ngang
    return img


def random_rotation(img, max_deg=MAX_ROTATION_DEG):
    """Xoay anh mot goc ngau nhien trong khoang [-max_deg, +max_deg]."""
    angle = random.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, rot_matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101  # tranh vien den khi xoay
    )
    return rotated


def random_brightness(img, brightness_range=BRIGHTNESS_RANGE):
    """Dieu chinh do sang ngau nhien (nhan pixel voi 1 he so)."""
    factor = random.uniform(*brightness_range)
    img_float = img.astype(np.float32) * factor
    img_clipped = np.clip(img_float, 0, 255).astype(np.uint8)
    return img_clipped


def augment_image(img):
    """Ap dung toan bo augmentation theo thu tu: flip -> rotation -> brightness."""
    img = random_flip(img)
    img = random_rotation(img)
    img = random_brightness(img)
    return img


# ============ PIPELINE TONG HOP ============
def preprocess_for_training(img_path, apply_augmentation=True):
    """
    Pipeline day du cho 1 anh:
    doc anh -> (augment neu can) -> resize -> chuan hoa ImageNet
    Tra ve tensor float32 san sang dua vao model.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Khong doc duoc anh: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if apply_augmentation:
        img_rgb = augment_image(img_rgb)

    img_resized = resize_image(img_rgb)
    img_normalized = normalize_imagenet(img_resized)

    return img_normalized


# ============ DEMO / KIEM TRA PIPELINE ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Thu muc chua anh dataset mau")
    parser.add_argument("--out", type=str, default="augmented_samples", help="Thu muc luu anh minh hoa sau augmentation")
    parser.add_argument("--max_images", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Khong tim thay thu muc: {args.dir}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in sorted(os.listdir(args.dir)) if f.lower().endswith(valid_ext)][: args.max_images]

    if not image_files:
        print(f"Khong tim thay anh nao trong: {args.dir}")
        sys.exit(1)

    print(f"Tim thay {len(image_files)} anh, dang chay pipeline...\n")

    for fname in image_files:
        fpath = os.path.join(args.dir, fname)

        # 1) Chay pipeline day du (co augmentation + chuan hoa)
        try:
            tensor = preprocess_for_training(fpath, apply_augmentation=True)
        except Exception as e:
            print(f"[LOI] {fname}: {e}")
            continue

        print(f"Anh: {fname}")
        print(f"  Shape sau pipeline: {tensor.shape}")
        print(f"  Gia tri min/max sau chuan hoa: {tensor.min():.3f} / {tensor.max():.3f}")

        # 2) Luu them 1 ban anh da augment (chua chuan hoa) de xem bang mat thuong
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_aug_visual = augment_image(img_rgb)
        img_aug_visual = resize_image(img_aug_visual)

        out_path = os.path.join(args.out, f"aug_{fname}")
        cv2.imwrite(out_path, cv2.cvtColor(img_aug_visual, cv2.COLOR_RGB2BGR))
        print(f"  Da luu anh minh hoa: {out_path}\n")

    print("=" * 50)
    print(f"Hoan tat. Xem thu muc '{args.out}' de kiem tra anh sau augmentation.")
    print("Dung cac anh nay de chup man hinh dua vao bao cao (chung minh pipeline hoat dong).")


if __name__ == "__main__":
    main()