"""
Viec 3 - Tuan 2: Thu nghiem suy luan mo hinh (Inference)
- Tai trong so pre-trained MobileNetV2
- Chay thu du doan tren vai anh lay tu FER-2013
- Muc dich: KIEM TRA DONG DU LIEU (data flow), CHUA phai phan loai cam xuc chinh xac

Luu y quan trong:
MobileNetV2 pretrained goc (ImageNet) duoc train de nhan dien 1000 loai
vat the thong thuong (cho, meo, xe hoi...), KHONG phai de nhan dien cam xuc.
Vi vay ket qua du doan o day se la ten vat the ImageNet (vi du: "sunglasses",
"mask"...) chu KHONG PHAI la ten cam xuc (vui/buon/gian...).

Dieu do la BINH THUONG va DUNG VOI YEU CAU tuan 2: muc tieu chi la kiem tra
xem pipeline "anh -> tien xu ly -> dua vao model -> nhan ket qua" co chay
thong suot hay khong. Viec fine-tune de model thuc su phan loai duoc
cam xuc se lam o tuan 5 (theo dung ke hoach 12 tuan).

Cach chay: python test_inference_mobilenet.py --dir <duong_dan_thu_muc_anh_FER2013>
"""

import os
import cv2
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image as keras_image


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Doc anh, resize ve 224x224, chuan hoa theo chuan MobileNetV2.
    FER-2013 anh goc thuong la grayscale 48x48 -> can convert sang RGB truoc.
    """
    img = keras_image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    """
    Tạo heatmap Grad-CAM cho class có xác suất cao nhất.
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        predicted_class = tf.argmax(predictions[0])

        class_channel = predictions[:, predicted_class]

    # Gradient của class dự đoán theo feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Trung bình gradient theo chiều không gian
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Nhân feature map với trọng số gradient
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Chuẩn hóa 0-1
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def save_gradcam(original_image_path, heatmap, output_path):
    """
    Ghép heatmap Grad-CAM lên ảnh gốc.
    """

    img = cv2.imread(original_image_path)

    if img is None:
        print("Khong doc duoc anh de tao Grad-CAM.")
        return

    # Heatmap 0-255
    heatmap = np.uint8(255 * heatmap)

    # Tạo màu heatmap
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Resize heatmap theo ảnh gốc
    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # Ghép ảnh gốc + heatmap
    result = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(output_path, result)

    print(f"  Grad-CAM da luu: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Thu muc chua vai anh mau tu FER-2013 (xin Phuc vai chuc anh)"
    )
    parser.add_argument(
        "--max_images", type=int, default=5,
        help="So luong anh toi da se test (mac dinh 5)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Khong tim thay thu muc: {args.dir}")
        sys.exit(1)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in sorted(os.listdir(args.dir))
        if f.lower().endswith(valid_ext)
    ][: args.max_images]

    if not image_files:
        print(f"Khong tim thay anh nao trong thu muc: {args.dir}")
        sys.exit(1)

    print(f"Tim thay {len(image_files)} anh, bat dau test...\n")

    # ---- Buoc 1: Tai model pretrained ----
    print("Dang tai MobileNetV2 (pretrained ImageNet)...")
    model = MobileNetV2(weights="imagenet")
    print("Tai model thanh cong.\n")
    print(f"Kich thuoc model: {model.count_params():,} tham so")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}  (1000 lop ImageNet)\n")

    # ---- Buoc 2: Chay inference tung anh ----
    print("=" * 60)
    for fname in image_files:
        fpath = os.path.join(args.dir, fname)
        try:
            img_array = load_and_preprocess_image(fpath)
        except Exception as e:
            print(f"[LOI] Khong doc duoc anh {fname}: {e}")
            continue

        preds = model.predict(img_array, verbose=0)
        top3 = decode_predictions(preds, top=3)[0]
        # ============ GRAD-CAM ============
        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer_name="Conv_1"
        )

        gradcam_name = f"gradcam_{fname}"

        gradcam_path = os.path.join(
            args.dir,
            gradcam_name
        )

        save_gradcam(
            fpath,
            heatmap,
            gradcam_path
        )

        print(f"Anh: {fname}")
        print(f"  Input shape sau tien xu ly: {img_array.shape}")
        print(f"  Output shape cua model:     {preds.shape}")
        print("  Top-3 du doan (ten lop ImageNet, khong phai cam xuc):")
        for i, (class_id, class_name, prob) in enumerate(top3, start=1):
            print(f"    {i}. {class_name:<20s}  {prob * 100:.2f}%")
        print("-" * 60)

    print("\n===== KET LUAN =====")
    print("Pipeline 'anh -> tien xu ly -> model -> ket qua' chay thong suot.")
    print("Ket qua du doan hien la nhan ImageNet (vi model chua fine-tune).")
    print("Buoc tiep theo (tuan 5): fine-tune model tren FER-2013 de model")
    print("thuc su phan loai duoc 7 loai cam xuc co ban.")


if __name__ == "__main__":
    main()