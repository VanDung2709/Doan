import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

FER_CLASSES = [
    'angry', 'disgust', 'fear', 'happy',
    'neutral', 'sad', 'surprise'
]

DISPLAY = {
    'angry': 'Tuc gian',
    'disgust': 'Kho chiu',
    'fear': 'So hai',
    'happy': 'Vui',
    'neutral': 'Trung tinh',
    'sad': 'Buon',
    'surprise': 'Bat ngo',
}

DEFAULT_MODEL = r'D:\Doan\mobilenetv2_fer2013_final.keras'
DEFAULT_TEST = r'D:\Doan\datasets\FER2013\test'
DEFAULT_OUT = r'D:\Doan\Week8_GradCAM_FER_V14'

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Khong doc duoc anh: {path}')
    return img


def model_input_from_bgr(img_bgr, size=(224, 224)):
    # Giong V14: model da co Rescaling noi bo, nen GIU pixel 0..255.
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    return np.expand_dims(x, axis=0)


def batch_from_paths(paths, size=(224, 224)):
    xs = []
    good_paths = []
    for p in paths:
        try:
            bgr = load_bgr(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            xs.append(rgb.astype(np.float32))
            good_paths.append(p)
        except Exception:
            pass
    if not xs:
        return None, []
    return np.stack(xs, axis=0), good_paths


def find_backbone_layer(model):
    # Kien truc da chot trong bao cao:
    # Input -> augmentation -> Rescaling -> MobileNetV2 -> GAP -> Dropout -> Dense7.
    candidates = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                shape = layer.output_shape
            except Exception:
                shape = None
            if shape is not None and len(shape) == 4:
                candidates.append((i, layer))

    if not candidates:
        raise RuntimeError(
            'Khong tim thay backbone 4D trong model.\n'
            'Hay gui model.summary() de kiem tra kien truc.'
        )

    # Uu tien nested model co ten MobileNet.
    for i, layer in candidates:
        if 'mobilenet' in layer.name.lower():
            return i, layer

    return candidates[-1]


def build_gradcam_parts(model):
    """
    Tra ve backbone index + backbone.

    Keras 3 co the bao KeyError neu tao Model(model.inputs, backbone.output)
    khi backbone MobileNetV2 la nested Functional model. Vi vay ban FIXED
    khong tao feature_model tu symbolic tensor cua nested model.

    Thay vao do, luc Grad-CAM ta chay tuan tu CHINH CAC LAYER cua model:
      input -> cac layer truoc backbone -> backbone -> cac layer sau backbone.
    Cach nay giu dung preprocessing noi bo va dung trong so fine-tune.
    """
    backbone_index, backbone = find_backbone_layer(model)
    feature_shape = tuple(int(d) for d in backbone.output.shape[1:])
    return backbone_index, backbone, backbone.name, feature_shape


def forward_split_model(model, backbone_index, backbone, x_tensor):
    """Forward giong model goc nhung tach tai output backbone.

    Tra ve (feature_maps, predictions).
    """
    z = x_tensor

    # Layer 0 thuong la InputLayer. Goi InputLayer truc tiep khong can thiet.
    for layer in model.layers[:backbone_index]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        z = layer(z, training=False)

    feature_maps = backbone(z, training=False)

    y = feature_maps
    for layer in model.layers[backbone_index + 1:]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        y = layer(y, training=False)

    return feature_maps, y


def make_gradcam(model, backbone_index, backbone, x, class_index=None):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, predictions = forward_split_model(
            model, backbone_index, backbone, x_tensor
        )
        # feature_maps nam tren duong tinh gradient toi classifier output.
        tape.watch(feature_maps)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, feature_maps)
    if grads is None:
        raise RuntimeError(
            'Khong tinh duoc gradient cho Grad-CAM. '
            'Hay gui model.summary() neu loi nay xuat hien.'
        )

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    cam = tf.nn.relu(cam)

    max_val = float(tf.reduce_max(cam).numpy())
    if max_val > 1e-12:
        cam = cam / max_val

    return cam.numpy().astype(np.float32), predictions.numpy()[0], class_index


def overlay_gradcam(original_bgr, heatmap, alpha=0.45):
    h, w = original_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, colored, alpha, 0)
    return colored, overlay


def add_ascii_header(image, lines):
    # FER labels dung ASCII de OpenCV hien thi chac chan tren Windows.
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 26
    for line in lines:
        cv2.putText(out, line, (8, y), font, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), font, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
        y += 25
    return out


def save_example(out_root, kind, true_label, image_path, model,
                 backbone_index, backbone, index_no):
    bgr = load_bgr(image_path)
    x = model_input_from_bgr(bgr)

    # Prediction chinh thuc cua model V14.
    probs_main = model(x, training=False).numpy()[0]
    pred_idx = int(np.argmax(probs_main))
    pred_label = FER_CLASSES[pred_idx]
    conf = float(probs_main[pred_idx])

    # Grad-CAM cho lop MODEL DU DOAN.
    heatmap, probs_grad, cam_idx = make_gradcam(
        model,
        backbone_index,
        backbone,
        x,
        class_index=pred_idx,
    )

    # Kiem tra classifier tach ra cho ket qua gan model goc.
    max_diff = float(np.max(np.abs(probs_main - probs_grad)))

    heat_bgr, overlay = overlay_gradcam(bgr, heatmap)

    folder = out_root / kind / true_label
    folder.mkdir(parents=True, exist_ok=True)

    stem = (
        f'{index_no:02d}_true-{true_label}_pred-{pred_label}_'
        f'conf-{conf:.3f}'
    )

    original_path = folder / f'{stem}_original.png'
    heatmap_path = folder / f'{stem}_heatmap.png'
    overlay_path = folder / f'{stem}_overlay.png'

    original_show = add_ascii_header(
        bgr,
        [
            f'True: {true_label}',
            f'Pred: {pred_label} | Conf: {conf:.3f}',
        ]
    )
    overlay_show = add_ascii_header(
        overlay,
        [
            f'Grad-CAM - True: {true_label}',
            f'Pred: {pred_label} | Conf: {conf:.3f}',
        ]
    )

    cv2.imwrite(str(original_path), original_show)
    cv2.imwrite(str(heatmap_path), heat_bgr)
    cv2.imwrite(str(overlay_path), overlay_show)

    return {
        'kind': kind,
        'true_label': true_label,
        'pred_label': pred_label,
        'confidence': conf,
        'source_image': str(image_path),
        'original_file': str(original_path),
        'heatmap_file': str(heatmap_path),
        'overlay_file': str(overlay_path),
        'gradcam_class_index': cam_idx,
        'split_forward_max_prob_diff': max_diff,
    }


def scan_candidates(model, test_dir, scan_per_class=200, batch_size=64):
    """
    Tim 1 vi du dung co confidence cao va 1 vi du sai co confidence cao
    cho moi true class. Khong cherry-pick theo ket qua Test tong the;
    chi dung de chon anh minh hoa Grad-CAM.
    """
    selected = {}

    for true_idx, true_label in enumerate(FER_CLASSES):
        class_dir = test_dir / true_label
        if not class_dir.exists():
            print(f'[BO QUA] Khong co thu muc: {class_dir}')
            continue

        paths = sorted([
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXT
        ])[:scan_per_class]

        best_correct = None
        best_wrong = None

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch_x, good_paths = batch_from_paths(batch_paths)
            if batch_x is None:
                continue

            probs = model(batch_x, training=False).numpy()
            pred_idxs = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            for p, pred_idx, conf in zip(good_paths, pred_idxs, confs):
                pred_idx = int(pred_idx)
                conf = float(conf)
                item = (conf, p, pred_idx)

                if pred_idx == true_idx:
                    if best_correct is None or conf > best_correct[0]:
                        best_correct = item
                else:
                    if best_wrong is None or conf > best_wrong[0]:
                        best_wrong = item

        selected[true_label] = {
            'correct': best_correct,
            'wrong': best_wrong,
        }

        print(
            f'{true_label:9s} | '
            f'correct={best_correct[0]:.3f}' if best_correct else f'{true_label:9s} | correct=NONE',
            end=''
        )
        if best_wrong:
            print(f' | wrong={FER_CLASSES[best_wrong[2]]} ({best_wrong[0]:.3f})')
        else:
            print(' | wrong=NONE')

    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Week 8 - Grad-CAM FER fine-tune dung model V14'
    )
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--test', default=DEFAULT_TEST)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--scan_per_class', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test)
    out_root = Path(args.out)

    if not model_path.exists():
        raise FileNotFoundError(f'Khong tim thay model FER: {model_path}')
    if not test_dir.exists():
        raise FileNotFoundError(f'Khong tim thay FER test: {test_dir}')

    out_root.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('WEEK 8 - GRAD-CAM FER FINE-TUNE - V14 FIXED')
    print('=' * 70)
    print('Model :', model_path)
    print('Test  :', test_dir)
    print('Output:', out_root)
    print('Luu y : model co Rescaling noi bo -> input 0..255 nhu V14.')
    print('=' * 70)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = False

    backbone_index, backbone, backbone_name, feature_shape = build_gradcam_parts(model)

    print('Backbone Grad-CAM:', backbone_name)
    print('Feature shape     :', feature_shape)

    # Self-test split forward tren 1 anh truoc khi quet toan bo.
    first_test = None
    for cls in FER_CLASSES:
        cls_dir = test_dir / cls
        if cls_dir.exists():
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_EXT:
                    first_test = p
                    break
        if first_test is not None:
            break

    if first_test is not None:
        x_test = model_input_from_bgr(load_bgr(first_test))
        main_test = model(x_test, training=False).numpy()[0]
        _, split_test = forward_split_model(
            model, backbone_index, backbone, tf.convert_to_tensor(x_test, dtype=tf.float32)
        )
        split_test = split_test.numpy()[0]
        self_diff = float(np.max(np.abs(main_test - split_test)))
        print(f'Split-forward self-test diff: {self_diff:.8f}')
        if self_diff > 1e-4:
            print('[CANH BAO] Split forward lech model goc > 1e-4.')

    selected = scan_candidates(
        model,
        test_dir,
        scan_per_class=args.scan_per_class,
        batch_size=args.batch_size,
    )

    rows = []
    correct_no = 0
    wrong_no = 0

    for true_label in FER_CLASSES:
        pack = selected.get(true_label, {})

        item = pack.get('correct')
        if item is not None:
            correct_no += 1
            _, path, _ = item
            row = save_example(
                out_root, 'correct', true_label, path, model,
                backbone_index, backbone, correct_no
            )
            rows.append(row)

        item = pack.get('wrong')
        if item is not None:
            wrong_no += 1
            _, path, _ = item
            row = save_example(
                out_root, 'wrong', true_label, path, model,
                backbone_index, backbone, wrong_no
            )
            rows.append(row)

    csv_path = out_root / 'gradcam_examples.csv'
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    max_diff = max((r['split_forward_max_prob_diff'] for r in rows), default=0.0)

    summary_path = out_root / 'GradCAM_Summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('WEEK 8 - GRAD-CAM FER FINE-TUNE - V14 FIXED\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model: {model_path}\n')
        f.write(f'Backbone: {backbone_name}\n')
        f.write(f'Feature shape: {feature_shape}\n')
        f.write(f'Scan per class: {args.scan_per_class}\n')
        f.write(f'Correct examples saved: {correct_no}\n')
        f.write(f'Wrong examples saved: {wrong_no}\n')
        f.write(f'Max diff model vs split forward: {max_diff:.8f}\n')
        f.write('\n')
        f.write('Giai thich:\n')
        f.write('- Grad-CAM duoc tinh tren feature map cuoi cua MobileNetV2 fine-tune FER-2013.\n')
        f.write('- Heatmap chi ho tro giai thich vung anh anh huong toi du doan; khong chung minh quan he nhan qua.\n')
        f.write('- Anh wrong duoc giu lai de phan tich loi, khong loai bo khoi bao cao.\n')

    print('\n' + '=' * 70)
    print('HOAN TAT GRAD-CAM TUAN 8')
    print('Correct examples:', correct_no)
    print('Wrong examples  :', wrong_no)
    print('CSV             :', csv_path)
    print('Summary         :', summary_path)
    print('Thu muc anh     :', out_root)
    print('Max classifier diff:', f'{max_diff:.8f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
