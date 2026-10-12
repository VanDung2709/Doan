import os
import csv
import argparse


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def count_images(folder):
    return sum(
        1
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Thu muc FER2013/train"
    )

    args = parser.parse_args()

    train_dir = args.train

    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    counts = {}

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        counts[class_name] = count_images(class_dir)

    total = sum(counts.values())
    num_classes = len(class_names)

    print("\n===== PHAN TICH MAT CAN BANG FER-2013 =====")
    print(f"Tong so anh train: {total}")
    print(f"So lop: {num_classes}\n")

    rows = []

    for i, class_name in enumerate(class_names):

        count = counts[class_name]

        percent = (
            count / total * 100
            if total > 0
            else 0
        )

        # Công thức balanced class weight
        balanced_weight = (
            total / (num_classes * count)
            if count > 0
            else 1.0
        )

        # Weight giới hạn để tránh lớp quá ít
        # bị tăng trọng số quá mạnh
        capped_weight = min(
            balanced_weight,
            3.0
        )

        rows.append([
            i,
            class_name,
            count,
            percent,
            balanced_weight,
            capped_weight
        ])

        print(
            f"{i} - {class_name:10s} | "
            f"{count:5d} anh | "
            f"{percent:6.2f}% | "
            f"balanced={balanced_weight:.4f} | "
            f"capped={capped_weight:.4f}"
        )

    max_count = max(counts.values())
    min_count = min(counts.values())

    print("\n===== DANH GIA =====")

    print(
        f"Lop nhieu nhat : "
        f"{max(counts, key=counts.get)} "
        f"({max_count} anh)"
    )

    print(
        f"Lop it nhat    : "
        f"{min(counts, key=counts.get)} "
        f"({min_count} anh)"
    )

    print(
        f"Ty le max/min : "
        f"{max_count / min_count:.2f} lan"
    )

    output_file = "fer2013_class_weights.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "class_name",
            "count",
            "percentage",
            "balanced_weight",
            "capped_weight"
        ])

        writer.writerows(rows)

    print(
        f"\nDa luu ket qua: {output_file}"
    )


if __name__ == "__main__":
    main()