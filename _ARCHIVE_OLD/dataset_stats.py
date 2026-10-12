"""
Tuan 1 - Muc 4: Thu thap va lam quen dataset
Thong ke so luong anh/clip tung class de nhan dien mat can bang du lieu.

Dung cho:
  A) FER-2013: cau truc thu muc train/<emotion>/*.jpg
  B) DAiSEE:   file CSV nhan (vi du TrainLabels.csv) co cot Engagement/Boredom/...

Cach chay:
  python dataset_stats.py fer2013 --dir "D:\\Doan\\datasets\\fer2013\\train"
  python dataset_stats.py daisee  --csv "D:\\Doan\\datasets\\daisee\\Labels\\TrainLabels.csv" --column Engagement
"""

import os
import argparse
import matplotlib.pyplot as plt


def stats_fer2013(root_dir, out_path="fer2013_distribution.png"):
    """
    root_dir: thu muc chua cac thu muc con theo ten cam xuc
              (vd: train/angry, train/happy, ...)
    """
    if not os.path.isdir(root_dir):
        print(f"Khong tim thay thu muc: {root_dir}")
        return

    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not classes:
        print("Khong tim thay thu muc con nao (class) ben trong.")
        return

    counts = {}
    valid_ext = (".jpg", ".jpeg", ".png")
    for c in classes:
        class_dir = os.path.join(root_dir, c)
        n = len([f for f in os.listdir(class_dir) if f.lower().endswith(valid_ext)])
        counts[c] = n

    total = sum(counts.values())
    print("\n===== THONG KE FER-2013 =====")
    print(f"Thu muc: {root_dir}")
    print(f"Tong so anh: {total}\n")
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (n / total * 100) if total else 0
        bar = "#" * int(pct / 2)
        print(f"  {c:<10s}: {n:>6d}  ({pct:5.1f}%)  {bar}")

    max_c = max(counts, key=counts.get)
    min_c = min(counts, key=counts.get)
    ratio = counts[max_c] / counts[min_c] if counts[min_c] > 0 else float("inf")
    print(f"\nClass nhieu nhat: {max_c} ({counts[max_c]})")
    print(f"Class it nhat:    {min_c} ({counts[min_c]})")
    print(f"Ty le mat can bang (max/min): {ratio:.1f} lan")

    # Ve bieu do
    plt.figure(figsize=(8, 5))
    names = list(counts.keys())
    values = [counts[n] for n in names]
    colors = ["#C62828" if n == min_c else "#1565C0" for n in names]
    plt.bar(names, values, color=colors)
    plt.title("Phan bo so luong anh theo class - FER-2013")
    plt.ylabel("So luong anh")
    plt.xticks(rotation=30)
    for i, v in enumerate(values):
        plt.text(i, v + total * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nDa luu bieu do: {out_path}")


def stats_daisee(csv_path, column="Engagement", out_path="daisee_distribution.png"):
    """
    csv_path: duong dan file nhan cua DAiSEE (vd TrainLabels.csv)
    column: ten cot muon thong ke (Boredom / Engagement / Confusion / Frustration)
    Cac muc nhan DAiSEE: 0 = Very Low, 1 = Low, 2 = High, 3 = Very High
    """
    import pandas as pd  # import cuc bo, chi can khi dung nhanh DAiSEE

    if not os.path.isfile(csv_path):
        print(f"Khong tim thay file: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # phong truong hop co khoang trang

    if column not in df.columns:
        print(f"Khong tim thay cot '{column}'. Cac cot co san: {list(df.columns)}")
        return

    level_names = {0: "Very Low", 1: "Low", 2: "High", 3: "Very High"}
    counts = df[column].value_counts().sort_index()
    total = counts.sum()

    print(f"\n===== THONG KE DAiSEE - cot '{column}' =====")
    print(f"File: {csv_path}")
    print(f"Tong so clip: {total}\n")
    for level, n in counts.items():
        name = level_names.get(level, str(level))
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        print(f"  Muc {level} ({name:<10s}): {n:>5d}  ({pct:5.1f}%)  {bar}")

    print("\n=> DAiSEE thuong LECH RAT NANG ve phia 'High/Very High' engagement")
    print("   (vi hau het video quay canh hoc binh thuong) - can luu y khi train model.")

    plt.figure(figsize=(7, 5))
    labels = [level_names.get(l, str(l)) for l in counts.index]
    plt.bar(labels, counts.values, color="#2E7D32")
    plt.title(f"Phan bo muc {column} - DAiSEE")
    plt.ylabel("So luong clip")
    for i, v in enumerate(counts.values):
        plt.text(i, v + total * 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nDa luu bieu do: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="dataset", required=True)

    p1 = sub.add_parser("fer2013")
    p1.add_argument("--dir", required=True, help="Thu muc train (co cac thu muc con theo cam xuc)")

    p2 = sub.add_parser("daisee")
    p2.add_argument("--csv", required=True, help="Duong dan file nhan CSV (vd TrainLabels.csv)")
    p2.add_argument("--column", default="Engagement", help="Ten cot: Boredom/Engagement/Confusion/Frustration")

    args = parser.parse_args()

    if args.dataset == "fer2013":
        stats_fer2013(args.dir)
    elif args.dataset == "daisee":
        stats_daisee(args.csv, args.column)


if __name__ == "__main__":
    main()