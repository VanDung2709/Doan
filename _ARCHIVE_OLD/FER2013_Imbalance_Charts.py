import csv
import matplotlib.pyplot as plt


CSV_FILE = "fer2013_class_weights.csv"


class_names = []
counts = []
balanced_weights = []
capped_weights = []


with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        class_names.append(row["class_name"])
        counts.append(int(row["count"]))
        balanced_weights.append(float(row["balanced_weight"]))
        capped_weights.append(float(row["capped_weight"]))


# ==========================================
# BIEU DO 1 - PHAN BO DU LIEU
# ==========================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    class_names,
    counts
)

plt.title("Phan bo du lieu FER-2013 - Training Set")
plt.xlabel("Lop cam xuc")
plt.ylabel("So luong anh")

for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        str(count),
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "fer2013_class_distribution.png",
    dpi=200
)

plt.close()


# ==========================================
# BIEU DO 2 - CLASS WEIGHT
# ==========================================

x = range(len(class_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    balanced_weights,
    width=width,
    label="Balanced weight"
)

plt.bar(
    [i + width / 2 for i in x],
    capped_weights,
    width=width,
    label="Capped weight"
)

plt.xticks(
    list(x),
    class_names
)

plt.title("So sanh Balanced Class Weight va Capped Class Weight")
plt.xlabel("Lop cam xuc")
plt.ylabel("Trong so")
plt.legend()

plt.tight_layout()

plt.savefig(
    "fer2013_class_weight_comparison.png",
    dpi=200
)

plt.close()


print("===== TAO BIEU DO THANH CONG =====")
print("- fer2013_class_distribution.png")
print("- fer2013_class_weight_comparison.png")