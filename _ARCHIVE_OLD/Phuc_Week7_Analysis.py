
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = Path(r"D:\Doan\Week7_Binary_Comparison\binary_comparison_week7.csv")
OUTPUT_DIR = Path(r"D:\Doan\Week7_Phuc_Analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOUR_CLASS = {
    "Baseline": {
        "accuracy": 0.4867, "macro_f1": 0.2869,
        "very_low_f1": 0.0000, "low_f1": 0.1400,
        "high_f1": 0.5003, "very_high_f1": 0.5071,
    },
    "Config1_Unfreeze40": {
        "accuracy": 0.4928, "macro_f1": 0.2879,
        "very_low_f1": 0.0000, "low_f1": 0.1370,
        "high_f1": 0.4982, "very_high_f1": 0.5164,
    },
    "Config2_ClassWeight10": {
        "accuracy": 0.4640, "macro_f1": 0.2779,
        "very_low_f1": 0.0000, "low_f1": 0.1509,
        "high_f1": 0.4815, "very_high_f1": 0.4790,
    },
    "Config3_FocalLoss": {
        "accuracy": 0.4676, "macro_f1": 0.2766,
        "very_low_f1": 0.0000, "low_f1": 0.1521,
        "high_f1": 0.4373, "very_high_f1": 0.5171,
    },
}

DISPLAY_NAMES = {
    "Baseline": "Baseline",
    "Config1_Unfreeze40": "Config 1\nUnfreeze 40",
    "Config2_ClassWeight10": "Config 2\nClass Weight 10",
    "Config3_FocalLoss": "Config 3\nFocal Loss",
}

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Khong tim thay file: {INPUT_CSV}")

rows = []
with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        parsed = {"model": row["model"]}
        for key, value in row.items():
            if key != "model":
                parsed[key] = float(value)
        rows.append(parsed)

order = [
    "Baseline",
    "Config1_Unfreeze40",
    "Config2_ClassWeight10",
    "Config3_FocalLoss",
]
row_map = {row["model"]: row for row in rows}
rows = [row_map[name] for name in order if name in row_map]

models = [row["model"] for row in rows]
labels = [DISPLAY_NAMES.get(model, model) for model in models]
x = np.arange(len(models))
width = 0.36

# 1) 4-class Accuracy vs Macro F1
acc_4 = [FOUR_CLASS[m]["accuracy"] * 100 for m in models]
macro_4 = [FOUR_CLASS[m]["macro_f1"] * 100 for m in models]

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, acc_4, width, label="Accuracy")
plt.bar(x + width/2, macro_4, width, label="Macro F1")
plt.xticks(x, labels)
plt.ylabel("Ty le (%)")
plt.title("So sanh 4-class DAiSEE: Accuracy va Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_4class_accuracy_macro_f1.png", dpi=180)
plt.close()

# 2) F1 theo tung lop
class_keys = ["very_low_f1", "low_f1", "high_f1", "very_high_f1"]
class_labels = ["Very Low", "Low", "High", "Very High"]

plt.figure(figsize=(11, 6))
for class_key, class_label in zip(class_keys, class_labels):
    values = [FOUR_CLASS[m][class_key] for m in models]
    plt.plot(labels, values, marker="o", label=class_label)
plt.ylabel("F1-score")
plt.title("F1-score theo tung lop Engagement")
plt.ylim(0, 0.60)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_4class_f1_by_class.png", dpi=180)
plt.close()

# 3) Binary Validation/Test Macro F1
val_macro = [row["val_macro_f1"] for row in rows]
test_macro = [row["test_macro_f1"] for row in rows]

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, val_macro, width, label="Validation Macro F1")
plt.bar(x + width/2, test_macro, width, label="Test Macro F1")
plt.xticks(x, labels)
plt.ylabel("Macro F1")
plt.title("Binary Low/High - So sanh Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_binary_macro_f1.png", dpi=180)
plt.close()

# 4) Test LOW F1 + Test Accuracy
test_low_f1 = [row["test_low_f1"] for row in rows]
test_accuracy = [row["test_accuracy"] for row in rows]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(x, test_low_f1, width=0.5, label="Test LOW F1")
ax1.set_ylabel("LOW F1")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

ax2 = ax1.twinx()
ax2.plot(x, [v * 100 for v in test_accuracy], marker="o", label="Test Accuracy")
ax2.set_ylabel("Test Accuracy (%)")
ax1.set_title("Binary Low/High - LOW F1 va Accuracy tren Test")

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "04_binary_low_f1_accuracy.png", dpi=180)
plt.close()

# 5) Threshold tren Validation
thresholds = [row["threshold_from_validation"] for row in rows]

plt.figure(figsize=(10, 6))
plt.bar(labels, thresholds)
plt.ylabel("Threshold")
plt.title("Threshold Low Engagement chon tren Validation")
plt.ylim(0, max(thresholds) + 0.08)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_validation_threshold.png", dpi=180)
plt.close()

best_val = max(
    rows,
    key=lambda row: (row["val_macro_f1"], row["val_low_f1"])
)
baseline = row_map["Baseline"]

summary_lines = [
    "PHUC - TUAN 7 - PHAN TICH KET QUA TOI UU DAISEE",
    "=" * 65,
    "",
    "1. KET QUA 4-CLASS",
    "",
]

for model in models:
    m = FOUR_CLASS[model]
    summary_lines.append(
        f"{model}: Accuracy={m['accuracy']*100:.2f}% | "
        f"Macro F1={m['macro_f1']:.4f} | "
        f"Very Low F1={m['very_low_f1']:.4f} | "
        f"Low F1={m['low_f1']:.4f}"
    )

summary_lines += [
    "",
    "Nhan xet:",
    "- Very Low co F1 = 0 o ca 4 cau hinh.",
    "- Nguyen nhan chinh la mat can bang du lieu rat manh va so clip Very Low qua it.",
    "- Unfreeze 40, Class Weight 10 va Focal Loss khong cai thien on dinh bai toan 4 lop.",
    "",
    "2. KET QUA BINARY LOW / HIGH",
    "",
]

for row in rows:
    summary_lines.append(
        f"{row['model']}: Threshold={row['threshold_from_validation']:.2f} | "
        f"Val Macro F1={row['val_macro_f1']:.4f} | "
        f"Test Macro F1={row['test_macro_f1']:.4f} | "
        f"Test LOW F1={row['test_low_f1']:.4f} | "
        f"Test Accuracy={row['test_accuracy']*100:.2f}%"
    )

summary_lines += [
    "",
    "3. LUA CHON MODEL",
    "",
    f"Model co Validation Macro F1 cao nhat: {best_val['model']} "
    f"({best_val['val_macro_f1']:.4f}).",
    "Tuy nhien cau hinh nay khong duy tri duoc hieu nang tot tren Test.",
    f"Baseline: Test Accuracy={baseline['test_accuracy']*100:.2f}%, "
    f"Test Macro F1={baseline['test_macro_f1']:.4f}, "
    f"Test LOW F1={baseline['test_low_f1']:.4f}.",
    "Config 3 co Test Macro F1/LOW F1 cao hon, nhung khong duoc chon chi dua tren Test "
    "de tranh cherry-pick.",
    "",
    "KET LUAN:",
    "- Khong co cau hinh moi nao cai thien on dinh tren ca Validation va Test.",
    "- Giu Baseline lam model Engagement chinh.",
    "- Giu threshold Low Engagement = 0.27 da chon tren Validation.",
    "- Binary Low/High phu hop muc tieu phat hien mat tap trung hon phan loai 4 muc.",
]

summary_path = OUTPUT_DIR / "Phuc_Week7_Analysis_Summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

compact_csv = OUTPUT_DIR / "Phuc_Week7_Comparison_Table.csv"
with compact_csv.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model", "4Class Accuracy", "4Class Macro F1",
        "4Class Very Low F1", "4Class Low F1",
        "Binary Threshold", "Val Macro F1",
        "Test Macro F1", "Test LOW F1", "Test Accuracy",
    ])
    for row in rows:
        m = FOUR_CLASS[row["model"]]
        writer.writerow([
            row["model"],
            f"{m['accuracy']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['very_low_f1']:.4f}",
            f"{m['low_f1']:.4f}",
            f"{row['threshold_from_validation']:.2f}",
            f"{row['val_macro_f1']:.4f}",
            f"{row['test_macro_f1']:.4f}",
            f"{row['test_low_f1']:.4f}",
            f"{row['test_accuracy']:.4f}",
        ])

print("===== PHUC WEEK 7 ANALYSIS DONE =====")
print("Output:", OUTPUT_DIR)
print("Da tao 5 bieu do PNG.")
print("Da tao:", summary_path.name)
print("Da tao:", compact_csv.name)
print("Ket luan: GIU BASELINE + threshold 0.27")
