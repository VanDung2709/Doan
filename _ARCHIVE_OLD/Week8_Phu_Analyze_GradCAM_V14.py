from pathlib import Path
import csv
from collections import Counter

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "Week8_GradCAM_FER_V14" / "gradcam_examples.csv"
OUT_PATH = BASE / "Week8_Phu_GradCAM_AutoSummary.txt"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay: {CSV_PATH}")

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    correct = [r for r in rows if r.get("kind") == "correct"]
    wrong = [r for r in rows if r.get("kind") == "wrong"]

    conf_correct = [float(r["confidence"]) for r in correct]
    conf_wrong = [float(r["confidence"]) for r in wrong]

    wrong_pairs = Counter(
        (r["true_label"], r["pred_label"]) for r in wrong
    )

    lines = []
    lines.append("=" * 70)
    lines.append("WEEK 8 - PHUC - PHAN TICH GRAD-CAM / DUNG-SAI - V14")
    lines.append("=" * 70)
    lines.append(f"Correct examples: {len(correct)}")
    lines.append(f"Wrong examples  : {len(wrong)}")
    if conf_correct:
        lines.append(f"Correct conf avg: {sum(conf_correct)/len(conf_correct):.3f}")
        lines.append(f"Correct conf min: {min(conf_correct):.3f}")
        lines.append(f"Correct conf max: {max(conf_correct):.3f}")
    if conf_wrong:
        lines.append(f"Wrong conf avg  : {sum(conf_wrong)/len(conf_wrong):.3f}")
        lines.append(f"Wrong conf min  : {min(conf_wrong):.3f}")
        lines.append(f"Wrong conf max  : {max(conf_wrong):.3f}")

    lines.append("")
    lines.append("CAC VI DU DUNG:")
    for r in correct:
        lines.append(
            f"- {r['true_label']} -> {r['pred_label']} | conf={float(r['confidence']):.3f}"
        )

    lines.append("")
    lines.append("CAC VI DU SAI:")
    for r in wrong:
        lines.append(
            f"- {r['true_label']} -> {r['pred_label']} | conf={float(r['confidence']):.3f}"
        )

    lines.append("")
    lines.append("CAC CAP NHAM:")
    for (true_label, pred_label), n in wrong_pairs.items():
        lines.append(f"- {true_label} -> {pred_label}: {n}")

    lines.append("")
    lines.append("NHAN XET:")
    lines.append("- Confidence cao khong dam bao du doan dung.")
    lines.append("- FER chi nen la tin hieu ho tro trong V14.")
    lines.append("- Grad-CAM chi ho tro giai thich vung anh anh huong den du doan.")
    lines.append("- Khong dung FER/Grad-CAM de chan doan tam ly.")
    lines.append("- Can ket hop FER + Engagement + hanh vi + thoi gian trong Fusion V14.")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("")
    print("Da luu:", OUT_PATH)

if __name__ == "__main__":
    main()
