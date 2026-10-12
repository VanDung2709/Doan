#!/usr/bin/env python3
"""Restore clean FER originals and rebuild Grad-CAM overlays from existing heatmaps.

Run from D:\\Doan after extracting this script there. This does not retrain the model
or change any AI/Fusion logic. It uses source_image paths recorded in the Grad-CAM CSV.
"""
from __future__ import annotations

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Thieu Pillow. Cai bang: python -m pip install Pillow")
    raise SystemExit(2)

PROJECT_DIR = Path(__file__).resolve().parent
GRADCAM_DIR = PROJECT_DIR / "Week8_GradCAM_FER_V14"
CSV_PATH = GRADCAM_DIR / "gradcam_examples.csv"


def resolve_project_path(raw_path: str, kind: str) -> Path:
    """Resolve original Windows paths, including if the project drive changed."""
    raw = (raw_path or "").strip().strip('"')
    if not raw:
        return Path()
    direct = Path(raw)
    if direct.exists():
        return direct

    # The CSV stores absolute Windows paths such as D:\\Doan\\datasets\\... .
    # When the project is moved, rebuild a path relative to this script's directory.
    parts = list(PureWindowsPath(raw).parts)
    marker = "datasets" if kind == "source" else "Week8_GradCAM_FER_V14"
    lower = [part.lower() for part in parts]
    try:
        idx = lower.index(marker.lower())
    except ValueError:
        return direct
    return PROJECT_DIR.joinpath(*parts[idx:])


def inside_project(path: Path) -> Path:
    """Ensure output files stay in the project's Grad-CAM folder."""
    path = path.resolve()
    try:
        path.relative_to(GRADCAM_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Duong dan output nam ngoai thu muc Grad-CAM: {path}") from exc
    return path


def main() -> int:
    if not CSV_PATH.is_file():
        print(f"[ERROR] Khong tim thay: {CSV_PATH}")
        print("Hay dat script nay trong D:\\Doan va giu thu muc Week8_GradCAM_FER_V14.")
        return 2

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_image", "original_file", "heatmap_file", "overlay_file"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            print("[ERROR] CSV thieu cot:", ", ".join(sorted(missing_columns)))
            return 2
        rows = list(reader)

    rows = [row for row in rows if row.get("source_image")]
    if len(rows) != 14:
        print(f"[WARNING] CSV hien co {len(rows)} dong co source_image; mong doi 14 mau.")

    jobs = []
    errors = []
    for line_num, row in enumerate(rows, start=2):
        src = resolve_project_path(row.get("source_image", ""), "source")
        original = inside_project(resolve_project_path(row.get("original_file", ""), "output"))
        heatmap = inside_project(resolve_project_path(row.get("heatmap_file", ""), "output"))
        overlay = inside_project(resolve_project_path(row.get("overlay_file", ""), "output"))
        if not src.is_file():
            errors.append(f"Dong {line_num}: khong tim thay anh FER goc: {src}")
        if not heatmap.is_file():
            errors.append(f"Dong {line_num}: khong tim thay heatmap: {heatmap}")
        if not original.is_file():
            errors.append(f"Dong {line_num}: khong tim thay anh original can thay: {original}")
        if not overlay.is_file():
            errors.append(f"Dong {line_num}: khong tim thay overlay can thay: {overlay}")
        jobs.append((src, original, heatmap, overlay))

    if errors:
        print("[ERROR] Chua sua file nao vi con thieu dau vao:")
        for error in errors:
            print(" -", error)
        print("Kiem tra du lieu tai D:\\Doan\\datasets\\FER2013\\test.")
        return 2

    backup_dir = PROJECT_DIR / ("Week8_GradCAM_FER_V14_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    saved = 0
    try:
        for src, original, heatmap, overlay in jobs:
            for target in (original, overlay):
                rel = target.relative_to(GRADCAM_DIR.resolve())
                backup_target = backup_dir / rel
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)

            # Use the real source image from FER2013, not the old annotated thumbnail.
            with Image.open(src) as im:
                clean_original = im.convert("RGB")
            with Image.open(heatmap) as im:
                clean_heatmap = im.convert("RGB")

            if clean_heatmap.size != clean_original.size:
                clean_heatmap = clean_heatmap.resize(clean_original.size, Image.Resampling.BILINEAR)

            original.parent.mkdir(parents=True, exist_ok=True)
            overlay.parent.mkdir(parents=True, exist_ok=True)
            clean_original.save(original, format="PNG")

            # Rebuild the overlay without any text burned into the prior overlay file.
            blended = Image.blend(clean_original, clean_heatmap, alpha=0.42)
            blended.save(overlay, format="PNG")
            saved += 1

    except Exception as exc:
        print(f"[ERROR] Dung khi dang sua file: {exc}")
        print(f"Ban sao cac file da duoc tao o: {backup_dir}")
        return 1

    print("[OK] Da lam sach anh goc va tao lai overlay Grad-CAM.")
    print(f"[OK] So mau xu ly: {saved}")
    print(f"[OK] Heatmap goc duoc giu nguyen; khong retrain model/Fusion V14.")
    print(f"[OK] Ban sao truoc khi sua: {backup_dir}")
    print("Mo lai Dashboard > Grad-CAM de kiem tra anh goc khong con chu True/Pred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
