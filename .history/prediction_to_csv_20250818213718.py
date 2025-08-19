#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a CSV summarizing polygon coverage and area (cm²) from YOLO-seg labels.

Filename format: FECHA_numeroid1_numeroid2_numeroid3.jpg
- `FECHA` is the date string.
- `numeroid1_numeroid2_numeroid3` is the unique ID.

Output CSV:
- image_name
- id_combo
- For each FECHA in the dataset, two columns:
    <FECHA>_coverage_pct, <FECHA>_area_cm2

Dependencies:
    pip install opencv-python-headless pandas
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png")


# === Helpers ===
def split_filename(fname: str):
    """Return (fecha, id_combo) from filename FECHA_id1_id2_id3.ext"""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return "", stem
    return parts[0], "_".join(parts[1:])


def load_yolo_seg_polygons(label_path: str):
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 7:  # class + at least 3 (x,y)
                polys.append((t[0], list(map(float, t[1:]))))
    return polys


def norm_to_px(coords, W, H):
    return [(coords[i] * W, coords[i + 1] * H) for i in range(0, len(coords), 2)]


def polygon_area_px(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return 0.0
    return float(cv2.contourArea(pts))


def compute_coverages(img_path, lbl_path, target_class, image_area_cm2):
    """Return (coverage_pct, area_cm2)."""
    img = cv2.imread(img_path)
    if img is None:
        return 0.0, 0.0
    H, W = img.shape[:2]
    if W == 0 or H == 0:
        return 0.0, 0.0

    polys = load_yolo_seg_polygons(lbl_path)
    area_poly_px = 0.0
    for cls, coords in polys:
        if cls != target_class or len(coords) < 6 or len(coords) % 2 != 0:
            continue
        pts = norm_to_px(coords, W, H)
        area_poly_px += polygon_area_px(pts)

    coverage = area_poly_px / (W * H)
    return coverage * 100.0, coverage * image_area_cm2


# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-area-cm2", type=float, default=1000.0)
    parser.add_argument("--target-class", default="0")
    args = parser.parse_args()

    # collect images
    images = [f for f in os.listdir(args.images_dir) if f.lower().endswith(IMG_EXTS)]
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")

    rows = []
    fechas = set()

    for fname in images:
        fecha, id_combo = split_filename(fname)
        fechas.add(fecha)

        img_path = os.path.join(args.images_dir, fname)
        lbl_path = os.path.join(args.labels_dir, os.path.splitext(fname)[0] + ".txt")
        coverage_pct, area_cm2 = compute_coverages(
            img_path, lbl_path, args.target_class, args.image_area_cm2
        )

        rows.append({
            "image_name": fname,
            "id_combo": id_combo,
            "fecha": fecha,
            "coverage_pct": coverage_pct,
            "area_cm2": area_cm2
        })

    # build dataframe
    df = pd.DataFrame(rows)

    # pivot: one row per image, wide format by fecha
    df_wide = df.pivot_table(
        index=["image_name", "id_combo"],
        columns="fecha",
        values=["coverage_pct", "area_cm2"],
        aggfunc="first"
    )

    # flatten MultiIndex columns
    df_wide.columns = [f"{f}_{c}" for f, c in df_wide.columns]
    df_wide = df_wide.reset_index()

    # save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_wide.to_csv(args.output, index=False)
    print(f"✅ CSV written to {args.output}")


if __name__ == "__main__":
    main()
