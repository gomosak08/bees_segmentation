#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a CSV keyed ONLY by id_combo.

Expected filename format:
    FECHA_numeroid1_numeroid2_numeroid3.jpg
Where FECHA varies per image, and id_combo = "numeroid1_numeroid2_numeroid3".

Output CSV:
    One row per id_combo
    Two columns per FECHA: <FECHA>_coverage_pct, <FECHA>_area_cm2

Coverage/area are computed from YOLO-seg labels (normalized polygons).


# Promedio cuando hay múltiples imágenes del mismo id_combo y fecha
python prediction_to_csv.py \
  --images-dir /home/gomosak/abejas/outputs/test_predict \
  --labels-dir /home/gomosak/abejas/outputs/test_predict/labels \
  --output /home/gomosak/abejas/outputs/test_predict/salida.csv \
  --image-area-cm2 1000 \
  --target-class 0 

"""

#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png")

def split_filename(fname: str):
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
            if len(t) >= 7:
                polys.append((t[0], list(map(float, t[1:]))))
    return polys

def norm_to_px(coords, W, H):
    return [(coords[i] * W, coords[i + 1] * H) for i in range(0, len(coords), 2)]

def polygon_area_px(points):
    pts = np.asarray(points, dtype=np.float32)
    return float(cv2.contourArea(pts)) if pts.shape[0] >= 3 else 0.0

def compute_coverages(img_path, lbl_path, target_class, image_area_cm2):
    img = cv2.imread(img_path)
    if img is None:
        return 0.0, 0.0
    H, W = img.shape[:2]
    polys = load_yolo_seg_polygons(lbl_path)
    area_poly_px = 0.0
    for cls, coords in polys:
        if cls != target_class: 
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        pts = norm_to_px(coords, W, H)
        area_poly_px += polygon_area_px(pts)
    coverage = area_poly_px / (W * H) if W > 0 and H > 0 else 0.0
    return coverage * 100.0, coverage * image_area_cm2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-area-cm2", type=float, default=1000.0)
    ap.add_argument("--target-class", default="0")
    args = ap.parse_args()

    images = [f for f in os.listdir(args.images_dir) if f.lower().endswith(IMG_EXTS)]
    if not images:
        raise SystemExit(f"No images in {args.images_dir}")

    rows = []
    for fname in images:
        fecha, id_combo = split_filename(fname)
        img_path = os.path.join(args.images_dir, fname)
        lbl_path = os.path.join(args.labels_dir, os.path.splitext(fname)[0] + ".txt")
        cov_pct, area_cm2 = compute_coverages(img_path, lbl_path, args.target_class, args.image_area_cm2)
        rows.append({"id_combo": id_combo, f"{fecha}_coverage_pct": cov_pct, f"{fecha}_area_cm2": area_cm2})

    df_new = pd.DataFrame(rows)
    df_new = df_new.groupby("id_combo").mean().reset_index()

    # si ya existe, lo leemos y mergeamos
    if os.path.exists(args.output):
        df_old = pd.read_csv(args.output)
        df_out = pd.merge(df_old, df_new, on="id_combo", how="outer")
    else:
        df_out = df_new

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"✅ CSV updated at {args.output}")

if __name__ == "__main__":
    main()
