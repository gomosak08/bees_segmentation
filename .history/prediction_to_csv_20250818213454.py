#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construye un CSV con:
  - image_name
  - id_combo (numeroid1_numeroid2_numeroid3)
  - Dos columnas por cada FECHA encontrada en los nombres de archivo:
      <FECHA>_coverage_pct, <FECHA>_area_cm2

Asume nombres: FECHA_numeroid1_numeroid2_numeroid3.ext
y labels en formato YOLO-seg (coordenadas normalizadas) en .txt con mismo basename.

Cálculo:
  coverage = area_poligono_px / (W * H)
  area_cm2 = coverage * area_imagen_cm2

Uso:
  python build_csv_by_date.py \
    --images-dir /ruta/a/images \
    --labels-dir /ruta/a/labels \
    --output /ruta/a/salida.csv \
    --image-area-cm2 1000 \
    --target-class 0
"""

from __future__ import annotations
import argparse
import csv
import os
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Genera CSV con dos columnas por fecha (coverage%% y area cm²) a partir de imágenes y labels YOLO-seg."
    )
    ap.add_argument("--images-dir", required=True, help="Directorio con imágenes.")
    ap.add_argument("--labels-dir", required=True, help="Directorio con labels .txt.")
    ap.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    ap.add_argument("--image-area-cm2", type=float, default=1000.0,
                    help="Área física de cada imagen completa en cm².")
    ap.add_argument("--target-class", default="0",
                    help="ID de clase objetivo a medir (por defecto '0').")
    return ap.parse_args()

def list_images(images_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)])

def split_filename(fname: str) -> Tuple[str, str]:
    """
    Devuelve (fecha, id_combo) desde un nombre FECHA_numeroid1_numeroid2_numeroid3.ext
    """
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        # Si no cumple, deja fecha vacía y todo como id_combo
        return "", stem
    fecha = parts[0]
    id_combo = "_".join(parts[1:])
    return fecha, id_combo

def load_yolo_seg_polygons(label_path: str) -> List[Tuple[str, List[float]]]:
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 7:  # class + 3 pares (x,y)
                polys.append((t[0], list(map(float, t[1:]))))
    return polys

def norm_to_px(coords: List[float], W: int, H: int) -> List[Tuple[float, float]]:
    return [(coords[i]*W, coords[i+1]*H) for i in range(0, len(coords), 2)]

def polygon_area_px(points: List[Tuple[float, float]]) -> float:
    # usa OpenCV (shoelace); espera array Nx2
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return 0.0
    return float(cv2.contourArea(pts))

def compute_coverages_for_image(
    img_path: str, lbl_path: str, target_class: str, image_area_cm2: float
) -> Tuple[float, float]:
    """
    Retorna (coverage_pct, area_cm2) para la clase objetivo en una imagen.
    Si no hay label o no hay polígonos válidos => (0.0, 0.0)
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0, 0.0
    H, W = img.shape[:2]
    area_img_px = W * H if (W > 0 and H > 0) else 0
    if area_img_px == 0:
        return 0.0, 0.0

    polys = load_yolo_seg_polygons(lbl_path)
    area_poly_px = 0.0
    for cls, coords in polys:
        if cls != target_class or len(coords) < 6 or len(coords) % 2 != 0:
            continue
        pts = norm_to_px(coords, W, H)
        area_poly_px += polygon_area_px(pts)

    coverage = area_poly_px / area_img_px if area_img_px > 0 else 0.0
    area_cm2 = coverage * image_area_cm2
    return coverage * 100.0, area_cm2  # porcentaje, área en cm²

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No se encontraron imágenes en: {args.images_dir}")

    # Recolectar todas las fechas para definir columnas
    fechas = []
    by_image = []  # filas temporales
    for fname in images:
        fecha, id_combo = split_filename(fname)
        if fecha and fecha not in fechas:
            fechas.append(fecha)

        img_path = os.path.join(args.images_dir, fname)
        lbl_path = os.path.join(args.labels_dir, os.path.splitext(fname)[0] + ".txt")
        coverage_pct, area_cm2 = compute_coverages_for_image(
            img_path, lbl_path, args.target_class, args.image_area_cm2
        )
        by_image.append({
            "image_name": fname,
            "id_combo": id_combo,
            "fecha": fecha,
            "coverage_pct": coverage_pct,
            "area_cm2": area_cm2
        })

    # Armar header: primeras columnas fijas + 2 columnas por fecha
    header = ["image_name", "id_combo"]
    for f in fechas:
        header.append(f"{f}_coverage_pct")
        header.append(f"{f}_area_cm2")

    # Escribir CSV; una fila por imagen: solo columnas de su fecha tendrán datos
    with open(args.output, "w", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=header)
        writer.writeheader()
        for row in by_image:
            out = {"image_name": row["image_name"], "id_combo": row["id_combo"]}
            # inicializa vacías
            for f in fechas:
                out[f"{f}_coverage_pct"] = ""
                out[f"{f}_area_cm2"] = ""
            # rellena las de la fecha de la imagen
            f = row["fecha"]
            if f:
                out[f"{f}_coverage_pct"] = f"{row['coverage_pct']:.6f}"
                out[f"{f}_area_cm2"]     = f"{row['area_cm2']:.6f}"
            writer.writerow(out)

    print(f"✅ CSV generado en: {args.output}")
    print(f"   Columnas fechas: {', '.join(fechas) if fechas else '(sin fechas)'}")

if __name__ == "__main__":
    main()
