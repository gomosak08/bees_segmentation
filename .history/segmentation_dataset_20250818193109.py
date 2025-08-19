#!/usr/bin/env python3

"""
Dataset segmentation and splitting script with cropping by 'frame' (class 1) 
and reprojection/clipping of polygons (class 0) to the cropped region.

- Reads images and YOLO-seg labels from flat directories: images/ and labels/.
- Splits the dataset into train/valid/test according to custom percentages.
- For each image, looks for a polygon of class 'frame' (id_clase_marco) and crops 
  the image to the bounding box of that polygon.
- Polygons of the target class (id_clase_obj) are clipped to the cropped area and 
  rewritten normalized to the crop size.

Output structure:
    output_base/<split>/{images,labels}

Expected label format (YOLO-seg):
    <class> x1 y1 x2 y2 x3 y3 ...
where x,y are normalized in [0,1].

Example usage:
    python segmentation_dataset.py \
        --images-dir /home/user/project/images \
        --labels-dir /home/user/project/labels \
        --output-base /home/user/project/segmentation \
        --train 0.7 --valid 0.2 --test 0.1 \
        --id-clase-marco 1 --id-clase-obj 0 \
        --seed 42
"""

from __future__ import annotations
import argparse
import logging
import os
import random
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


# ----------------------------- I/O Utilities -------------------------------- #
def load_label_file(path: str) -> List[List[str]]:
    """Load a YOLO-seg label file into a list of tokens per line."""
    with open(path, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


def save_label_file(path: str, entries: Sequence[Sequence[str]]) -> None:
    """Save a list of label entries (lists of strings) into 'path'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(" ".join(map(str, entry)) + "\n")


# ------------------------- Coordinate Conversions --------------------------- #
def normalized_to_pixel(coords: Sequence[float], img_w: int, img_h: int) -> List[Tuple[float, float]]:
    """Convert normalized coordinates [x1,y1,x2,y2,...] to pixel coordinates [(x,y),...]."""
    return [(float(x) * img_w, float(y) * img_h) for x, y in zip(coords[::2], coords[1::2])]


def pixel_to_normalized(coords: Iterable[Tuple[float, float]], img_w: int, img_h: int) -> List[float]:
    """Convert pixel coordinates [(x,y),...] to normalized coordinates [x1,y1,...]."""
    out: List[float] = []
    for x, y in coords:
        out.extend([x / img_w, y / img_h])
    return out


def bbox_from_polygon(poly_px: Sequence[Tuple[float, float]], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Return bounding box limited to image size from a polygon in pixel coordinates."""
    xs, ys = zip(*poly_px)
    x1 = int(max(0, min(xs)))
    y1 = int(max(0, min(ys)))
    x2 = int(min(img_w, max(xs)))
    y2 = int(min(img_h, max(ys)))
    return x1, y1, x2, y2


# --------------------- Polygon Clipping (Sutherland–Hodgman) ---------------- #
def clip_polygon_to_rect(poly: Sequence[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    """
    Clip a polygon to a rectangle [0,w]x[0,h].
    Returns a list of (x,y) in pixels. If the result is degenerate, returns [].
    """
    def inside(p: Tuple[float, float], edge: str) -> bool:
        x, y = p
        if edge == "left":   return x >= 0
        if edge == "right":  return x <= w
        if edge == "top":    return y >= 0
        if edge == "bottom": return y <= h
        return True

    def intersect(p1: Tuple[float, float], p2: Tuple[float, float], edge: str) -> Tuple[float, float]:
        x1, y1 = p1; x2, y2 = p2
        if (x1, y1) == (x2, y2):
            return p1
        if edge in ("left", "right"):
            x_edge = 0 if edge == "left" else w
            if x2 != x1:
                t = (x_edge - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                return (x_edge, y)
            return (x1, y1)
        else:
            y_edge = 0 if edge == "top" else h
            if y2 != y1:
                t = (y_edge - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                return (x, y_edge)
            return (x1, y1)

    out = list(poly)
    for edge in ("left", "right", "top", "bottom"):
        if not out:
            break
        inp = out
        out = []
        s = inp[-1]
        for e in inp:
            if inside(e, edge):
                if inside(s, edge):
                    out.append(e)
                else:
                    out.append(intersect(s, e, edge))
                    out.append(e)
            else:
                if inside(s, edge):
                    out.append(intersect(s, e, edge))
            s = e

    out = [(float(x), float(y)) for x, y in out if np.isfinite(x) and np.isfinite(y)]
    return out if len(out) >= 3 else []


# --------------------------- Main Processing -------------------------------- #
def build_splits(
    images_dir: str,
    labels_dir: str,
    train: float,
    valid: float,
    test: float,
    seed: int | None = None,
) -> dict[str, List[str]]:
    """Generate file lists for each split, filtering only those with matching label files."""
    if not (0 < train <= 1 and 0 <= valid <= 1 and 0 <= test <= 1):
        raise ValueError("Split ratios must be in (0,1].")

    if not np.isclose(train + valid + test, 1.0):
        raise ValueError("The sum of --train + --valid + --test must equal 1.0.")

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
    paired = [f for f in all_images if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))]

    if seed is not None:
        random.seed(seed)
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(n_total * train)
    n_valid = int(n_total * valid)

    files_train = paired[:n_train]
    files_valid = paired[n_train:n_train + n_valid]
    files_test  = paired[n_train + n_valid:]

    return {"train": files_train, "valid": files_valid, "test": files_test}


def process_dataset(
    images_dir: str,
    labels_dir: str,
    output_base: str,
    split_dict: dict[str, List[str]],
    frame_class: str = "1",
    target_class: str = "0",
) -> Tuple[int, int, int]:
    """
    Process files split into train/valid/test:
    - Crop each image by 'frame' polygon (class=frame_class).
    - Clip and normalize polygons of target_class to the crop.
    - Save results into output_base/<split>/images and labels.

    Returns counters: (processed, skipped_no_frame, skipped_bad).
    """
    for split in split_dict:
        os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)

    processed = skipped_no_frame = skipped_bad = 0

    for split, files in split_dict.items():
        out_img_dir = os.path.join(output_base, split, "images")
        out_lbl_dir = os.path.join(output_base, split, "labels")

        for file in files:
            base, _ext = os.path.splitext(file)
            img_path = os.path.join(images_dir, file)
            lbl_path = os.path.join(labels_dir, base + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                logging.warning("Could not read image: %s", file)
                skipped_bad += 1
                continue

            img_h, img_w = img.shape[:2]
            labels = load_label_file(lbl_path)

            # 1) Find frame (class=frame_class)
            frame_box: Tuple[int, int, int, int] | None = None
            for lab in labels:
                if lab and lab[0] == frame_class and len(lab) >= 6:
                    coords = list(map(float, lab[1:]))
                    poly_px = normalized_to_pixel(coords, img_w, img_h)
                    x1, y1, x2, y2 = bbox_from_polygon(poly_px, img_w, img_h)
                    if (x2 - x1) > 1 and (y2 - y1) > 1:
                        frame_box = (x1, y1, x2, y2)
                        break

            if frame_box is None:
                logging.info("%s: no frame (class %s), skipped.", file, frame_class)
                skipped_no_frame += 1
                continue

            # 2) Crop image
            x1, y1, x2, y2 = frame_box
            crop = img[y1:y2, x1:x2]
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w <= 1 or crop_h <= 1:
                logging.info("%s: invalid frame box, skipped.", file)
                skipped_bad += 1
                continue

            # 3) Clip and normalize polygons of target class
            updated: List[List[str]] = []
            for lab in labels:
                if not lab or lab[0] != target_class:
                    continue
                coords = list(map(float, lab[1:]))
                if len(coords) < 6 or len(coords) % 2 != 0:
                    continue

                poly_px = normalized_to_pixel(coords, img_w, img_h)
                shifted = [(x - x1, y - y1) for (x, y) in poly_px]
                clipped = clip_polygon_to_rect(shifted, crop_w, crop_h)
                if not clipped:
                    continue
                norm_coords = pixel_to_normalized(clipped, crop_w, crop_h)
                updated.append([target_class] + [f"{v:.6f}" for v in norm_coords])

            # 4) Save cropped image and labels (empty file if no polygons remain)
            out_img_path = os.path.join(out_img_dir, file)
            out_lbl_path = os.path.join(out_lbl_dir, base + ".txt")
            cv2.imwrite(out_img_path, crop)
            save_label_file(out_lbl_path, updated)
            processed += 1
            logging.debug("%s/%s: cropped and labels updated.", split, file)

    return processed, skipped_no_frame, skipped_bad


# ------------------------------ CLI / Main ---------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split dataset (images/labels) into train/valid/test, "
                    "crop by 'frame' polygons and rescale target polygons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", required=True, help="Input images directory.")
    p.add_argument("--labels-dir", required=True, help="Input labels directory.")
    p.add_argument("--output-base", required=True, help="Output base directory.")

    p.add_argument("--train", type=float, default=0.7, help="Split ratio for training.")
    p.add_argument("--valid", type=float, default=0.2, help="Split ratio for validation.")
    p.add_argument("--test",  type=float, default=0.1, help="Split ratio for testing.")

    p.add_argument("--frame-class", default="1", help="Class ID of the frame polygon.")
    p.add_argument("--target-class", default="0", help="Class ID of target polygons to keep.")

    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()

def create_yml(path: str) -> None:
    data = {
        train: "../train/images",
        val: "../valid/images",
        test: "../test/images",
        nc: 2,
        names: ['cria', 'marco'],
        roboflow:
            workspace: unam-1m4k0
            project: abejas-2
            version: 5
            license: CC BY 4.0
            url: https://universe.roboflow.com/unam-1m4k0/abejas-2/dataset/5
    }

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    split_dict = build_splits(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        train=args.train, valid=args.valid, test=args.test,
        seed=args.seed,
    )

    processed, no_frame, bad = process_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_base=args.output_base,
        split_dict=split_dict,
        frame_class=args.frame_class,
        target_class=args.target_class,
    )

    print(f"\n✅ Done. Processed: {processed} | Skipped (no frame): {no_frame} | Skipped (errors): {bad}")
    print(f"Output structure: {args.output_base}/<train|valid|test>/(images|labels)")


if __name__ == "__main__":
    main()
