#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base loader to iterate over a directory of images and a directory of labels.
- Pairs files by basename (e.g., IMG_001.jpg <-> IMG_001.txt)
- Reports missing counterparts
- Yields each (image_path, label_path) so you can plug further processing

Usage:
  python dataset_loader.py \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --exts .jpg .jpeg .png \
    --print-first 5 \
    --strict
"""

from __future__ import annotations
import argparse
import logging
import os
from typing import Generator, Iterable, List, Tuple

IMG_EXTS_DEFAULT = (".jpg", ".jpeg", ".png")

def list_files_with_exts(d: str, exts: Iterable[str]) -> List[str]:
    """List files in directory d filtering by extensions (case-insensitive)."""
    exts = tuple(e.lower() for e in exts)
    return sorted([f for f in os.listdir(d) if f.lower().endswith(exts)])

def pair_images_and_labels(
    images_dir: str,
    labels_dir: str,
    img_exts: Iterable[str],
    label_ext: str = ".txt",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (paired_basenames, missing_labels, missing_images)
    - paired_basenames: names present in both images_dir and labels_dir
    - missing_labels: image basenames that have no label
    - missing_images: label basenames that have no image
    """
    images = list_files_with_exts(images_dir, img_exts)
    labels = list_files_with_exts(labels_dir, (label_ext,))

    img_bases = {os.path.splitext(f)[0] for f in images}
    lbl_bases = {os.path.splitext(f)[0] for f in labels}

    paired = sorted(img_bases & lbl_bases)
    missing_labels = sorted(img_bases - lbl_bases)
    missing_images = sorted(lbl_bases - img_bases)

    return paired, missing_labels, missing_images

def iter_pairs(
    images_dir: str,
    labels_dir: str,
    basenames: Iterable[str],
    img_exts: Iterable[str],
    label_ext: str = ".txt",
) -> Generator[Tuple[str, str], None, None]:
    """
    Yields (image_path, label_path) for each basename.
    Picks the first matching image extension found in img_exts order.
    """
    for base in basenames:
        img_path = None
        for ext in img_exts:
            p = os.path.join(images_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            logging.warning("No image file found for base '%s' using %s", base, img_exts)
            continue

        lbl_path = os.path.join(labels_dir, base + label_ext)
        if not os.path.exists(lbl_path):
            logging.warning("No label file found for base '%s'", base)
            continue

        yield img_path, lbl_path

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Iterate over images and labels directories, pairing by basename.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images-dir", required=True, help="Directory containing images.")
    ap.add_argument("--labels-dir", required=True, help="Directory containing label .txt files.")
    ap.add_argument("--exts", nargs="+", default=list(IMG_EXTS_DEFAULT),
                    help="Accepted image extensions (order matters).")
    ap.add_argument("--label-ext", default=".txt", help="Label file extension.")
    ap.add_argument("--print-first", type=int, default=0,
                    help="If >0, print the first N pairs and exit.")
    ap.add_argument("--strict", action="store_true",
                    help="If set, exit with error code when missing pairs are found.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    # Validate directories
    if not os.path.isdir(args.images_dir):
        raise NotADirectoryError(f"Images dir not found: {args.images_dir}")
    if not os.path.isdir(args.labels_dir):
        raise NotADirectoryError(f"Labels dir not found: {args.labels_dir}")

    paired, missing_labels, missing_images = pair_images_and_labels(
        args.images_dir, args.labels_dir, tuple(args.exts), args.label_ext
    )

    logging.info("Found %d paired items.", len(paired))
    if missing_labels:
        logging.warning("Missing labels for %d images: e.g. %s", len(missing_labels), missing_labels[:5])
    if missing_images:
        logging.warning("Missing images for %d labels: e.g. %s", len(missing_images), missing_images[:5])

    if args.strict and (missing_labels or missing_images):
        logging.error("Strict mode: missing counterparts detected. Exiting with error.")
        raise SystemExit(1)

    # Print N pairs (dry run) or iterate all (hook for future processing)
    if args.print_first > 0:
        for i, (img_p, lbl_p) in enumerate(iter_pairs(args.images_dir, args.labels_dir, paired, args.exts, args.label_ext), 1):
            print(f"[{i}] IMG: {img_p} | LBL: {lbl_p}")
            if i >= args.print_first:
                break
        return

    # Placeholder for future processing loop
    for i, (img_p, lbl_p) in enumerate(iter_pairs(args.images_dir, args.labels_dir, paired, args.exts, args.label_ext), 1):
        # TODO: add your processing here (read image, parse label, etc.)
        # Example (commented):
        # img = cv2.imread(img_p)
        # labels = open(lbl_p).read().strip().splitlines()
        if i % 200 == 0:
            logging.info("Iterated %d pairs...", i)

    logging.info("Done. Total paired items iterated: %d", len(paired))

if __name__ == "__main__":
    main()
    """# Solo listar y contar (imprime los primeros 5 pares)
        python dataset_loader.py \
        --images-dir /home/gomosak/abejas/abejas/images \
        --labels-dir /home/gomosak/abejas/abejas/labels \
        --print-first 5

        # Iterar todo y avisar si faltan pares (modo estricto)
        python dataset_loader.py \
        --images-dir /home/gomosak/abejas/abejas/images \
        --labels-dir /home/gomosak/abejas/abejas/labels \
        --strict --log-level INFO
        """