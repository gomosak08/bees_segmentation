#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rename_yolo_pairs.py
--------------------
Safely rename paired YOLO image/label files **in-place** across two folders.
- Pairs are matched by base filename (e.g., "abc.jpg" ↔ "abc.txt").
- Creates sequential new names with a chosen prefix (default: img_000001).
- Does NOT move files; only renames inside the same folders.
- Avoids collisions by using a two-phase rename (via temporary names).
- Can export a CSV mapping and undo from it later.

Usage (preview / dry-run by default):
    python rename_yolo_pairs.py --images /path/images --labels /path/labels --prefix bee --start 1 --zfill 6

Apply changes:
    python rename_yolo_pairs.py --images /path/images --labels /path/labels --prefix bee --apply

Export mapping CSV:
    python rename_yolo_pairs.py --images ... --labels ... --apply --map-out rename_map.csv

Undo from mapping CSV:
    python rename_yolo_pairs.py --undo-from rename_map.csv
"""

from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_images(images_dir: Path) -> Dict[str, Path]:
    out = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out

def find_labels(labels_dir: Path, label_ext: str) -> Dict[str, Path]:
    out = {}
    for p in labels_dir.iterdir():
        if p.is_file() and p.suffix.lower() == label_ext.lower():
            out[p.stem] = p
    return out

def collect_pairs(images_dir: Path, labels_dir: Path, label_ext: str) -> List[Tuple[str, Path, Path]]:
    imgs = find_images(images_dir)
    labs = find_labels(labels_dir, label_ext)
    common = sorted(set(imgs.keys()) & set(labs.keys()))
    pairs = [(stem, imgs[stem], labs[stem]) for stem in common]
    missing_labels = sorted(set(imgs.keys()) - set(labs.keys()))
    missing_images = sorted(set(labs.keys()) - set(imgs.keys()))
    return pairs, missing_labels, missing_images

def build_new_names(n: int, prefix: str, start: int, zfill: int) -> List[str]:
    if zfill <= 0:
        # auto zfill based on highest index
        max_idx = start + n - 1
        zfill = max(1, len(str(max_idx)))
    return [f"{prefix}_{str(start + i).zfill(zfill)}" for i in range(n)]

def two_phase_rename(renames: List[Tuple[Path, Path]], dry_run: bool):
    """
    Perform two-phase rename to avoid name collisions:
    1) rename source -> source.with_name(source.name + '.tmp_renaming')
    2) rename tmp -> final
    """
    tmp_suffix = ".tmp_renaming"
    tmps: List[Tuple[Path, Path]] = []

    # Phase 0: check collisions up-front
    for src, dst in renames:
        if src.resolve() == dst.resolve():
            # no-op
            continue
        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")

    # Phase 1: to temp
    for src, dst in renames:
        if src.resolve() == dst.resolve():
            continue
        tmp = src.with_name(src.name + tmp_suffix)
        tmps.append((tmp, dst))
        print(f"[TEMP ] {src.name}  ->  {tmp.name}")
        if not dry_run:
            src.rename(tmp)

    # Phase 2: temp to final
    for tmp, dst in tmps:
        print(f"[FINAL] {tmp.name}  ->  {dst.name}")
        if not dry_run:
            tmp.rename(dst)

def write_mapping_csv(path: Path, rows: List[Tuple[Path, Path, Path, Path]]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_image", "old_label", "new_image", "new_label"])
        for oi, ol, ni, nl in rows:
            w.writerow([str(oi), str(ol), str(ni), str(nl)])

def read_mapping_csv(path: Path) -> List[Tuple[Path, Path, Path, Path]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            oi = Path(row["old_image"]); ol = Path(row["old_label"])
            ni = Path(row["new_image"]); nl = Path(row["new_label"])
            rows.append((oi, ol, ni, nl))
    return rows

def do_apply(images_dir: Path, labels_dir: Path, label_ext: str,
             prefix: str, start: int, zfill: int, dry_run: bool, map_out: Path | None):
    pairs, missing_labels, missing_images = collect_pairs(images_dir, labels_dir, label_ext)
    print(f"Found {len(pairs)} matching pairs.")
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images without labels (skipped). Examples: {missing_labels[:5]}")
    if missing_images:
        print(f"WARNING: {len(missing_images)} labels without images (skipped). Examples: {missing_images[:5]}")

    new_bases = build_new_names(len(pairs), prefix, start, zfill)
    mapping_rows: List[Tuple[Path, Path, Path, Path]] = []
    renames: List[Tuple[Path, Path]] = []

    for (stem, img_p, lab_p), base in zip(pairs, new_bases):
        new_img = img_p.with_name(base + img_p.suffix.lower())
        new_lab = lab_p.with_name(base + label_ext.lower())
        mapping_rows.append((img_p, lab_p, new_img, new_lab))
        renames.append((img_p, new_img))
        renames.append((lab_p, new_lab))

    # Execute two-phase rename
    two_phase_rename(renames, dry_run=dry_run)

    # Write mapping CSV
    if map_out:
        if map_out.exists() and not dry_run:
            print(f"Overwriting mapping CSV: {map_out}")
        print(f"Writing mapping CSV to: {map_out}")
        if not dry_run:
            write_mapping_csv(map_out, mapping_rows)

    print("\nDONE.")
    if dry_run:
        print("This was a DRY RUN. Use --apply to perform changes.")

def do_undo_from(csv_path: Path, dry_run: bool):
    rows = read_mapping_csv(csv_path)
    if not rows:
        print("Mapping CSV is empty. Nothing to undo.")
        return

    # Build reverse rename list (new -> old)
    renames: List[Tuple[Path, Path]] = []
    for (old_img, old_lab, new_img, new_lab) in rows:
        # Only rename back if current file matches 'new_*'
        if new_img.exists():
            renames.append((new_img, old_img))
        if new_lab.exists():
            renames.append((new_lab, old_lab))

    if not renames:
        print("Nothing to undo. No 'new_*' files found.")
        return

    print(f"Undoing {len(renames)} renames based on mapping CSV...")
    two_phase_rename(renames, dry_run=dry_run)
    print("\nUNDO complete.")
    if dry_run:
        print("This was a DRY RUN. Use --apply to perform changes.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename paired YOLO image/label files in-place.")
    p.add_argument("--images", type=Path, help="Images folder", required=False)
    p.add_argument("--labels", type=Path, help="Labels folder", required=False)
    p.add_argument("--label-ext", default=".txt", help="Label file extension (default: .txt)")
    p.add_argument("--prefix", default="img", help="New filename prefix (default: img)")
    p.add_argument("--start", type=int, default=1, help="Starting index (default: 1)")
    p.add_argument("--zfill", type=int, default=0, help="Zero padding width; 0 = auto (default)")
    p.add_argument("--map-out", type=Path, help="Write mapping CSV to this path")
    p.add_argument("--apply", action="store_true", help="Actually perform renames (default is dry-run)")
    p.add_argument("--undo-from", dest="undo_from", type=Path, help="Undo renames from a mapping CSV")
    args = p.parse_args()

    # validate
    if args.undo_from:
        return args
    if not args.images or not args.labels:
        p.error("--images and --labels are required unless --undo-from is used.")
    if not args.images.is_dir() or not args.labels.is_dir():
        p.error("Both --images and --labels must be existing directories.")
    if not args.label_ext.startswith("."):
        args.label_ext = "." + args.label_ext
    return args

def main():
    args = parse_args()
    if args.undo_from:
        do_undo_from(args.undo_from, dry_run=(not args.apply))
        return

    do_apply(
        images_dir=args.images,
        labels_dir=args.labels,
        label_ext=args.label_ext,
        prefix=args.prefix,
        start=args.start,
        zfill=args.zfill,
        dry_run=(not args.apply),
        map_out=args.map_out,
    )

if __name__ == "__main__":
    main()