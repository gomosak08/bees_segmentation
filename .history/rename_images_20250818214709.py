#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_rename_images.py
-----------------------
Rename all image files in ONE folder sequentially without moving them.

Examples:
  # Preview (default): shows planned renames without changing files
  python simple_rename_images.py /path/to/images --prefix bee --start 1 --zfill 5

  # Apply changes
  python rename_images.py /path/to/images --prefix bee --start 1 --zfill 5 --apply
"""

from __future__ import annotations
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple in-place image renamer for a single folder.")
    p.add_argument("folder", type=Path, help="Folder containing images")
    p.add_argument("--prefix", default="img", help="New filename prefix (default: img)")
    p.add_argument("--start", type=int, default=1, help="Starting index (default: 1)")
    p.add_argument("--zfill", type=int, default=4, help="Zero padding width (default: 4)")
    p.add_argument("--apply", action="store_true", help="Actually perform changes (default: preview only)")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Not a directory: {args.folder}")

    files = sorted([p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    if not files:
        print("No image files found.")
        return

    # Build target names and check for collisions
    targets = []
    for i, p in enumerate(files, start=args.start):
        new_name = f"{args.prefix}_id1_id_2_{str(i).zfill(args.zfill)}{p.suffix.lower()}"
        dst = p.with_name(new_name)
        targets.append((p, dst))

    # Warn if any destination already exists and is different from source
    collisions = [dst for src, dst in targets if dst.exists() and dst != src]
    if collisions:
        print("WARNING: Some target files already exist. Those will be skipped:")
        for c in collisions[:10]:
            print(" -", c.name)
        print("Proceeding will skip these colliding files.")

    # Preview / Apply
    for src, dst in targets:
        if dst == src:
            # already the desired name
            continue
        if dst.exists():
            print(f"[SKIP ] {src.name} -> {dst.name} (target exists)")
            continue
        print(f"[RENAME] {src.name} -> {dst.name}")
        if args.apply:
            src.rename(dst)

    if not args.apply:
        print("\nThis was a PREVIEW. Add --apply to perform the renames.")

if __name__ == "__main__":
    main()