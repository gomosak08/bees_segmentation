# Bees Project — Segmentation with Ultralytics YOLO

This README gathers **everything** you need to set up the environment, split the dataset, train, validate, and run **segmentation** with Ultralytics YOLO. It also documents utility scripts and an **interactive script** for training/prediction.

---

## Table of Contents

1. [Requirements & Environment](#1-requirements--environment)
2. [Data Structure & Label Format](#2-data-structure--label-format)
3. [Split the Dataset (train/valid/test)](#3-split-the-dataset-trainvalidtest)
4. [Interactive Training & Prediction Script](#4-interactive-training--prediction-script)
5. [Training & Validation](#5-training--validation)
6. [Testing and Saving Images](#6-testing-and-saving-images)
7. [Visualize GT vs Pred in Jupyter (overlay + quick IoU)](#7-visualize-gt-vs-pred-in-jupyter-overlay--quick-iou)
8. [Polygon Area Calculation](#8-polygon-area-calculation)
9. [Visual Verification of Crops & Polygons](#9-visual-verification-of-crops--polygons)
10. [Best Practices](#10-best-practices)
11. [Quick Commands](#11-quick-commands)

---

## 1) Requirements & Environment

### Activate the virtual environment
```bash
# Handy alias (if it exists in your shell)
act bb

# If the alias doesn't exist, activate manually:
source /home/gomosak/ambientes/bb/bin/activate
```

### (Optional) Install JupyterLab and register a kernel
> Useful for experimentation from notebooks.
```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name=bb --display-name "Python (bb)"
jupyter lab
```

### Install core dependencies
```bash
pip install ultralytics opencv-python matplotlib numpy
```

> **Note**: For segmentation, prefer `*-seg.pt` weights (e.g., `yolo11n-seg.pt`).


---

## 2) Data Structure & Label Format

Expected layout (example):
```
abejas_segmentation/
 ├── train/
 │   ├── images/
 │   └── labels/
 ├── valid/
 │   ├── images/
 │   └── labels/
 └── test/
     ├── images/
     └── labels/
```

- **Labels** are in **YOLO-seg** (segmentation) format:  
  One line per **instance** in the image:
  ```
  <class> x1 y1 x2 y2 ... xN yN
  ```
  where `x, y` are **normalized** to `[0,1]` relative to width/height.

- This project has **a single foreground class** (everything else is **background**).  
  Define only that class in `data.yaml`:
  ```yaml
  path: /home/gomosak/abejas/abejas_segmentation
  train: train/images
  val: valid/images
  test: test/images
  names:
    0: object
  ```

---

## 3) Split the Dataset (train/valid/test)

Script: `segmentation_dataset.py`

### Generic usage
```bash
python segmentation_dataset.py \
  --images-dir /path/to/images \
  --labels-dir /path/to/labels \
  --output-base output/path \
  --train 0.7 --valid 0.2 --test 0.1 \
  --frame-class 1 --target-class 0 \
  --seed 42 --log-level INFO
```

### Example with project paths
```bash
python segmentation_dataset.py \
  --images-dir /home/gomosak/abejas/abejas/images \
  --labels-dir /home/gomosak/abejas/abejas/labels \
  --output-base /home/gomosak/abejas/abejas_segmentation \
  --train 0.7 --valid 0.2 --test 0.1 \
  --frame-class 1 --target-class 0 \
  --seed 42 --log-level INFO
```

**What it does**:
- Reads images from `abejas/images/` and labels from `abejas/labels/`.
- Crops each image by a **frame (class 1)** and reprojects/clips **polygons (class 0)**.
- Splits into `train/valid/test` with the given proportions.
- Ensures image/label filename pairing is consistent.

---

## 4) Interactive Training & Prediction Script

Suggested file: `yolo_interactive.py`  
An **interactive** console workflow for **TRAIN** or **PREDICT** in segmentation.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive YOLO pipeline script.

- Asks whether to TRAIN or PREDICT
- Prompts for model, dataset, and output paths
- Runs the chosen task using Ultralytics YOLO interface

Requirements:
    pip install ultralytics
"""

import os
from ultralytics import YOLO

def ask(msg: str, default: str | None = None) -> str:
    """Prompt with optional default."""
    if default:
        msg = f"{msg} [{default}]: "
    else:
        msg = f"{msg}: "
    val = input(msg).strip()
    return val if val else (default or "")

def do_train():
    print("\n=== TRAIN MODE ===")
    model_path = ask("Path to model (.pt)", "yolo11n-seg.pt")
    data_yaml = ask("Path to dataset YAML", "/home/gomosak/abejas/abejas_segmentation/data.yaml")
    epochs = int(ask("Number of epochs", "100"))
    imgsz = int(ask("Image size", "640"))

    print("\nStarting training...")
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz
    )
    print("\n✅ Training finished.")

def do_predict():
    print("\n=== PREDICT MODE ===")
    model_path = ask("Path to trained model (.pt)", "/home/gomosak/abejas/runs/segment/train/weights/best.pt")
    source = ask("Source directory or file", "/home/gomosak/abejas/abejas_segmentation/test/images")
    project = ask("Project output folder", "/home/gomosak/abejas/outputs")
    name = ask("Experiment name", "test_predict")
    conf = float(ask("Confidence threshold", "0.25"))
    iou = float(ask("IoU threshold", "0.7"))

    print("\nStarting prediction...")
    model = YOLO(model_path)
    model.predict(
        source=source,
        task="segment",
        save=True,
        conf=conf,
        iou=iou,
        project=project,
        name=name,
        exist_ok=True
    )
    print("\n✅ Prediction finished. Results saved in:", os.path.join(project, name))

def main():
    print("YOLO Interactive Script")
    choice = ask("Do you want to [train] or [predict]?", "train").lower()

    if choice.startswith("t"):
        do_train()
    elif choice.startswith("p"):
        do_predict()
    else:
        print("❌ Invalid choice. Please run again and select 'train' or 'predict'.")

if __name__ == "__main__":
    main()
```

### Requirements
```bash
act bb   # or: source /home/gomosak/ambientes/bb/bin/activate
pip install ultralytics
```

### Run
```bash
python yolo_interactive.py
```

---

## 5) Training & Validation

### Train (CLI)
```bash
yolo train task=segment \
  model=yolo11n-seg.pt \
  data=/home/gomosak/abejas/abejas_segmentation/data.yaml \
  epochs=100 imgsz=640
```

### Train (Python/Jupyter)
```python
from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")
model.train(data="/home/gomosak/abejas/abejas_segmentation/data.yaml",
            epochs=100, imgsz=640)
```

### Validate (uses the `val` split from `data.yaml` by default)
```bash
yolo val task=segment \
  model=/home/gomosak/abejas/runs/segment/train/weights/best.pt \
  data=/home/gomosak/abejas/abejas_segmentation/data.yaml \
  imgsz=640 save=True save_txt=True save_json=True \
  conf=0.25 iou=0.7
```

Outputs: `runs/segment/val*` (metrics, curves, `val_batch*_pred.jpg` mosaics, prediction `.txt` in `labels/`).

---

## 6) Testing and Saving Images

### Metrics on **test** (uses GT from `labels/test`)
```bash
yolo val task=segment \
  model=/home/gomosak/abejas/runs/segment/train/weights/best.pt \
  data=/home/gomosak/abejas/abejas_segmentation/data.yaml \
  split=test imgsz=640 \
  save=True save_txt=True save_json=True \
  conf=0.25 iou=0.7
```

> `val` on test saves **batch mosaics** + `.txt` predictions and computes metrics.

### **Per-image** outputs (prediction/visualization)
To generate **one output image per test file**, use `predict`:

```bash
yolo predict task=segment \
  model=/home/gomosak/abejas/runs/segment/train/weights/best.pt \
  source=/home/gomosak/abejas/abejas_segmentation/test/images \
  save=True conf=0.25 iou=0.7 \
  project=/home/gomosak/abejas/outputs name=test_predict exist_ok=True
```

Images are saved at:  
`/home/gomosak/abejas/outputs/test_predict/`

---

## 7) Visualize GT vs Pred in Jupyter (overlay + quick IoU)

```python
from pathlib import Path
import numpy as np, cv2, matplotlib.pyplot as plt

def yolo_poly_to_abs(xy_norm, w, h):
    xy = np.array(xy_norm, dtype=np.float32).reshape(-1, 2)
    return np.stack([xy[:,0]*w, xy[:,1]*h], axis=1).astype(np.int32)

def load_yolo_poly_txt(txt_file):
    polys = []
    p = Path(txt_file)
    if not p.exists(): return polys
    for line in p.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3: continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0: continue
        polys.append((cls, coords))
    return polys

def rasterize_mask(shape, polys_abs):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for pa in polys_abs:
        cv2.fillPoly(mask, [pa], 1)
    return mask

DATA = Path("/home/gomosak/abejas/abejas_segmentation")
img_dir = DATA/"test/images"
gt_dir  = DATA/"test/labels"
# Adjust to the latest val*
val_dir = sorted((Path("/home/gomosak/abejas/runs/segment")).glob("val*"), key=lambda p: p.stat().st_mtime)[-1]
pred_dir = val_dir / "labels"

img_path = sorted(img_dir.glob("*.*"))[0]  # change index to preview another image
stem = img_path.stem
gt_txt, pred_txt = gt_dir/f"{stem}.txt", pred_dir/f"{stem}.txt"

img = cv2.imread(str(img_path))[:, :, ::-1]
h, w = img.shape[:2]

# Load GT and Pred polygons
gt_polys = [yolo_poly_to_abs(coords, w, h) for _, coords in load_yolo_poly_txt(gt_txt)]
pr_polys = [yolo_poly_to_abs(coords, w, h) for _, coords in load_yolo_poly_txt(pred_txt)]

# Overlay
vis = img.copy()
for pa in gt_polys:
    cv2.polylines(vis, [pa], True, (0,255,0), 2)
    cv2.fillPoly(vis, [pa], (0,255,0))
for pa in pr_polys:
    cv2.polylines(vis, [pa], True, (255,0,0), 2)

plt.figure(figsize=(8,8)); plt.title(f"GT (green) vs Pred (red): {stem}")
plt.imshow(vis); plt.axis('off'); plt.show()

# Global mask IoU
gt_mask = rasterize_mask(img, gt_polys)
pr_mask = rasterize_mask(img, pr_polys)
inter = np.logical_and(gt_mask==1, pr_mask==1).sum()
union = np.logical_or(gt_mask==1, pr_mask==1).sum()
iou = inter / union if union > 0 else 0.0
print(f"Mask IoU (global): {iou:.3f}")
```

---

## 8) Polygon Area Calculation

Script: `area.py`  
Converts normalized coordinates to pixels and computes polygon areas (e.g., Shoelace formula).

```bash
python area.py
```

---

## 9) Visual Verification of Crops & Polygons

Script: `verification_polygons.py`  
Shows the original image with annotations and the crop by **frame (class 1)** with reprojected polygons. In headless environments, it can save verification images to disk.

```bash
python verification_polygons.py
```

---

## 10) Best Practices

- Keep **filenames** consistent between images and labels (`img.jpg` ↔ `img.txt`).
- Coordinates always **normalized** to `[0,1]` in YOLO-seg labels.
- One **line per instance** in each `.txt`.
- For single-class segmentation, do **not** define "background" as a class; everything not annotated is BG.
- Control environment versions (use your venv `bb`) and set **seeds** if you need reproducibility.
- If training from an architecture `.yaml` (from scratch), expect more epochs to converge.

---

## 11) Quick Commands

```bash
# Activate env
act bb

# Train
yolo train task=segment model=yolo11n-seg.pt \
  data=/home/gomosak/abejas/abejas_segmentation/data.yaml \
  epochs=100 imgsz=640

# Validate on val (default) or test (split=test)
yolo val task=segment \
  model=/home/gomosak/abejas/runs/segment/train/weights/best.pt \
  data=/home/gomosak/abejas/abejas_segmentation/data.yaml \
  split=test imgsz=640 save=True save_txt=True save_json=True conf=0.25 iou=0.7

# Prediction (one output image per input)
yolo predict task=segment \
  model=/home/gomosak/abejas/runs/segment/train/weights/best.pt \
  source=/home/gomosak/abejas/abejas_segmentation/test/images \
  save=True conf=0.25 iou=0.7 \
  project=/home/gomosak/abejas/outputs name=test_predict exist_ok=True
```

---

> **Notes**: If you change folder structure or names, update paths in `data.yaml` and the commands accordingly.