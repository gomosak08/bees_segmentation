import os
import cv2
import numpy as np

# === CONFIG ===
original_image_path = '/home/gomosak/abejas/abejas/test/images/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.jpg'
original_label_path = '/home/gomosak/abejas/abejas/test/labels/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.txt'
cropped_image_path = '/home/gomosak/abejas/abejas_segmentation/test/images/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.jpg'

# === LOAD ORIGINAL IMAGE ===
image_orig = cv2.imread(original_image_path)
if image_orig is None:
    raise FileNotFoundError(f"Original image not found: {original_image_path}")
orig_h, orig_w = image_orig.shape[:2]

# === LOAD LABELS ===
with open(original_label_path, 'r') as f:
    labels = [line.strip().split() for line in f if line.strip()]

# === PARSE LABELS ===
marco_box = None
polygons = []

for label in labels:
    cls = label[0]
    coords = list(map(float, label[1:]))
    if cls == '1':
        marco_box = coords
    elif cls == '0' and len(coords) >= 6:
        polygons.append(coords)

if marco_box is None:
    raise ValueError('Marco class not found')

# === GET MARCO BBOX COORDINATES ===
marco_points = [(marco_box[i], marco_box[i+1]) for i in range(0, len(marco_box), 2)]
marco_abs = [(int(x * orig_w), int(y * orig_h)) for x, y in marco_points]
x_coords, y_coords = zip(*marco_abs)
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)
crop_w, crop_h = x_max - x_min, y_max - y_min

# === LOAD CROPPED IMAGE ===
crop_img = cv2.imread(cropped_image_path)
if crop_img is None:
    raise FileNotFoundError(f"Cropped image not found: {cropped_image_path}")
crop_disp_h, crop_disp_w = crop_img.shape[:2]

# === DRAW POLYGONS ON CROPPED IMAGE ===
crop_draw = crop_img.copy()
for poly in polygons:
    abs_pts = [(float(poly[i]) * orig_w, float(poly[i + 1]) * orig_h) for i in range(0, len(poly), 2)]
    crop_rel = [(x - x_min, y - y_min) for x, y in abs_pts]
    norm_crop = [(x / crop_w, y / crop_h) for x, y in crop_rel]
    disp_pts = [(int(x * crop_disp_w), int(y * crop_disp_h)) for x, y in norm_crop]
    cv2.polylines(crop_draw, [np.array(disp_pts, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

# === DRAW POLYGONS ON ORIGINAL IMAGE ===
original_draw = image_orig.copy()
for poly in polygons:
    abs_pts = [(int(float(poly[i]) * orig_w), int(float(poly[i + 1]) * orig_h)) for i in range(0, len(poly), 2)]
    cv2.polylines(original_draw, [np.array(abs_pts, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

# === DISPLAY COMPARISON ===
resized_original = cv2.resize(original_draw, (crop_disp_w, crop_disp_h))
compare = np.hstack((resized_original, crop_draw))
cv2.imshow('Original vs Cropped (Green=Crop Polygon, Blue=Orig)', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()
