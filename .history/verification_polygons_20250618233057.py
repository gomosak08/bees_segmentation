import cv2
import numpy as np
import os

# === CONFIG ===
original_image_path = '/home/gomosak/abejas/abejas/test/images/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.jpg'
original_label_path = '/home/gomosak/abejas/abejas/test/labels/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.txt'
cropped_image_path = '/home/gomosak/abejas/abejas_segmentation/test/images/IMG_20250329_103241568_HDR_jpg.rf.436941d18b41509c0299d480430b32c2.jpg'

# === LOAD ORIGINAL IMAGE ===
image_orig = cv2.imread(original_image_path)
orig_h, orig_w = image_orig.shape[:2]

# === LOAD ORIGINAL LABELS ===
with open(original_label_path, 'r') as f:
    labels = [line.strip().split() for line in f if line.strip()]

# === FIND MARCO BOX ===
marco_box = None
polygons = []
for label in labels:
    cls = label[0]
    coords = list(map(float, label[1:]))
    if cls == '1' and len(coords) >= 8:  # allow more than 4 points for marco
        marco_box = coords
    elif cls == '0' and len(coords) >= 6:
        polygons.append(coords)

if marco_box is None:
    raise ValueError('Marco class not found')

# === GET CROP BOX (marco bounding box) ===
marco_points = [(coords[i], coords[i+1]) for i in range(0, len(marco_box), 2)]
marco_px = [(int(x * orig_w), int(y * orig_h)) for x, y in marco_points]
x_coords = [p[0] for p in marco_px]
y_coords = [p[1] for p in marco_px]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)
crop_w, crop_h = x_max - x_min, y_max - y_min

# === LOAD CROPPED IMAGE ===
cropped_image = cv2.imread(cropped_image_path)
if cropped_image is None:
    raise FileNotFoundError(f"Cropped image not found: {cropped_image_path}")

# === DRAW POLYGONS ON CROPPED IMAGE ===
crop_display = cropped_image.copy()
crop_disp_h, crop_disp_w = crop_display.shape[:2]

for poly in polygons:
    abs_points = [(float(poly[i]) * orig_w, float(poly[i + 1]) * orig_h)
                  for i in range(0, len(poly), 2)]
    crop_relative = [(x - x_min, y - y_min) for x, y in abs_points]
    norm_crop = [(x / crop_w, y / crop_h) for x, y in crop_relative]
    draw_pts = [(int(x * crop_disp_w), int(y * crop_disp_h)) for x, y in norm_crop]
    cv2.polylines(crop_display, [np.array(draw_pts, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

# === SHOW SIDE BY SIDE ===
image_orig_disp = cv2.resize(image_orig, (crop_disp_w, crop_disp_h))
compare = np.hstack((image_orig_disp, crop_display))
cv2.imshow('Original vs Cropped with Polygons', compare)
cv2.waitKey(0)
cv2.destroyAllWindows()
