import os
import cv2
import numpy as np

# === CONFIG ===
source_base = '/home/gomosak/abejas/abejas/'
output_base = '/home/gomosak/abejas/abejas_segmentation/'
splits = ['test', 'train', 'valid']

def load_label_file(path):
    with open(path, 'r') as f:
        return [line.strip().split() for line in f.readlines()]

def save_label_file(path, entries):
    with open(path, 'w') as f:
        for entry in entries:
            f.write(' '.join(map(str, entry)) + '\n')

def normalized_to_pixel(coords, img_w, img_h):
    return [(float(x)*img_w, float(y)*img_h) for x, y in zip(coords[::2], coords[1::2])]

def pixel_to_normalized(coords, img_w, img_h):
    out = []
    for x, y in coords:
        out.extend([x / img_w, y / img_h])
    return out

for split in splits:
    input_images = os.path.join(source_base, split, 'images')
    input_labels = os.path.join(source_base, split, 'labels')
    output_images = os.path.join(output_base, split, 'images')
    output_labels = os.path.join(output_base, split, 'labels')

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    for file in os.listdir(input_images):
        if not file.endswith('.jpg'):
            continue

        base = os.path.splitext(file)[0]
        img_path = os.path.join(input_images, file)
        lbl_path = os.path.join(input_labels, base + '.txt')

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        labels = load_label_file(lbl_path)

        marco_box = None
        polygons = []

        for label in labels:
            if label[0] == '1':
                coords = list(map(float, label[1:]))
                poly_px = normalized_to_pixel(coords, img_w, img_h)
                xs, ys = zip(*poly_px)
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                marco_box = (x1, y1, x2, y2)
            elif label[0] == '0':
                polygons.append(list(map(float, label[1:])))

        if marco_box is None:
            print(f"⚠️ No marco in {file}, skipping.")
            continue

        x1, y1, x2, y2 = marco_box
        crop = img[y1:y2, x1:x2]
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            print(f"❌ Invalid crop dimensions in: {split}/{file}")
            continue

        updated_labels = []
        crop_disp = crop.copy()

        for poly in polygons:
            abs_pts = normalized_to_pixel(poly, img_w, img_h)
            shifted = [(x - x1, y - y1) for x, y in abs_pts]

            if any(x < 0 or y < 0 or x > crop_w or y > crop_h for x, y in shifted):
                continue

            norm_coords = pixel_to_normalized(shifted, crop_w, crop_h)
            updated_labels.append(['0'] + [f'{v:.6f}' for v in norm_coords])

            draw_pts = [(int(x), int(y)) for x, y in shifted]
            cv2.polylines(crop_disp, [np.array(draw_pts, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        if updated_labels:
            cv2.imwrite(os.path.join(output_images, file), crop)
            save_label_file(os.path.join(output_labels, base + '.txt'), updated_labels)
            original_disp = img.copy()
            for poly in polygons:
                abs_pts = normalized_to_pixel(poly, img_w, img_h)
                cv2.polylines(original_disp, [np.array(abs_pts, np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            resized_original = cv2.resize(original_disp, (crop_w, crop_h))
            compare = np.hstack((resized_original, crop_disp))
            cv2.imshow(f'Compare: {split}/{file}', compare)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"✅ Processed and verified: {split}/{file}")
        else:
            print(f"⚠️ No valid polygons in: {split}/{file}")
