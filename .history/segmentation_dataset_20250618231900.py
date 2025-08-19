import os
import cv2

# === CONFIG ===
source_base = '/home/gomosak/abejas/abejas/'
output_base = '/home/gomosak/abejas/abejas_segmentation/'
splits = ['test', 'train', 'valid']
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

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
        labels = load_label_file(lbl_path)

        # === Find marco (class 1) ===
        marco_box = None
        for label in labels:
            if label[0] == '1':
                _, x, y, w, h = map(float, label)
                x1 = int((x - w/2) * IMAGE_WIDTH)
                y1 = int((y - h/2) * IMAGE_HEIGHT)
                x2 = int((x + w/2) * IMAGE_WIDTH)
                y2 = int((y + h/2) * IMAGE_HEIGHT)
                marco_box = (x1, y1, x2, y2)
                break

        if marco_box is None:
            print(f"⚠️ No marco in {file}, skipping.")
            continue

        # === Crop image ===
        x1, y1, x2, y2 = marco_box
        crop = img[y1:y2, x1:x2]
        crop_w = x2 - x1
        crop_h = y2 - y1

        updated_labels = []

        for label in labels:
            if label[0] != '0':
                continue  # skip non-polygon

            coords = list(map(float, label[1:]))
            poly_px = normalized_to_pixel(coords, IMAGE_WIDTH, IMAGE_HEIGHT)

            # Adjust to crop
            shifted = [(x - x1, y - y1) for x, y in poly_px]

            # Filter out if any point is outside crop
            if any(x < 0 or y < 0 or x > crop_w or y > crop_h for x, y in shifted):
                continue

            norm_coords = pixel_to_normalized(shifted, crop_w, crop_h)
            updated_labels.append(['0'] + [f'{v:.6f}' for v in norm_coords])

        if updated_labels:
            cv2.imwrite(os.path.join(output_images, file), crop)
            save_label_file(os.path.join(output_labels, base + '.txt'), updated_labels)
            print(f"✅ Processed: {split}/{file}")
        else:
            print(f"⚠️ No valid polygons in: {split}/{file}")
