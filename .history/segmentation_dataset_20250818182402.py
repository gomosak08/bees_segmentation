import os
#import cv2

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

    