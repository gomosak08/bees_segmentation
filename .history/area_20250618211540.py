import os
from functions import *

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
REAL_SQUARE_AREA = 100.0  # cm²

folder_path = '/home/gomosak/abejas/abejas/valid'
images_subdir = 'images'
labels_subdir = 'labels'

images_path = os.path.join(folder_path, images_subdir)
labels_path = os.path.join(folder_path, labels_subdir)

#image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

polygons = []

for file_name in label_files:
    full_path = os.path.join(labels_path, file_name)
    with open(full_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == 0:  # Only take polygons
                coords = list(map(float, parts[1:]))
                polygon = list(zip(coords[::2], coords[1::2]))  # (x, y) pairs
                polygons.append(polygon)

    # Show result
    for i, poly in enumerate(polygons):
        print(f"Polygon {i+1} with {len(poly)} points:")
        print(poly)
    






