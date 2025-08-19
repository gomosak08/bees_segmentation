import os
from functions import *

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
REAL_SQUARE_AREA = 1161  # cm²

folder_path = '/home/gomosak/abejas/abejas_segmentation/val/images'
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
        polygons, square = parse_yolo_file(full_path)
        print(f"Found {len(polygons)} polygons")
        if square:
            print("Reference square found")
        else:
            print("No square found")

        if square:
            for i, poly in enumerate(polygons, 1):
                area = convert_to_real_area(poly, square, IMAGE_WIDTH, IMAGE_HEIGHT, REAL_SQUARE_AREA)
                print(f"Polygon {i}: {area:.2f} cm²")



