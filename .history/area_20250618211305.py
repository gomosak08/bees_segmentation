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

for file_name in label_files:
    full_path = os.path.join(labels_path, file_name)
    #print("Label:", full_path)
    with open(full_path, 'r') as file:
        content = file.read()
        print(content)





# Replace this with reading from your .txt annotation file
annotation_lines = [
    # paste your label lines here
]

polygon_coords = []
square_coords = None

# Parse lines
for line in annotation_lines:
    class_id, coords = parse_annotation_line(line)
    if class_id == 1:
        square_coords = coords
    elif class_id == 0:
        polygon_coords.append(coords)

# Calculate area
if square_coords is None:
    print("No reference square found.")
else:
    for idx, poly in enumerate(polygon_coords, 1):
        area = convert_to_real_world_area(poly, square_coords, IMAGE_WIDTH, IMAGE_HEIGHT, REAL_SQUARE_AREA)
        print(f"Polygon {idx}: {area:.2f} real-world units²")