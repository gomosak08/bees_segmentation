import os

folder_path = '/home/gomosak/abejas/abejas/valid'
images_subdir = 'images'
labels_subdir = 'labels'

images_path = os.path.join(folder_path, images_subdir)
labels_path = os.path.join(folder_path, labels_subdir)

image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

# Print full paths
for file_name in image_files:
    full_path = os.path.join(images_path, file_name)
    print("Image:", full_path)

for file_name in label_files:
    full_path = os.path.join(labels_path, file_name)
    print("Label:", full_path)
