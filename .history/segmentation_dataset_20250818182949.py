import os
import shutil
import random
source_base = '/home/gomosak/abejas/abejas/'

# === CONFIG ===
images_dir = f"{source_base}images"
labels_dir = f"{source_base}images"
output_dir = "/home/gomosak/abejas/abejas_segmentation/"

# proporciones
train_split = 0.7
val_split = 0.2
test_split = 0.1

# crear carpetas destino
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)





