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

# tomar todos los nombres de imágenes
images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)

# dividir dataset
n_total = len(images)
n_train = int(n_total * train_split)
n_val = int(n_total * val_split)

train_files = images[:n_train]
val_files = images[n_train:n_train+n_val]
test_files = images[n_train+n_val:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# mover/copiar archivos
for split, files in splits.items():
    # crear carpetas en la forma que pediste
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    for f in files:
        img_src = os.path.join(images_dir, f)
        label_src = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")

        img_dst = os.path.join(output_dir, split, "images", f)
        label_dst = os.path.join(output_dir, split, "labels", os.path.splitext(f)[0] + ".txt")

        shutil.copy2(img_src, img_dst)   # o cambia a shutil.move si prefieres mover
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)

print("✅ Dataset dividido en train/val/test con imágenes y labels")