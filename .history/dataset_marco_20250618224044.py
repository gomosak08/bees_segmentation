import os

folder_path = '/home/gomosak/abejas/abejas/'
output_base = '/home/gomosak/abejas/abejas_marco/'


splits = ['test', 'train', 'valid'] 


# === SCRIPT ===
for split in splits:
    img_dir = os.path.join(folder_path, split)
    lbl_dir = os.path.join(folder_path, split)

    out_img_dir = os.path.join(output_base, 'images', split)
    out_lbl_dir = os.path.join(output_base, 'labels', split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

#images_path = os.path.join(folder_path, images_subdir)
#labels_path = os.path.join(folder_path, labels_subdir)