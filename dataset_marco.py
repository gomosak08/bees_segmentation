import os
import shutil
import yaml

source_base = '/home/gomosak/abejas/abejas/'
output_base = '/home/gomosak/abejas/abejas_marco/'


splits = ['test', 'train', 'valid'] 
class_name = 'marco'



# === SCRIPT ===
for split in splits:
    img_dir = os.path.join(source_base, split, 'images')
    lbl_dir = os.path.join(source_base, split, 'labels')

    out_img_dir = os.path.join(output_base, split, 'images')
    out_lbl_dir = os.path.join(output_base, split, 'labels')

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in os.listdir(img_dir):
        if not file.endswith('.jpg'):
            continue

        base = os.path.splitext(file)[0]
        img_src = os.path.join(img_dir, file)
        lbl_src = os.path.join(lbl_dir, base + '.txt')

        img_dst = os.path.join(out_img_dir, file)
        lbl_dst = os.path.join(out_lbl_dir, base + '.txt')

        # Copy image
        shutil.copy(img_src, img_dst)

        # Copy filtered label (only class 1)
        if os.path.exists(lbl_src):
            with open(lbl_src, 'r') as f:
                lines = f.readlines()

            marco_lines = []
            for line in lines:
                if line.startswith('1 '):
                    parts = line.strip().split()
                    parts[0] = '0'  # reassign class 1 → 0
                    marco_lines.append(' '.join(parts) + '\n')

            with open(lbl_dst, 'w') as f:
                f.writelines(marco_lines)

# === CREATE data.yaml ===
data_yaml = {
    'path': output_base,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 1,
    'names': [class_name]
}

yaml_path = os.path.join(output_base, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"\n✅ Marco dataset prepared at: {output_base}")
print(f"📄 YOLOv8 config written to: {yaml_path}")