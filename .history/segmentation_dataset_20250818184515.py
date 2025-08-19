import os
import cv2
import random

# === RUTAS (tus rutas) ===
images_dir = '/home/gomosak/abejas/abejas/images'
labels_dir = '/home/gomosak/abejas/abejas/labels'
output_base = '/home/gomosak/abejas/abejas_segmentation'

# splits (usando 'valid' como en tus carpetas)
splits = ['train', 'valid', 'test']
train_split, valid_split, test_split = 0.7, 0.2, 0.1

# extensiones aceptadas
IMG_EXTS = ('.jpg', '.jpeg', '.png')

# ========== utilidades ==========
def load_label_file(path):
    with open(path, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def save_label_file(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for e in entries:
            f.write(' '.join(map(str, e)) + '\n')

def normalized_to_pixel(coords, img_w, img_h):
    # coords = [x1,y1,x2,y2,...] en [0,1]
    return [(float(x) * img_w, float(y) * img_h) for x, y in zip(coords[::2], coords[1::2])]

def pixel_to_normalized(coords, img_w, img_h):
    out = []
    for x, y in coords:
        out.extend([x / img_w, y / img_h])
    return out

def bbox_from_polygon(poly_px, img_w, img_h):
    xs, ys = zip(*poly_px)
    x1 = int(max(0, min(xs)))
    y1 = int(max(0, min(ys)))
    x2 = int(min(img_w, max(xs)))
    y2 = int(min(img_h, max(ys)))
    return x1, y1, x2, y2

# --- Sutherland–Hodgman: recorte de polígono a rect [0,w]x[0,h] ---
def clip_polygon_to_rect(poly, w, h):
    def inside(p, edge):
        x, y = p
        if edge == 'left':   return x >= 0
        if edge == 'right':  return x <= w
        if edge == 'top':    return y >= 0
        if edge == 'bottom': return y <= h
        return True

    def intersect(p1, p2, edge):
        x1, y1 = p1; x2, y2 = p2
        if (x1, y1) == (x2, y2):
            return p1
        if edge in ('left', 'right'):
            x_edge = 0 if edge == 'left' else w
            if x2 != x1:
                t = (x_edge - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                return (x_edge, y)
            return (x1, y1)
        else:
            y_edge = 0 if edge == 'top' else h
            if y2 != y1:
                t = (y_edge - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                return (x, y_edge)
            return (x1, y1)

    out = poly[:]
    for edge in ['left', 'right', 'top', 'bottom']:
        if not out:
            break
        inp = out
        out = []
        s = inp[-1]
        for e in inp:
            if inside(e, edge):
                if inside(s, edge):
                    out.append(e)
                else:
                    out.append(intersect(s, e, edge))
                    out.append(e)
            else:
                if inside(s, edge):
                    out.append(intersect(s, e, edge))
            s = e
    out = [(float(x), float(y)) for x, y in out if x == x and y == y]
    return out if len(out) >= 3 else []

# ========== preparar lista y dividir ==========
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
# quedarnos solo con las que tienen label .txt
paired = [f for f in all_images if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt'))]

random.shuffle(paired)
n_total = len(paired)
n_train = int(n_total * train_split)
n_valid = int(n_total * valid_split)

files_train = paired[:n_train]
files_valid = paired[n_train:n_train + n_valid]
files_test  = paired[n_train + n_valid:]

split_dict = {
    'train': files_train,
    'valid': files_valid,
    'test':  files_test
}

# crear carpetas como pediste
for split in split_dict:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

processed, skipped_no_marco, skipped_bad = 0, 0, 0

# ========== procesar ==========
for split, files in split_dict.items():
    out_img_dir = os.path.join(output_base, split, 'images')
    out_lbl_dir = os.path.join(output_base, split, 'labels')

    for file in files:
        base, ext = os.path.splitext(file)
        img_path = os.path.join(images_dir, file)
        lbl_path = os.path.join(labels_dir, base + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            print(f'❌ No pude leer imagen: {file}')
            skipped_bad += 1
            continue

        img_h, img_w = img.shape[:2]
        labels = load_label_file(lbl_path)

        # 1) buscar marco (clase 1)
        marco_box = None
        for lab in labels:
            if lab[0] == '1' and len(lab) >= 6:
                coords = list(map(float, lab[1:]))
                poly_px = normalized_to_pixel(coords, img_w, img_h)
                x1, y1, x2, y2 = bbox_from_polygon(poly_px, img_w, img_h)
                if (x2 - x1) > 1 and (y2 - y1) > 1:
                    marco_box = (x1, y1, x2, y2)
                    break

        # Si no hay marco → NO copiar nada (según tu preferencia)
        if marco_box is None:
            print(f'⚠️ {file}: sin marco (clase 1), omitida.')
            skipped_no_marco += 1
            continue

        # 2) recortar
        x1, y1, x2, y2 = marco_box
        crop = img[y1:y2, x1:x2]
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w <= 1 or crop_h <= 1:
            print(f'⚠️ {file}: marco inválido, omitida.')
            skipped_bad += 1
            continue

        # 3) clippear y normalizar polígonos de clase 0
        updated = []
        for lab in labels:
            if lab[0] != '0':
                continue
            coords = list(map(float, lab[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            poly_px = normalized_to_pixel(coords, img_w, img_h)
            shifted = [(x - x1, y - y1) for (x, y) in poly_px]
            clipped = clip_polygon_to_rect(shifted, crop_w, crop_h)
            if not clipped:
                continue
            norm_coords = pixel_to_normalized(clipped, crop_w, crop_h)
            updated.append(['0'] + [f'{v:.6f}' for v in norm_coords])

        # Guardar (si no queda ningún polígono, de todos modos guardamos imagen recortada y label vacío)
        out_img_path = os.path.join(out_img_dir, file)
        out_lbl_path = os.path.join(out_lbl_dir, base + '.txt')
        cv2.imwrite(out_img_path, crop)
        save_label_file(out_lbl_path, updated)
        processed += 1
        print(f'✅ {split}/{file}: recortada al marco y labels actualizados.')

print(f'\n✅ Listo. Procesadas: {processed} | Omitidas sin marco: {skipped_no_marco} | Omitidas por error: {skipped_bad}')
print(f'Estructura: {output_base}/<train|valid|test>/(images|labels)')
