import os
import cv2

# === RUTAS (las que me diste) ===
source_base = '/home/gomosak/abejas/abejas'                 # contiene test/train/valid con images y labels
output_base = '/home/gomosak/abejas/abejas_segmentation'    # salida en la misma estructura
splits = ['test', 'train', 'valid']  # usa 'valid' (no 'val') como en tus carpetas

IMG_EXTS = ('.jpg', '.jpeg', '.png')

def load_label_file(path):
    with open(path, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def save_label_file(path, entries):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for entry in entries:
            f.write(' '.join(map(str, entry)) + '\n')

def normalized_to_pixel(coords, img_w, img_h):
    # coords=[x1,y1,x2,y2,...] normalizados [0,1]
    return [(float(x)*img_w, float(y)*img_h) for x, y in zip(coords[::2], coords[1::2])]

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

# --- Clip de polígono a rect [0,w]x[0,h] (Sutherland–Hodgman) ---
def clip_polygon_to_rect(poly, w, h):
    def inside(p, edge):
        x, y = p
        if edge == "left":   return x >= 0
        if edge == "right":  return x <= w
        if edge == "top":    return y >= 0
        if edge == "bottom": return y <= h
        return True

    def intersect(p1, p2, edge):
        x1, y1 = p1; x2, y2 = p2
        if (x1, y1) == (x2, y2):
            return p1
        if edge in ("left", "right"):
            x_edge = 0 if edge == "left" else w
            if x2 != x1:
                t = (x_edge - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                return (x_edge, y)
            return (x1, y1)
        else:
            y_edge = 0 if edge == "top" else h
            if y2 != y1:
                t = (y_edge - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                return (x, y_edge)
            return (x1, y1)

    out = poly[:]
    for edge in ["left", "right", "top", "bottom"]:
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

# === Procesamiento por split ===
for split in splits:
    input_images = os.path.join(source_base, split, 'images')
    input_labels = os.path.join(source_base, split, 'labels')
    output_images = os.path.join(output_base, split, 'images')
    output_labels = os.path.join(output_base, split, 'labels')

    # Mantengo tus mkdirs tal cual:
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)

    for file in os.listdir(input_images):
        if not file.lower().endswith(IMG_EXTS):
            continue

        base, ext = os.path.splitext(file)
        img_path = os.path.join(input_images, file)
        lbl_path = os.path.join(input_labels, base + '.txt')
        img_dst  = os.path.join(output_images, file)
        lbl_dst  = os.path.join(output_labels, base + '.txt')

        if not os.path.exists(lbl_path):
            # si no hay label, copio tal cual para no perder la imagen
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(img_dst, img)
            # y genero label vacío
            save_label_file(lbl_dst, [])
            print(f"ℹ️ {split}/{file}: sin label, copiado sin recorte.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ No pude leer imagen: {split}/{file}")
            continue

        img_h, img_w = img.shape[:2]
        labels = load_label_file(lbl_path)

        # 1) buscar "marco" (clase 1)
        marco_box = None
        for lab in labels:
            if lab[0] == '1' and len(lab) >= 6:
                coords = list(map(float, lab[1:]))
                poly_px = normalized_to_pixel(coords, img_w, img_h)
