import os
import cv2
import numpy as np

# === CONFIG ===
base_dir = '/home/gomosak/abejas/abejas'  # cambia esta ruta
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

IMG_EXTS = ('.jpg', '.jpeg', '.png')

def load_label_file(path):
    with open(path, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def normalized_to_pixel(coords, img_w, img_h):
    return [(float(x) * img_w, float(y) * img_h) for x, y in zip(coords[::2], coords[1::2])]

# === VISUALIZACIÓN INTERACTIVA ===
for file in os.listdir(images_dir):
    if not file.lower().endswith(IMG_EXTS):
        continue

    base, _ = os.path.splitext(file)
    img_path = os.path.join(images_dir, file)
    lbl_path = os.path.join(labels_dir, base + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ No pude leer: {file}")
        continue

    img_h, img_w = img.shape[:2]

    if os.path.exists(lbl_path):
        labels = load_label_file(lbl_path)
        for lab in labels:
            cls = lab[0]
            coords = list(map(float, lab[1:]))
            pts = normalized_to_pixel(coords, img_w, img_h)
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))

            # Colores: clase 0 = verde, clase 1 = azul, otras = rojo
            color = (0, 255, 0) if cls == '0' else (255, 0, 0) if cls == '1' else (0, 0, 255)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    else:
        print(f"⚠️ No label para {file}")

    cv2.imshow("Visualización", img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        print("👋 Saliendo.")
        break

cv2.destroyAllWindows()
