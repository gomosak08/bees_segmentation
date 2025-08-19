import os
import cv2
import numpy as np

# === CONFIG ===
base_dir = '/home/gomosak/abejas/abejas_segmentation/valid'  # <- tu ruta
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')

IMG_EXTS = ('.jpg', '.jpeg', '.png')
MAX_W, MAX_H = 1600, 1000  # tamaño máximo de visualización

def load_label_file(path):
    with open(path, 'r') as f:
        return [line.strip().split() for line in f if line.strip()]

def normalized_to_pixel(coords, img_w, img_h):
    # coords: [x1,y1,x2,y2,...] en [0,1]
    pts = []
    it = iter(coords)
    for x, y in zip(it, it):
        px = float(x) * img_w
        py = float(y) * img_h
        pts.append((px, py))
    return pts

def ensure_bgr(img):
    # Convierte 1 canal o 4 canales a BGR
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def scale_to_fit(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

# recopilar lista
files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
files.sort()
if not files:
    print(f'No hay imágenes en: {images_dir}')
    raise SystemExit

idx = 0
cv2.namedWindow("Visualización", cv2.WINDOW_NORMAL)  # ventana redimensionable
cv2.resizeWindow("Visualización", 1200, 800)

def draw_overlay(img_bgr, labels, show_text=True):
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    any_drawn = False

    for lab in labels:
        cls = lab[0]
        coords = list(map(float, lab[1:]))
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        # validación básica: normalizados en [0,1]
        if not all(0.0 <= c <= 1.0 for c in coords):
            # si alguna coord está fuera, solo avisa y continúa
            # (no transformamos para no dibujar basura)
            # Puedes optar por clamp a [0,1] si lo deseas
            continue

        pts = normalized_to_pixel(coords, w, h)
        pts_i = np.array(pts, np.int32).reshape((-1, 1, 2))

        # color por clase
        color = (0, 255, 0) if cls == '0' else (255, 0, 0) if cls == '1' else (0, 0, 255)

        # polilínea
        cv2.polylines(overlay, [pts_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # relleno semitransparente para ver mejor
        mask = np.zeros_like(img_bgr)
        cv2.fillPoly(mask, [pts_i], color)
        overlay = cv2.addWeighted(overlay, 1.0, mask, 0.15, 0.0)

        if show_text:
            # etiqueta cerca del primer punto
            x0, y0 = pts_i[0,0]
            cv2.putText(overlay, f'class {cls}', (x0+4, y0-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        any_drawn = True

    return overlay, any_drawn

while True:
    file = files[idx]
    base, _ = os.path.splitext(file)
    img_path = os.path.join(images_dir, file)
    lbl_path = os.path.join(labels_dir, base + '.txt')

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        disp = np.zeros((400, 800, 3), np.uint8)
        cv2.putText(disp, f'No pude leer: {file}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        img = ensure_bgr(img)

        # debug opcional: detectar imágenes en negro
        if img.dtype == np.uint8 and np.max(img) == 0:
            print(f'⚠️ Imagen negra (todo 0): {file} — revisa el recorte/al origen.')

        if os.path.exists(lbl_path):
            labels = load_label_file(lbl_path)
        else:
            labels = []
            print(f'⚠️ Sin label: {file}')

        drawn, any_drawn = draw_overlay(img, labels)
        disp = scale_to_fit(drawn)  # escalar para que quepa en pantalla

    cv2.imshow("Visualización", disp)
    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key in (81, ord('a')):  # flecha izquierda o 'a'
        idx = (idx - 1) % len(files)
    elif key in (83, ord('d')):  # flecha derecha o 'd'
        idx = (idx + 1) % len(files)
    else:
        # cualquier otra tecla: siguiente
        idx = (idx + 1) % len(files)

cv2.destroyAllWindows()
