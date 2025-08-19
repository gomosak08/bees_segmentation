#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import cv2
from functions import *


REAL_SQUARE_AREA = 1161
IMG_EXTS_DEFAULT = (".jpg", ".jpeg", ".png")
def polygon_area_px(points):
    pts = np.asarray(points, dtype=np.float32)
    return float(cv2.contourArea(pts))

def get_files(dir: str) -> None:
    
    images = sorted([f for f in os.listdir(dir) if f.lower().endswith(IMG_EXTS_DEFAULT)])
    labels = sorted([f for f in os.listdir(dir+str("/labels"))])
     
    #print(images)
    #print(labels)

    for i,j in zip(images,labels):
        target_class = "0"   
        #print(f'image {i} label {j}')
        # Leer imagen (por defecto en BGR)
        img = cv2.imread(f'{dir}/{i}')

        # Dimensiones
        #print("Shape:", img.shape) 

        H, W = img.shape[:2]
        A_img_px = W * H

        polys = load_yolo_seg_polygons(f'{dir}/{j}')

        A_poly_px = 0.0
        for cls, coords in polys:
            if cls != target_class:
                continue
            pts = norm_to_px(coords, W, H)
            A_poly_px += polygon_area_px(pts)

        coverage = A_poly_px / A_img_px if A_img_px > 0 else 0.0
        A_poly_cm2 = coverage * REAL_SQUARE_AREA

        print(f"Imagen: {i}")
        #print(f"Área imagen: {A_img_cm2:.3f} cm²")
        print(f"Área polígono: {A_poly_cm2:.3f} cm²")
        print(f"Cobertura: {coverage*100:.3f}%")
get_files("/home/gomosak/abejas/outputs/test_predict")