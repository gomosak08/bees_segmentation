#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os

import cv2




IMG_EXTS_DEFAULT = (".jpg", ".jpeg", ".png")

def get_files(dir: str) -> None:
    
    images = sorted([f for f in os.listdir(dir) if f.lower().endswith(IMG_EXTS_DEFAULT)])
    labels = sorted([f for f in os.listdir(dir+str("/labels"))])
     
    #print(images)
    #print(labels)

    for i,j in zip(images,labels):
        print(f'image {i} label {j}')
        # Leer imagen (por defecto en BGR)
        img = cv2.imread(f'{dir}/{i}')

        # Dimensiones
        print("Shape:", img.shape) 
get_files("/home/gomosak/abejas/outputs/test_predict")