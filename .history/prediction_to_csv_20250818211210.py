#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse
import logging
import os
from typing import Generator, Iterable, List, Tuple

IMG_EXTS_DEFAULT = (".jpg", ".jpeg", ".png")

def get_files(dir: str) -> None:
    
    images = sorted([f for f in os.listdir(dir) if f.lower().endswith(IMG_EXTS_DEFAULT)])
    labels = sorted([f for f in os.listdir(dir+str("/labels"))])
     
    #print(images)
    #print(labels)

    for i,j in zip(images,labels):
        print(f'image {i} label {j}')