import math
import os

folder_path = '/home/gomosak/abejas/abejas/valid'
images = '/images'
labels = '/labels'
images = [f for f in os.listdir(folder_path+images) if f.endswith('.jpg')]
labels = [f for f in os.listdir(folder_path+images) if f.endswith('.txt')]


for file_name in images:
    full_path = os.path.join(folder_path, file_name)
    print(full_path)

for file_name in labels:
    full_path = os.path.join(folder_path, file_name)
    print(full_path)