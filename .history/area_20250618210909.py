import math
import os

folder_path = '/home/gomosak/abejas/abejas/valid'
images = 'images'
labels = 'labels'
files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

for file_name in files:
    full_path = os.path.join(folder_path, file_name)
    print(full_path)