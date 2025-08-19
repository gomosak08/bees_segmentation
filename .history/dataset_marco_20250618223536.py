import os

folder_path = '/home/gomosak/abejas/abejas/'
folders = ['test', 'train', 'valid'] 
for i in folders:
    path = f'{folder_path}{i}'
    images_path = os.path.join(folder_path, i,'images')
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

#images_path = os.path.join(folder_path, images_subdir)
#labels_path = os.path.join(folder_path, labels_subdir)