import os

folder_path = '/home/gomosak/abejas/abejas/'
folders = ['test', 'train', 'valid'] 
output_base = '/home/gomosak/abejas/abejas_marco'


for i in folders:
    path = f'{folder_path}{i}'
    images_path = os.path.join(folder_path, i,'images')
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    label_path  = os.path.join(folder_path, i,'labels')
    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    print(label_files)

#images_path = os.path.join(folder_path, images_subdir)
#labels_path = os.path.join(folder_path, labels_subdir)