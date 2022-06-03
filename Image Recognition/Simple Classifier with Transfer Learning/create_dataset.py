# Notes:
# Folder organisation Main\Train\(class id)\(class imgs)

import os
from keras.preprocessing import image

img_size = (224,224)
folders = os.listdir('Train')
print('[note] class folders')
print(folders)

img_data = []; labels = []; ii=0
for folder in folders:
    path = os.path.join("Train", folder)
    print(path, ii) 
    for im in os.listdir(path):
        try:
            img = image.load_img(os.path.join(path,im),
                                 target_size = img_size)
            img_array = image.img_to_array(img)
            img_data.append(img_array)
            labels.append(ii)
        except:
            pass
    ii+=1
