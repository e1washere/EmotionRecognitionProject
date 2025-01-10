import os
import shutil
import random
from PIL import Image

# directory_path = './NHFI_uniform'
# original_directory_path = './test'
# new_directory_path = './test_uniform'

def resize_set(original_directory_path, new_directory_path, size):
    for folder_name in os.listdir(original_directory_path):
        count = 0
        folder_path = os.path.join(original_directory_path, folder_name)
        new_folder_path = os.path.join(new_directory_path, folder_name)

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            resized_image = image.resize(size)

            new_image_path = os.path.join(new_folder_path, filename)
            resized_image.save(new_image_path)

            count += 1
            if count%100 == 0:
                print(f"Changed {count} images")
        
        print(f"Processed {folder_name}")

        