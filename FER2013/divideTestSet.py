import os
import shutil
from PIL import Image 

# directory_path = './NHFI_uniform'
original_directory_path = './test'
new_directory_path = './test_uniform'

def divide_test_set():
    for folder_name in os.listdir(original_directory_path):
        folder_path = os.path.join(original_directory_path, folder_name)
        new_folder_path = os.path.join(new_directory_path, folder_name)

        for filename in os.listdir(folder_path):
            if 'PrivateTest' in filename:
                image_path = os.path.join(folder_path, filename)
                new_image_path = os.path.join(new_folder_path, filename)
                shutil.copy(image_path, new_image_path)
        
        print(f"Processed {folder_name}")

divide_test_set()
        