import os
import shutil
import random

# directory_path = './NHFI_uniform'
# original_directory_path = './test'
# new_directory_path = './test_uniform'
new_directory_path = './Mixed/test'

def copy_to_mixed_set(original_directory_path, number_of_images):
    for folder_name in os.listdir(original_directory_path):
        if folder_name == 'contempt':
            continue
        folder_path = os.path.join(original_directory_path, folder_name)
        new_folder_path = os.path.join(new_directory_path, folder_name)
        files = os.listdir(folder_path)

        file_paths = [os.path.join(folder_path, f) for f in files]

        random_files = random.sample(file_paths, number_of_images)


        for file_path in random_files:
            shutil.copy(file_path, new_folder_path)
        
        print(f"Processed {folder_name}")

        