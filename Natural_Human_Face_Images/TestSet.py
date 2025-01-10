import os
import shutil
from PIL import Image 
from CreateTestSet import create_test_set
# directory_path = './NHFI_uniform'
original_directory_path = './Natural_Human_Face_Images/NHFI_uniform'
new_directory_path = './Natural_Human_Face_Images/NHFI_test'

create_test_set(original_directory_path, new_directory_path, 140)


        