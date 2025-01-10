import os
import shutil
from PIL import Image, ImageEnhance

from ResizeSet import resize_set


original_directory_path = './Natural_Human_Face_Images/NHFI_test'
new_directory_path = './Natural_Human_Face_Images/NHFI_test_resized'

resize_set(original_directory_path, new_directory_path, (224, 224))
