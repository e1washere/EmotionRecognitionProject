import os
import shutil
from PIL import Image, ImageEnhance

from ResizeSet import resize_set


original_directory_path = './Mixed/test'
new_directory_path = './Mixed/test_resized'

resize_set(original_directory_path, new_directory_path, (96, 96))
