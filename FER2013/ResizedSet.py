import os
import shutil
from PIL import Image, ImageEnhance

from ResizeSet import resize_set


original_directory_path = './FER2013/test_uniform'
new_directory_path = './FER2013/test_resized'

resize_set(original_directory_path, new_directory_path, (48, 48))
