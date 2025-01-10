import os
import shutil
from PIL import Image, ImageEnhance

from ResizeSet import resize_set


original_directory_path = './KDEF/KDEF_test'
new_directory_path = './KDEF/KDEF_test_resized'

resize_set(original_directory_path, new_directory_path, (224, 224))
