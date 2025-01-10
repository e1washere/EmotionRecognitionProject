import os
import shutil
from PIL import Image 

from RotateImages import rotate_images

directory_path = './Natural_Human_Face_Images/NHFI_uniform'


rotate_images(directory_path, 1400)
