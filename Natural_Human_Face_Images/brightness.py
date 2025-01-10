import os
import shutil
from PIL import Image, ImageEnhance

from BrightenImages import brighten_images

directory_path = './Natural_Human_Face_Images/NHFI_uniform'

brighten_images(directory_path, 1400)
