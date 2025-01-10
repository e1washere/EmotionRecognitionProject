import os
import shutil
from PIL import Image, ImageEnhance

from BrightenImages import brighten_images

directory_path = './FER2013/test_uniform'

brighten_images(directory_path, 880)
