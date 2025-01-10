import os
import shutil
from PIL import Image, ImageEnhance

from DarkenImages import darken_images

directory_path = './FER2013/test_uniform'

darken_images(directory_path, 880)
