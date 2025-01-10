import os
import shutil
from PIL import Image 

from MirrorImages import mirror_images

directory_path = './Natural_Human_Face_Images/NHFI_uniform'


mirror_images(directory_path, 1400)

