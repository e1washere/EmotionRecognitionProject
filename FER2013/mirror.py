import os
import shutil
from PIL import Image 

from MirrorImages import mirror_images

directory_path = directory_path = './FER2013/test_uniform'


mirror_images(directory_path, 880)

