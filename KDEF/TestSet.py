import os
import shutil
from PIL import Image 
from CreateTestSet import create_test_set
# directory_path = './NHFI_uniform'
original_directory_path = './KDEF/KDEF_sorted'
new_directory_path = './KDEF/KDEF_test'

create_test_set(original_directory_path, new_directory_path, 70)


        