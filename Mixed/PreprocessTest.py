import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PreprocessSet import preprocess_set

current_dir = os.path.dirname(os.path.abspath(__file__))

directory_path = current_dir
dataset_path = os.path.join(current_dir, 'test_resized')
image_size = (96, 96)
emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']

preprocess_set(directory_path, dataset_path, image_size, emotions, output_prefix='test')