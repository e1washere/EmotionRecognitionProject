from PreprocessSet import preprocess_set


directory_path = './Mixed'
dataset_path = './Mixed/train_resized'
image_size = (96, 96)
emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']


preprocess_set(directory_path, dataset_path, image_size, emotions)