from PreprocessSet import preprocess_set


directory_path = './FER2013'
dataset_path = './FER2013/train_resized'
image_size = (48, 48)
emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']


preprocess_set(directory_path, dataset_path, image_size, emotions)