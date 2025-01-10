from PreprocessSet import preprocess_set


directory_path = './KDEF'
dataset_path = './KDEF/KDEF_train_resized'
image_size = (224, 224)
emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']


preprocess_set(directory_path, dataset_path, image_size, emotions)