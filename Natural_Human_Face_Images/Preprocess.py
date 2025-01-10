from PreprocessSet import preprocess_set


directory_path = './Natural_Human_Face_Images'
dataset_path = './Natural_Human_Face_Images/NHFI_train_resized'
image_size = (224, 224)
emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised', 'contempt']


preprocess_set(directory_path, dataset_path, image_size, emotions)