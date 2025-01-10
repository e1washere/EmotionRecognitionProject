import os
import numpy as np
from PIL import Image

def preprocess_set(directory_path, dataset_path, image_size, emotions, output_prefix):
    """
    Preprocess the dataset by resizing images, converting to grayscale, normalizing,
    and saving them along with labels as .npy files.

    Parameters:
    - directory_path: Path where the .npy files will be saved.
    - dataset_path: Path to the dataset containing emotion folders.
    - image_size: Tuple (width, height) indicating the desired image size.
    - emotions: List of emotion labels (folder names).
    - output_prefix: Prefix for the output .npy files (e.g., 'train' or 'test').
    """

    emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}

    image_list = []
    label_list = []

    for emotion in emotions:
        emotion_folder = os.path.join(dataset_path, emotion)
        label = emotion_to_label[emotion]
        count = 0

        if not os.path.isdir(emotion_folder):
            print(f"Folder {emotion_folder} does not exist.")
            continue

        for filename in os.listdir(emotion_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(emotion_folder, filename)
                try:
                    img = Image.open(image_path)
                    img = img.convert('L')
                    if img.size != image_size:
                        img = img.resize(image_size)
                    img_array = np.array(img, dtype=np.float32)
                    img_array /= 255.0
                    img_array = np.expand_dims(img_array, axis=-1)
                    image_list.append(img_array)
                    label_list.append(label)

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images of {emotion}")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
        print(f"Processed {count} images of {emotion}")

    images = np.array(image_list)
    labels = np.array(label_list)

    images_save_path = os.path.join(directory_path, f'{output_prefix}_images.npy')
    labels_save_path = os.path.join(directory_path, f'{output_prefix}_labels.npy')

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    np.save(images_save_path, images)
    np.save(labels_save_path, labels)

    print(f"Total number of images: {images.shape[0]}")
    print(f"Images array shape: {images.shape}")
    print(f"Labels array shape: {labels.shape}")
    print("\nLabel mapping:")
    for emotion, idx in emotion_to_label.items():
        print(f"Label {idx}: {emotion}")

    print(f"\nPixel values range from {images.min()} to {images.max()}")