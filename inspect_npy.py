import numpy as np

# Исправленные пути
train_images_path = "/Users/e1/PycharmProjects/EmotionRecognitionProject_Clean/Natural_Human_Face_Images/train_images.npy"
test_images_path = "/Users/e1/PycharmProjects/EmotionRecognitionProject_Clean/Natural_Human_Face_Images/test_images.npy"
train_labels_path = "/Users/e1/PycharmProjects/EmotionRecognitionProject_Clean/Natural_Human_Face_Images/train_labels.npy"
test_labels_path = "/Users/e1/PycharmProjects/EmotionRecognitionProject_Clean/Natural_Human_Face_Images/test_labels.npy"

# Загрузка данных
train_images = np.load(train_images_path)
train_labels = np.load(train_labels_path)
test_images = np.load(test_images_path)
test_labels = np.load(test_labels_path)

# Вывод информации
print("Train Images Shape:", train_images.shape)  # Размер массива изображений
print("Train Labels Shape:", train_labels.shape)  # Размер массива лейблов
print("Test Images Shape:", test_images.shape)    # Размер массива изображений
print("Test Labels Shape:", test_labels.shape)    # Размер массива лейблов

print("\nExample of Train Image Pixel Values (first image):")
print(train_images[0])
print("\nExample of Train Label (first label):")
print(train_labels[0])
