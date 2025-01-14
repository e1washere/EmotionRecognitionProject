import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.metrics import classification_report
import time

train_images = np.load('./KDEF/train_images.npy')
train_labels = np.load('./KDEF/train_labels.npy')
test_images = np.load('./KDEF/test_images.npy')
test_labels = np.load('./KDEF/test_labels.npy')

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=7)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=7)

input_shape = train_images.shape[1:]
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
x = Flatten()(base_model.output)
output = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=32, epochs=10, verbose=1)

    if time.time() - start_time > 600:
        print("Stopping training as it exceeded 10 minutes.")
        break

loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']))