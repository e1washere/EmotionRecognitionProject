import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

train_images = np.load('./FER2013/train_images.npy')
train_labels = np.load('./FER2013/train_labels.npy')
test_images = np.load('./FER2013/test_images.npy')
test_labels = np.load('./FER2013/test_labels.npy')

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.repeat(train_images, 3, axis=-1)
test_images = np.repeat(test_images, 3, axis=-1)

num_classes = len(np.unique(train_labels))
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

start_time = time.time()
history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    batch_size=32,
    epochs=10,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: exit() if time.time() - start_time > 600 else None
        )
    ]
)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

report = classification_report(true_classes, predicted_classes, target_names=['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised'])
print(report)

with open("FER2013_results_tensorflow.txt", "w") as f:
    f.write(report)