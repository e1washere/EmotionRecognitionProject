import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, roc_auc_score
import time
import os

def load_dataset(dataset_path):
    train_images = np.load(f"{dataset_path}/train_images.npy")
    train_labels = np.load(f"{dataset_path}/train_labels.npy")
    test_images = np.load(f"{dataset_path}/test_images.npy")
    test_labels = np.load(f"{dataset_path}/test_labels.npy")

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=8)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=8)

    return train_images, train_labels, test_images, test_labels

def create_resnet50_model(input_shape):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    elapsed_time = time.time() - start_time
    return model, elapsed_time

if __name__ == "__main__":
    datasets = ["FER2013", "KDEF", "Mixed", "Natural_Human_Face_Images"]
    results = []

    for dataset in datasets:
        print(f"Training on dataset: {dataset}")
        dataset_path = f"./{dataset}"
        train_images, train_labels, test_images, test_labels = load_dataset(dataset_path)

        model = create_resnet50_model(input_shape=train_images.shape[1:])
        trained_model, elapsed_time = train_model(model, train_images, train_labels, test_images, test_labels, epochs=10)

        predictions = trained_model.predict(test_images)
        y_true = np.argmax(test_labels, axis=1)
        y_pred = np.argmax(predictions, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True)
        f1_score = report["weighted avg"]["f1-score"]
        accuracy = report["accuracy"]
        auc = roc_auc_score(test_labels, predictions, multi_class="ovr")

        results.append({"Dataset": dataset, "Accuracy": accuracy, "F1 Score": f1_score, "AUC": auc, "Time": elapsed_time})
        print(f"Results for {dataset}: Accuracy={accuracy}, F1 Score={f1_score}, AUC={auc}, Time={elapsed_time}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("tensorflow_results.csv", index=False)