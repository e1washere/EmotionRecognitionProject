import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_data(dataset_path):
    train_images = np.load(f"{dataset_path}/train_images.npy")
    train_labels = np.load(f"{dataset_path}/train_labels.npy")
    test_images = np.load(f"{dataset_path}/test_images.npy")
    test_labels = np.load(f"{dataset_path}/test_labels.npy")
    return train_images, train_labels, test_images, test_labels

def build_resnet50(input_shape, num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 54 * 54)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    dataset_path = "datasets/Natural_Human_Face_Images"
    input_shape = (224, 224, 1)
    num_classes = 7

    train_images, train_labels, test_images, test_labels = load_data(dataset_path)

    resnet_model = build_resnet50(input_shape, num_classes)
    resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    resnet_model.fit(
        train_images, tf.keras.utils.to_categorical(train_labels, num_classes),
        validation_split=0.2, epochs=20, batch_size=32,
        callbacks=[ModelCheckpoint("models/resnet50_model.h5", save_best_only=True)]
    )

    train_dataset = TensorDataset(torch.tensor(train_images).float(), torch.tensor(train_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = CustomCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")