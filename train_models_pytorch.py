import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score
import time
import numpy as np
import pandas as pd

def load_dataset(dataset_path):
    train_images = np.load(f"{dataset_path}/train_images.npy")
    train_labels = np.load(f"{dataset_path}/train_labels.npy")
    test_images = np.load(f"{dataset_path}/test_images.npy")
    test_labels = np.load(f"{dataset_path}/test_labels.npy")

    train_images = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
    test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_images, train_labels, test_images, test_labels

def create_resnet50_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model

def train_model(model, train_loader, test_loader, device, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    elapsed_time = time.time() - start_time
    return model, elapsed_time

if __name__ == "__main__":
    datasets = ["FER2013", "KDEF", "Mixed", "Natural_Human_Face_Images"]
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in datasets:
        print(f"Training on dataset: {dataset}")
        dataset_path = f"./{dataset}"
        train_images, train_labels, test_images, test_labels = load_dataset(dataset_path)

        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = create_resnet50_model()
        trained_model, elapsed_time = train_model(model, train_loader, test_loader, device, epochs=10)

        model.eval()
        y_true = []
        y_pred = []
        predictions = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                predictions.extend(outputs.cpu().numpy())

        report = classification_report(y_true, y_pred, output_dict=True)
        f1_score = report["weighted avg"]["f1-score"]
        accuracy = report["accuracy"]
        auc = roc_auc_score(test_labels.numpy(), predictions, multi_class="ovr")

        results.append({"Dataset": dataset, "Accuracy": accuracy, "F1 Score": f1_score, "AUC": auc, "Time": elapsed_time})
        print(f"Results for {dataset}: Accuracy={accuracy}, F1 Score={f1_score}, AUC={auc}, Time={elapsed_time}")

    df = pd.DataFrame(results)
    df.to_csv("pytorch_results.csv", index=False)