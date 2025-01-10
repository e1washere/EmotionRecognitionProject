from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}")

test_labels = np.load("datasets/Natural_Human_Face_Images/test_labels.npy")
resnet_preds = np.argmax(np.load("models/resnet_preds.npy"), axis=1)
evaluate_model(test_labels, resnet_preds)