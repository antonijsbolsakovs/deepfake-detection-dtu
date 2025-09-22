import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import timm

# paths
MODEL_PATH = "/content/drive/MyDrive/deepfake_detection/efficientnet_best.pth"
TEST_DIR = "/content/face_data/test"
SAVE_DIR = "/content/drive/MyDrive/deepfake_detection"
os.makedirs(SAVE_DIR, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transforms RGB
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# dataset
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# load model
model = timm.create_model('efficientnet_b0', pretrained=False)
model.reset_classifier(2)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# grad-CAM functions
def generate_gradcam(model, input_tensor, target_class, target_layer, device):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    output = model(input_tensor)
    model.zero_grad()
    target = output[0, target_class]
    target.backward()

    grads_val = gradients[0][0].cpu().data.numpy()
    activations_val = activations[0][0].cpu().data.numpy()
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)

    forward_handle.remove()
    backward_handle.remove()

    return cam

def save_cam_overlay(original_img_tensor, cam, save_path):
    img = original_img_tensor[:3].cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255
    img = np.uint8(np.clip(img, 0, 255))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = np.float32(heatmap) * 0.5 + np.float32(img)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    cv2.imwrite(save_path, overlay[:, :, ::-1])

# evaluate
y_true, y_pred = [], []
correct_samples, wrong_samples = [], []

for inputs, labels in tqdm(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    preds = torch.argmax(outputs, 1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

    for i in range(len(labels)):
        img = inputs[i][:3].cpu()
        label = labels[i].item()
        pred = preds[i].item()

        if label == pred and len(correct_samples) < 10:
            correct_samples.append((img, label, pred))
            cam = generate_gradcam(model, inputs[i].detach().clone(), pred, model.conv_stem, device)
            save_cam_overlay(inputs[i][:3], cam, os.path.join(SAVE_DIR, f"gradcam_correct_{i}_true_{idx_to_class[label]}_pred_{idx_to_class[pred]}.png"))

        elif label != pred and len(wrong_samples) < 10:
            wrong_samples.append((img, label, pred))
            cam = generate_gradcam(model, inputs[i].detach().clone(), pred, model.conv_stem, device)
            save_cam_overlay(inputs[i][:3], cam, os.path.join(SAVE_DIR, f"gradcam_wrong_{i+100}_true_{idx_to_class[label]}_pred_{idx_to_class[pred]}.png"))

# metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(idx_to_class.values())))

# save metrics
df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1
}])
df.to_csv(os.path.join(SAVE_DIR, "metrics_efficientnet.csv"), index=False)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values(), cmap="Blues")
plt.title("Confusion Matrix - EfficientNet")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_efficientnet.png"))

print(f"\n Evaluation complete. All results saved to: {SAVE_DIR}")