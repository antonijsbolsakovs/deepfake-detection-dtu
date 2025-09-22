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

# paths
MODEL_PATH = "/content/drive/MyDrive/deepfake_detection/custom_deepfake_cnn_best.pth"
TEST_DIR = "/content/face_data/test"
SAVE_DIR = "/content/drive/MyDrive/deepfake_detection"
os.makedirs(SAVE_DIR, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# laplacian transform
class AddLaplacian:
    def __call__(self, img):
        img_gray = img.convert("L")
        np_img = np.array(img_gray)
        lap = cv2.Laplacian(np_img, cv2.CV_64F)
        lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
        lap_img = Image.fromarray(np.uint8(lap * 255))
        lap_tensor = transforms.ToTensor()(lap_img)
        img_tensor = transforms.ToTensor()(img)
        return torch.cat([img_tensor, lap_tensor], dim=0)

# transforms
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    AddLaplacian(),
    transforms.Normalize(mean=[0.5]*4, std=[0.5]*4)
])

# dataset
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# CBAM components
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + max)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class BottleneckCBAM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attn = CBAM(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.attn(self.conv(x)) + self.shortcut(x))

# full model
class DeepFakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        self.blocks = nn.Sequential(
            BottleneckCBAM(32, 32, 64), nn.MaxPool2d(2),
            BottleneckCBAM(64, 64, 128), nn.MaxPool2d(2),
            BottleneckCBAM(128, 128, 256), nn.MaxPool2d(2),
            BottleneckCBAM(256, 256, 512)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        avg = self.pool[0](x)
        max = self.pool[1](x)
        x = torch.cat([avg, max], dim=1)
        return self.classifier(x)

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
    cam = cv2.resize(cam, (299, 299))
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

# load model
model = DeepFakeCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

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
            cam = generate_gradcam(model, inputs[i].detach().clone(), pred, model.blocks[-1], device)
            save_cam_overlay(inputs[i][:3], cam, os.path.join(SAVE_DIR, f"gradcam_correct_{i}_true_{idx_to_class[label]}_pred_{idx_to_class[pred]}.png"))

        elif label != pred and len(wrong_samples) < 10:
            wrong_samples.append((img, label, pred))
            cam = generate_gradcam(model, inputs[i].detach().clone(), pred, model.blocks[-1], device)
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
df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values(), cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))

print(f"\n✅ Evaluation complete. All results saved to: {SAVE_DIR}")
