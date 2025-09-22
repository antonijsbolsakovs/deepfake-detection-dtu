import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import cv2

# config
TRAIN_DIR = "/content/face_data/train"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 299
VALIDATION_SPLIT = 0.1

MODEL_SAVE_PATH = "custom_deepfake_cnn_best.pth"
LOSS_PLOT_PATH = "custom_model_train_loss.png"
ACC_PLOT_PATH = "custom_model_val_accuracy.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# custom laplacian transform
class AddLaplacian:
    def __call__(self, img):
        img_gray = img.convert("L")
        np_img = np.array(img_gray)
        lap = cv2.Laplacian(np_img, cv2.CV_64F)
        lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
        lap_img = Image.fromarray(np.uint8(lap * 255))
        lap_tensor = ToTensor()(lap_img)
        img_tensor = ToTensor()(img)
        return torch.cat([img_tensor, lap_tensor], dim=0)

# combined transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    AddLaplacian(),
    transforms.Normalize(mean=[0.5]*4, std=[0.5]*4)
])

# load dataset
full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# attention blocks
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
        attn = self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# bottleneck with CBAM
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
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attn = CBAM(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.attn(self.conv(x)) + self.shortcut(x))

# main model
class DeepFakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
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

# evaluation
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# training loop
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        scheduler.step(avg_loss)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Validation Accuracy={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved to: {MODEL_SAVE_PATH}")

    # save plots
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)

    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(ACC_PLOT_PATH)

# run
if __name__ == "__main__":
    model = DeepFakeCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)