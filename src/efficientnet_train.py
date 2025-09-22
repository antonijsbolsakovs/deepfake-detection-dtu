import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm

# config
TRAIN_DIR = "/content/face_data/train"
SAVE_DIR = "/content/drive/MyDrive/deepfake_detection"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 3e-5
EPOCHS = 10
IMAGE_SIZE = 224  # efficientNet-B0 default input size
VALIDATION_SPLIT = 0.1

MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "efficientnet_best.pth")
LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "efficientnet_train_loss.png")
ACC_PLOT_PATH = os.path.join(SAVE_DIR, "efficientnet_val_accuracy.png")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# transform RGB
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# dataset
full_train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# efficientnet model from timm
model = timm.create_model('efficientnet_b0', pretrained=True)
model.reset_classifier(2)   # set output for 2 classes
model = model.to(device)

# unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# evaluation function
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

    # save loss plot
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)

    # save accuracy plot
    plt.figure()
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(ACC_PLOT_PATH)

# run training
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)