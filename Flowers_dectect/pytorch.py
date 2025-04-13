import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # Import tqdm để hiển thị progress bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Thiết lập tham số
im_height, im_width = 224, 224
batch_size = 128
epochs = 30
data_path = 'jpeg-224x224/'
os.makedirs(data_path, exist_ok=True)

# Tạo thư mục lưu biểu đồ metric
metric_dir = 'metric'
os.makedirs(metric_dir, exist_ok=True)

# -----------------------------
# Data augmentation và transforms
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=30,
        translate=(0.3, 0.3),
        scale=(0.7, 1.3),
        shear=0.3,         # shear ở đây là độ lệch
        fill=0             # điền giá trị cho các pixel ngoài biên (giống fill_mode='nearest')
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=(0.8, 1.3)),
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor()  # tự động rescale về [0,1]
])

val_transforms = transforms.Compose([
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor()
])

# -----------------------------
# Dataset và DataLoader
# -----------------------------
train_dir = 'jpeg-224x224/train/'
if not os.path.exists(train_dir):
    print("Không tìm thấy thư mục train!")
    exit()

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)

val_dir = os.path.join(data_path, "val/")
if not os.path.exists(val_dir):
    print("Thư mục val/ không tồn tại! Vui lòng tạo thư mục và thêm dữ liệu.")
    exit()
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

print(f"Tổng số ảnh train: {len(train_dataset)}")
print(f"Tổng số ảnh validation: {len(val_dataset)}")

# Sử dụng num_workers = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

steps_per_epoch = len(train_loader)
validation_steps = len(val_loader)
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# -----------------------------
# Định nghĩa model
# -----------------------------
base_model = models.densenet121(pretrained=True)
for name, param in base_model.features.named_parameters():
    if "denseblock4" not in name:
        param.requires_grad = False

num_features = base_model.classifier.in_features
num_classes = len(train_dataset.classes)

custom_classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
    nn.Flatten(),
    nn.BatchNorm1d(num_features),
    nn.Dropout(0.5),
    nn.Linear(num_features, 1024),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

class DenseNet121Custom(nn.Module):
    def __init__(self, base_model, classifier):
        super(DenseNet121Custom, self).__init__()
        self.features = base_model.features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DenseNet121Custom(base_model, custom_classifier)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Định nghĩa loss, optimizer và scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=0.0003, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

# -----------------------------
# Training Loop với ModelCheckpoint, EarlyStopping và đo thời gian chạy
# -----------------------------
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0
patience = 5

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()  # Bắt đầu đo thời gian

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in train_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)
        train_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)
            val_total += inputs.size(0)
            val_bar.set_postfix(loss=loss.item())

    val_loss = val_running_loss / val_total
    val_acc = val_running_corrects.double() / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping")
            break

end_time = time.time()  # Kết thúc đo thời gian
elapsed_time = end_time - start_time
print(f"Thời gian huấn luyện: {elapsed_time:.2f} giây")

# Load model tốt nhất
model.load_state_dict(best_model_wts)

# -----------------------------
# Vẽ biểu đồ train và validation
# -----------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(metric_dir, 'train_val_metrics.png'))
plt.close()

# -----------------------------
# Dự đoán trên ảnh test
# -----------------------------
test_images = [
    "/content//jpeg-224x224/test/003882deb.jpeg",
    "/content/data1/jpeg-192x192/test/0021f0d33.jpeg",
    "/content//jpeg-224x224/test/004b88e09.jpeg",
    "/content/drive/MyDrive/Hình/nho.jpg",
    "/content/drive/MyDrive/Hình/hd.jpg",
    "/content/drive/MyDrive/Hình/lili.jpg"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

model.eval()
for ax, img_path in zip(axes, test_images):
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((im_width, im_height))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            prob = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(prob, 1)
            predicted_label = idx_to_class[pred.item()]
        ax.imshow(img)
        ax.set_title(f"{predicted_label}\nConf: {conf.item():.2f}")
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(metric_dir, 'test_predictions.png'))
plt.close()

# -----------------------------
# Đánh giá trên tập test
# -----------------------------
test_dir = os.path.join(data_path, "test1/")
if not os.path.exists(test_dir):
    print("Thư mục test/ không tồn tại! Vui lòng tạo thư mục và thêm dữ liệu.")
else:
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    test_running_loss = 0.0
    test_running_corrects = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_running_corrects += torch.sum(preds == labels.data)
            test_total += inputs.size(0)

    test_loss = test_running_loss / test_total
    test_acc = test_running_corrects.double() / test_total

    print(f"Số mẫu test: {len(test_dataset)}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Độ chính xác trên tập test: {test_acc.item()*100:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([test_loss], label='Test Loss', marker='o')
    plt.legend()
    plt.title('Test Loss')
    plt.xlabel('Evaluation')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot([test_acc.item()], label='Test Accuracy', marker='o')
    plt.legend()
    plt.title('Test Accuracy')
    plt.xlabel('Evaluation')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(metric_dir, 'test_metrics.png'))
    plt.close()

# -----------------------------
# Lưu lại mô hình đã huấn luyện
# -----------------------------
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, 'final_model.pth')
torch.save(model.state_dict(), model_save_path)
print("Mô hình đã được lưu tại:", model_save_path)
