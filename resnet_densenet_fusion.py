import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

DATA_DIR = "aptos2019"
TRAIN_CSV = "train.csv"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(row['diagnosis']))
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = APTOSDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = APTOSDataset(val_df, TRAIN_IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 512)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5)
        )
    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.densenet(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3)

best_acc = 0

for epoch in range(15):
    print(f"\nEpoch {epoch + 1}/15")
    model.train()
    running_loss = 0.0

    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=True) as t_bar:
        for images, labels in t_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            t_bar.set_postfix(loss=loss.item())

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')

    scheduler.step(running_loss)

    print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_acc
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_combined_model.pth")
        print(f"Checkpoint saved: Best model updated (Accuracy: {best_acc:.4f})")

print("\nTraining Complete. Best Validation Accuracy:", best_acc)
