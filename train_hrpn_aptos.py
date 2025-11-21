import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

class APTOSGridDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, grid_size=16, base_size=256, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.grid_size = grid_size
        self.base_size = base_size
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['id_code']
        if not img_name.lower().endswith('.png'):
            img_name += '.png'
        img_path = os.path.join(self.img_dir, img_name)

        image = read_image(img_path).float() / 255.0
        image = transforms.functional.resize(image, (self.base_size, self.base_size))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            label = 0  # dummy label for test set
        else:
            label = int(row['diagnosis'])

        return image, label

class GridBlockCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(GridBlockCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.cnn(x)

class GridRetinopathyModel(nn.Module):
    def __init__(self, grid_size=16, cnn_out_channels=16, rnn_hidden_size=32, num_classes=5):
        super(GridRetinopathyModel, self).__init__()
        self.grid_size = grid_size
        self.block_cnn = GridBlockCNN(out_channels=cnn_out_channels)
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=rnn_hidden_size, batch_first=True)
        self.classifier = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        grid_H, grid_W = H // self.grid_size, W // self.grid_size

        small_features = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                patch = x[:, :, r*grid_H:(r+1)*grid_H, c*grid_W:(c+1)*grid_W]
                feat = self.block_cnn(patch)
                pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                small_features.append(pooled)

        seq_features = []
        size = 1
        while size <= self.grid_size:
            for r in range(0, self.grid_size, size):
                for c in range(0, self.grid_size, size):
                    blocks = []
                    for rr in range(r, min(r+size, self.grid_size)):
                        for cc in range(c, min(c+size, self.grid_size)):
                            idx = rr * self.grid_size + cc
                            blocks.append(small_features[idx])
                    block_feat = torch.stack(blocks, dim=1)
                    agg = block_feat.mean(dim=1)
                    seq_features.append(agg)
            size *= 2

        seq_tensor = torch.stack(seq_features, dim=1)
        rnn_out, _ = self.gru(seq_tensor)
        out = self.classifier(rnn_out[:, -1, :])
        return out

def main():
    train_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
    val_transform = train_transform

    train_dataset = APTOSGridDataset('train.csv', 'aptos2019/train_images', transform=train_transform, is_test=False)
    val_dataset = APTOSGridDataset('test.csv', 'aptos2019/test_images', transform=val_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GridRetinopathyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Validation Accuracy={accuracy:.4f}")

    torch.save(model.state_dict(), 'grid_retinopathy_model.pth')

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
