import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from cm_dataset import ASVspoofCMDataset

IMG_DIR = "/home/heniruqe/b/dataset/mel/LA/ASVspoof2019_LA_train/flac"
PROTOCOL = "/home/heniruqe/b/ASVspoof-Spectrogram/LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

dataset = ASVspoofCMDataset(IMG_DIR, PROTOCOL)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")


os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_mel_LA.pth")
