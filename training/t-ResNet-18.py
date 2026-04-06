import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, accuracy_score
from PIL import Image
import torchvision.transforms as T
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir_train", required=True)
parser.add_argument("--img_dir_dev", required=True)
parser.add_argument("--protocol_train", required=True)
parser.add_argument("--protocol_dev", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--img_dir_eval", required=True)
parser.add_argument("--protocol_eval", required=True)
args = parser.parse_args()

BATCH_SIZE = 32
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =========================================================
# DATASET + AUGMENTATION FORTE (CRUCIAL P/ EER)
# =========================================================

class ASVspoofCMDataset(Dataset):
    def __init__(self, img_dir, protocol_file, train=True):
        self.img_dir = img_dir
        self.samples = []

        if train:
            self.transform = T.Compose([
                T.Resize((256,256)),
                T.RandomResizedCrop(224, scale=(0.7,1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2,0.2,0.2),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(3, sigma=(0.1,2.0)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[1]
                label = 0 if parts[-1] == "bonafide" else 1
                self.samples.append((utt_id, label))

        print(len(self.samples), "samples loaded")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, utt_id+".png")).convert("RGB")
        return self.transform(img), torch.tensor(label), utt_id

# =========================================================
# FOCAL LOSS + LABEL SMOOTHING (MELHOR PRA EER)
# =========================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1-pt)**self.gamma * ce_loss).mean()

# =========================================================
# RESNET50 + HEAD PROFUNDA
# =========================================================

def get_model():
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()

    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features,256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256,2)
    )
    return nn.Sequential(backbone, head)

# =========================================================

def compute_eer(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr-fnr))
    eer = (fpr[idx]+fnr[idx])/2
    return eer*100, thr[idx]

def validate(model, loader, phase):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x,y,_ in loader:
            probs = torch.softmax(model(x.to(DEVICE)), dim=1)[:,1].cpu().numpy()
            scores.extend(probs); labels.extend(y.numpy())
    eer,_ = compute_eer(np.array(labels), np.array(scores))
    print(f"{phase} EER: {eer:.3f}%")
    return eer

# =========================================================
# TRAIN
# =========================================================

train_ds = ASVspoofCMDataset(args.img_dir_train, args.protocol_train, True)
dev_ds   = ASVspoofCMDataset(args.img_dir_dev, args.protocol_dev, False)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
dev_loader   = DataLoader(dev_ds, BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

model = get_model().to(DEVICE)

# FREEZE BACKBONE INICIO
for p in model[0].parameters():
    p.requires_grad = False

criterion = FocalLoss()

optimizer = torch.optim.AdamW([
    {"params": model[0].parameters(), "lr":1e-5},
    {"params": model[1].parameters(), "lr":1e-4}
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_eer = 100
patience = 10
no_improve = 0

for epoch in range(EPOCHS):

    # UNFREEZE depois de estabilizar head
    if epoch == 5:
        print("Unfreezing backbone")
        for p in model[0].parameters():
            p.requires_grad = True

    model.train()
    for x,y,_ in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    scheduler.step()
    eer = validate(model, dev_loader, "DEV")

    if eer < best_eer:
        best_eer = eer
        no_improve = 0
        torch.save(model.state_dict(), f"{args.model_name}_best.pth")
        print("BEST MODEL SAVED")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("EARLY STOP")
            break

# =========================================================
# EVAL FINAL
# =========================================================

print("\nFinal evaluation")
model.load_state_dict(torch.load(f"{args.model_name}_best.pth"))
eval_ds = ASVspoofCMDataset(args.img_dir_eval, args.protocol_eval, False)
eval_loader = DataLoader(eval_ds, BATCH_SIZE, shuffle=False)
validate(model, eval_loader, "EVAL")