import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_curve
from PIL import Image
import torchvision.transforms as T
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--img_dir_train", required=True)
parser.add_argument("--img_dir_dev", required=True)
parser.add_argument("--protocol_train", required=True)
parser.add_argument("--protocol_dev", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()

BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", DEVICE)

class ASVspoofCMDataset(Dataset):

    def __init__(self, img_dir, protocol_file):
        self.img_dir = img_dir
        self.samples = []

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()

                utt_id = parts[1]
                label = 0 if parts[-1] == "bonafide" else 1

                self.samples.append((utt_id, label))

        print(f"{len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        utt_id, label = self.samples[idx]

        img_path = os.path.join(self.img_dir, utt_id + ".png")

        if not os.path.exists(img_path):
            raise RuntimeError(f"Imagem faltando: {img_path}")

        img = Image.open(img_path).convert("L")
        spec = self.transform(img)

        return spec, torch.tensor(label), utt_id

def get_model():

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    return model

def compute_eer(y_true, scores):

    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100

def validate_and_save(model, loader, score_path):

    model.eval()

    scores = []
    labels = []
    lines = []

    with torch.no_grad():
        for x, y, utt in loader:

            x = x.to(DEVICE)

            logits = model(x)
            spoof_scores = logits[:,1].cpu().numpy()

            for u, s, l in zip(utt, spoof_scores, y):
                lines.append(f"{u} {s}\n")
                scores.append(s)
                labels.append(l.item())

    with open(score_path, "w") as f:
        f.writelines(lines)

    eer = compute_eer(np.array(labels), np.array(scores))

    return eer

def main():

    train_ds = ASVspoofCMDataset(
        args.img_dir_train,
        args.protocol_train
    )

    dev_ds = ASVspoofCMDataset(
        args.img_dir_dev,
        args.protocol_dev
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("scores", exist_ok=True)

    best_eer = 100
    patience = 15
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for x, y, _ in train_loader:

            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        score_file = f"scores/{args.model_name}_dev_scores.txt"

        eer = validate_and_save(
            model,
            dev_loader,
            score_file
        )

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} | "
            f"Dev EER: {eer:.2f}%"
        )

        if eer < best_eer:

            best_eer = eer
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                f"checkpoints/{args.model_name}_best.pth"
            )

            print("BEST MODEL SALVO")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("EARLY STOPPING")
                break


    print("\nTreino finalizado.")


if __name__ == "__main__":
    main()
