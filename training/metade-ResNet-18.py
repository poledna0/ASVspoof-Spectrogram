import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from PIL import Image
import torchvision.transforms as T
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

# Argumentos
parser.add_argument("--img_dir_train", required=True)
parser.add_argument("--img_dir_dev", required=True)
parser.add_argument("--protocol_train", required=True)
parser.add_argument("--protocol_dev", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", DEVICE)


class ASVspoofCMDataset(Dataset):

    def __init__(self, img_dir, protocol_file):
        self.img_dir = img_dir
        self.samples = []

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[1]
                label = 0 if parts[-1] == "bonafide" else 1
                self.samples.append((utt_id, label))

        print(f"{len(self.samples)} samples loaded from {protocol_file}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, utt_id + ".png")

        if not os.path.exists(img_path):
            raise RuntimeError(f"Imagem faltando: {img_path}")

        img = Image.open(img_path).convert("RGB")
        spec = self.transform(img)

        return spec, torch.tensor(label), utt_id


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # regularização, reduz overfitting
        nn.Linear(model.fc.in_features, 2)
    )

    return model


def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]

    return eer * 100, eer_threshold


def compute_confusion(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print("\n===== MATRIZ DE CONFUSÃO =====")
    print("Formato: [[TN FP] [FN TP]]")
    print(cm)
    print(f"Accuracy: {acc:.4f}\n")

    print("===== CLASSIFICATION REPORT =====")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["bonafide", "spoof"]
    ))


def validate_and_save(model, loader, score_path, phase_name="DEV"):

    model.eval()

    scores = []
    labels = []
    lines = []

    with torch.no_grad():
        for x, y, utt in loader:
            x = x.to(DEVICE)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)
            spoof_scores = probs[:, 1].cpu().numpy()

            for u, s, l in zip(utt, spoof_scores, y):
                lines.append(f"{u} {s}\n")
                scores.append(s)
                labels.append(l.item())

    with open(score_path, "w") as f:
        f.writelines(lines)

    scores = np.array(scores)
    labels = np.array(labels)

    eer, eer_th = compute_eer(labels, scores)

    print(f"\n--- RESULTADOS {phase_name} ---")
    print(f"EER: {eer:.2f}%")
    print(f"EER Threshold: {eer_th:.6f}")

    compute_confusion(labels, scores, eer_th)

    return eer


def main():

    train_ds = ASVspoofCMDataset(args.img_dir_train, args.protocol_train)
    dev_ds = ASVspoofCMDataset(args.img_dir_dev, args.protocol_dev)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5
    )

    # =========================
    # Pastas únicas por execução
    # =========================

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(
        "runs",
        f"{args.model_name}_{run_id}"
    )

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    scores_dir = os.path.join(output_dir, "scores")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    print(f"\nSALVANDO RESULTADOS EM: {output_dir}")

    # =========================

    best_eer = 100
    patience = 10
    epochs_without_improvement = 0

    print("\n--- INICIANDO TREINAMENTO ---")

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for x, y, _ in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        score_file_dev = os.path.join(
            scores_dir,
            f"{args.model_name}_DEV_scores.txt"
        )

        eer = validate_and_save(
            model,
            dev_loader,
            score_file_dev,
            "DEV"
        )

        scheduler.step(eer)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {train_loss:.4f}")

        if eer < best_eer:

            best_eer = eer
            epochs_without_improvement = 0

            best_model_path = os.path.join(
                checkpoints_dir,
                f"{args.model_name}_best.pth"
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(">>> BEST MODEL SALVO")

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(">>> EARLY STOPPING ACIONADO")
                break

    print("\n--- TREINO FINALIZADO ---")

    print(f"\nMELHOR EER DEV: {best_eer:.2f}%")

    print(f"MODELO SALVO EM: {best_model_path}")

    print(f"SCORES DEV SALVOS EM: {score_file_dev}")


if __name__ == "__main__":
    main()