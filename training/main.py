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
parser.add_argument("--img_dir", required=True)
parser.add_argument("--protocol_train", required=True)
parser.add_argument("--protocol_dev", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()

IMG_DIR = args.img_dir
PROTOCOL_TRAIN = args.protocol_train
PROTOCOL_DEV = args.protocol_dev
MODEL_NAME = args.model_name


#IMG_DIR = "/home/pato/patin/data-espectograma/logmel/LA/ASVspoof2019_LA_train/flac/"
#PROTOCOL_TRAIN = "/home/pato/patin/ASVspoof-Spectrogram/LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
#PROTOCOL_DEV = "/home/pato/patin/ASVspoof-Spectrogram/LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", DEVICE)



class ASVspoofCMDataset(Dataset):

    def __init__(self, img_dir, protocol_file):
        self.img_dir = img_dir
        self.samples = []

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                # padrão do protocolo ASVspoof
                utt_id = parts[1]
                label_str = parts[-1]

                label = 0 if label_str == "bonafide" else 1
                self.samples.append((utt_id, label))

        print(f"Carregado {len(self.samples)} amostras de {protocol_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]

        img_path = os.path.join(self.img_dir, utt_id + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        # abre imagem em grayscale
        img = Image.open(img_path).convert("L")

        # transforma em tensor (0–1) e mantém 1 canal
        transform = T.ToTensor()
        spec = transform(img)  # shape: (1, H, W)

        label = torch.tensor(label, dtype=torch.long)

        return spec, label


def get_model():
    """
    EfficientNet-B0 adaptada para:
        - 1 canal (espectrograma)
        - 2 classes (bonafide / spoof)
    """

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # muda a primeira convolução para aceitar 1 canal
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )

    # camada final para 2 classes
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    return model



def compute_eer(y_true, y_score):
    """
    Calcula Equal Error Rate (EER)
    y_true  -> labels reais (0/1)
    y_score -> score contínuo de spoof (probabilidade)
    """

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    # ponto onde FPR ~= FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100


def validate_eer(model, loader):
    """
    Roda a validação e retorna o EER.
    """

    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)

            out = model(x)
            prob_spoof = torch.softmax(out, dim=1)[:, 1]

            scores.extend(prob_spoof.cpu().numpy())
            labels.extend(y.numpy())

    eer = compute_eer(np.array(labels), np.array(scores))
    return eer



def main():

    # datasets oficiais (nada de split aleatório)
    train_ds = ASVspoofCMDataset(IMG_DIR, PROTOCOL_TRAIN)
    dev_ds = ASVspoofCMDataset(IMG_DIR, PROTOCOL_DEV)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = get_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_eer = 100
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        eer = validate_eer(model, dev_loader)

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev EER: {eer:.2f}%"
        )

        # salva sempre o melhor modelo
        if eer < best_eer:
            best_eer = eer
            torch.save(
                model.state_dict(),
                f"checkpoints/{MODEL_NAME}_best.pth"
            )
            print(">> Modelo salvo (melhor EER até agora)")

    # salva o modelo final (após todas as épocas) na pasta model
    os.makedirs("model", exist_ok=True)
    torch.save(
        model.state_dict(),
        f"model/{MODEL_NAME}_final.pth"
    )
    print("Modelo final salvo")

if __name__ == "__main__":
    main()
