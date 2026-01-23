from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


DATASET_ROOT = Path("/home/heniruqe/b/dataset/DATA-SET-ORIGINAL").resolve()
OUT_ROOT = Path("/home/heniruqe/b/dataset").resolve()

AUDIO_EXT = ".flac"
SR = 16000

TIPOS = ["mel", "logmel", "logstft"]

# cria as pastas base
for t in TIPOS:
    (OUT_ROOT / t).mkdir(parents=True, exist_ok=True)


def gerar_mel_espectrograma(y, sr, caminho_saida: Path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr)
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def gerar_log_mel_espectrograma(y, sr, caminho_saida: Path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr)
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def gerar_log_stft_espectrograma(y, sr, caminho_saida: Path):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr)
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def processar_audio(caminho_audio: Path):
    try:
        # caminho relativo ao dataset
        rel = caminho_audio.relative_to(DATASET_ROOT)

        # troca extensão
        rel_png = rel.with_suffix(".png")

        y, sr = librosa.load(caminho_audio, sr=SR)

        gerar_mel_espectrograma(
            y, sr,
            OUT_ROOT / "mel" / rel_png
        )

        gerar_log_mel_espectrograma(
            y, sr,
            OUT_ROOT / "logmel" / rel_png
        )

        gerar_log_stft_espectrograma(
            y, sr,
            OUT_ROOT / "logstft" / rel_png
        )

    except Exception as e:
        print(f"Erro em {caminho_audio}: {e}")


def coletar_audios(root: Path):
    return list(root.rglob(f"*{AUDIO_EXT}"))


if __name__ == "__main__":

    inicio = datetime.now()
    print(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET_ROOT}")

    audios = coletar_audios(DATASET_ROOT)
    print(f"Total de áudios encontrados: {len(audios)}")

    for audio in tqdm(audios):
        processar_audio(audio)

    fim = datetime.now()
    print(f"Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duração total: {fim - inicio}")


