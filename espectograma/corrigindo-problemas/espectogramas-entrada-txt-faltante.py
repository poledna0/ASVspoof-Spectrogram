from pathlib import Path
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import subprocess
import tempfile
import os


DATASET_ROOT = Path("/home/heniruqe/b/dataset/DATA-SET-ORIGINAL").resolve()
OUT_ROOT = Path("/home/heniruqe/b/dataset").resolve()

FALTANTES_TXT = Path(
    "/home/heniruqe/b/ASVspoof-Spectrogram-AntiSpoofing/espectograma/corrigindo-problemas/faltantes_espectrogramas.txt"
).resolve()

LOG_ERROS = OUT_ROOT / "erros_espectrogramas.log"

SR = 16000

N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 128

TIPOS = ["mel", "logmel", "logstft"]


for t in TIPOS:
    (OUT_ROOT / t).mkdir(parents=True, exist_ok=True)


def log_erro(msg: str):
    with open(LOG_ERROS, "a") as f:
        f.write(msg + "\n")


def carregar_audio(caminho_audio: Path):

    try:
        y, sr = sf.read(caminho_audio)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "quiet",
                    "-i", str(caminho_audio),
                    "-ac", "1",
                    "-ar", str(SR),
                    tmp_path
                ],
                check=True
            )

            y, sr = sf.read(tmp_path)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    if not np.isfinite(y).all():
        raise ValueError("Áudio contém NaN ou Inf")

    return y, sr


def salvar_fig(S_db, sr, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP)
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def gerar_mel(y, sr, out: Path):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
        power=2.0
    )

    if S.max() == 0 or not np.isfinite(S).all():
        raise ValueError("Mel inválido")

    salvar_fig(librosa.power_to_db(S, ref=np.max), sr, out)


def gerar_logmel(y, sr, out: Path):
    gerar_mel(y, sr, out)


def gerar_logstft(y, sr, out: Path):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    D = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        center=False
    )

    S = np.abs(D)

    if S.max() == 0 or not np.isfinite(S).all():
        raise ValueError("STFT inválida")

    salvar_fig(librosa.amplitude_to_db(S, ref=np.max), sr, out)


def processar_audio(caminho_audio: Path):
    try:
        rel = caminho_audio.relative_to(DATASET_ROOT)
        rel_png = rel.with_suffix(".png")

        y, sr = carregar_audio(caminho_audio)

        gerar_mel(y, sr, OUT_ROOT / "mel" / rel_png)
        gerar_logmel(y, sr, OUT_ROOT / "logmel" / rel_png)
        gerar_logstft(y, sr, OUT_ROOT / "logstft" / rel_png)

    except Exception as e:
        log_erro(f"{caminho_audio} | {e}")



if __name__ == "__main__":

    inicio = datetime.now()
    print(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")

    if not FALTANTES_TXT.exists():
        raise FileNotFoundError(FALTANTES_TXT)

    with open(FALTANTES_TXT, "r") as f:
        audios = [Path(l.strip()) for l in f if l.strip()]

    print(f"Arquivos a processar: {len(audios)}")

    for audio in tqdm(audios, smoothing=0.05):
        processar_audio(audio)

    fim = datetime.now()
    print(f"Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duração total: {fim - inicio}")
