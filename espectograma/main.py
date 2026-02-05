from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt

DATASET_ROOT = Path("/home/heniruqe/b/dataset/original/PA").resolve()
OUT_ROOT = Path("/home/heniruqe/b/dataset").resolve()

AUDIO_EXT = ".flac"
SR = 16000

HOP = 160
WIN_25 = 400
WIN_20 = 320

TIPOS = ["stft", "logmel", "cqt", "lfcc", "cqcc"]

for t in TIPOS:
    (OUT_ROOT / t / "PA").mkdir(parents=True, exist_ok=True)


def load_audio(path):

    try:
        y, sr = sf.read(path)

        if y.ndim > 1:
            y = y.mean(axis=1)

        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        return y.astype(np.float32)

    except:

        cmd = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-i", str(path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(SR),
            "-"
        ]

        out = subprocess.run(cmd, stdout=subprocess.PIPE).stdout

        if len(out) == 0:
            raise RuntimeError("ffmpeg falhou")

        return np.frombuffer(out, np.float32)

def normalize_db(S):
    S = np.clip(S, -80, 0)
    S = (S + 80) / 80   # escala 0-1
    return S


def salvar_imagem(spec, caminho):

    spec = normalize_db(spec)

    caminho = caminho.with_suffix(".png")
    caminho.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3,3), frameon=False)

    ax = plt.Axes(plt.gcf(), [0,0,1,1])
    plt.gcf().add_axes(ax)

    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest"
    )

    ax.set_axis_off()

    plt.savefig(
        caminho,
        dpi=100,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

def gerar_stft(y):

    D = librosa.stft(
        y,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN_25,
        window="hann"
    )

    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S


def gerar_logmel(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN_25,
        n_mels=128,
        fmin=0,
        fmax=8000
    )

    return librosa.power_to_db(mel, ref=np.max)


def gerar_cqt(y):

    C = librosa.cqt(
        y,
        sr=SR,
        hop_length=HOP,
        bins_per_octave=12,
        n_bins=84
    )

    return librosa.amplitude_to_db(np.abs(C), ref=np.max)


def linear_filterbank(sr, n_fft, n_filters):

    freqs = np.linspace(0, sr/2, int(1 + n_fft//2))
    edges = np.linspace(0, sr/2, n_filters + 2)

    fb = np.zeros((n_filters, len(freqs)))

    for i in range(1, n_filters + 1):

        left = edges[i-1]
        center = edges[i]
        right = edges[i+1]

        fb[i-1] = np.maximum(
            0,
            np.minimum(
                (freqs-left)/(center-left),
                (right-freqs)/(right-center)
            )
        )

    return fb


LFCC_FB = linear_filterbank(SR, 1024, 20)


def gerar_lfcc(y):

    S = np.abs(librosa.stft(
        y,
        n_fft=1024,
        hop_length=HOP,
        win_length=WIN_20,
        window="hann"
    ))**2

    linear_spec = np.dot(LFCC_FB, S)
    log_spec = np.log(linear_spec + 1e-6)

    return librosa.feature.mfcc(S=log_spec, n_mfcc=20)


def gerar_cqcc(y):

    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP))
    logC = np.log(C + 1e-6)

    return librosa.feature.mfcc(S=logC, n_mfcc=30)

def processar_audio(audio_path):

    try:

        rel = audio_path.relative_to(DATASET_ROOT)

        y = load_audio(audio_path)

        salvar_imagem(gerar_stft(y), OUT_ROOT/"stft"/"PA"/rel)
        salvar_imagem(gerar_logmel(y), OUT_ROOT/"logmel"/"PA"/rel)
        salvar_imagem(gerar_cqt(y), OUT_ROOT/"cqt"/"PA"/rel)
        salvar_imagem(gerar_lfcc(y), OUT_ROOT/"lfcc"/"PA"/rel)
        salvar_imagem(gerar_cqcc(y), OUT_ROOT/"cqcc"/"PA"/rel)

    except Exception as e:
        print(f"ERRO: {audio_path} -> {e}")

if __name__ == "__main__":

    inicio = datetime.now()
    print("Início:", inicio)

    audios = list(DATASET_ROOT.rglob(f"*{AUDIO_EXT}"))
    print("Total:", len(audios))

    for audio in tqdm(audios):
        processar_audio(audio)

    fim = datetime.now()

    print("Fim:", fim)
    print("Duração:", fim - inicio)
