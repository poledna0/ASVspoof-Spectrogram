from pathlib import Path
import numpy as np
import librosa
import subprocess
import matplotlib.pyplot as plt

AUDIO_PATH = Path("/home/henrique/minha/PA/ASVspoof2019_PA_train/flac/PA_T_0000001.flac")
OUT_DIR = Path("relatorio_parcial")
OUT_DIR.mkdir(exist_ok=True)

SR = 16000
HOP = 160
WIN_25 = 400
WIN_20 = 320



def load_audio(path):

    cmd = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SR),
        "-"
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE)

    y = np.frombuffer(result.stdout, np.float32)
    return y


def normalize_db(S):
    S = np.clip(S, -80, 0)
    return (S + 80) / 80

def salvar_imagem(spec, nome):

    spec = normalize_db(spec)

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

    plt.savefig(OUT_DIR / f"{nome}.png",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0)

    plt.close()


def gerar_stft(y):
    D = librosa.stft(y, n_fft=512, hop_length=HOP,
                     win_length=WIN_25, window="hann")
    return librosa.amplitude_to_db(np.abs(D), ref=np.max)

def gerar_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN_25,
        n_mels=128,
        fmin=0,
        fmax=8000
    )
    return librosa.power_to_db(mel, ref=np.max)

def gerar_cqt(y):
    C = librosa.cqt(y, sr=SR,
                    hop_length=HOP,
                    bins_per_octave=12,
                    n_bins=84)
    return librosa.amplitude_to_db(np.abs(C), ref=np.max)

def linear_filterbank(sr, n_fft, n_filters):
    freqs = np.linspace(0, sr/2, int(1 + n_fft//2))
    edges = np.linspace(0, sr/2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))

    for i in range(1, n_filters + 1):
        left, center, right = edges[i-1], edges[i], edges[i+1]
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


print("Carregando áudio...")
y = load_audio(AUDIO_PATH)

print("Gerando espectrogramas...")

salvar_imagem(gerar_stft(y), "stft")
salvar_imagem(gerar_logmel(y), "logmel")
salvar_imagem(gerar_cqt(y), "cqt")
salvar_imagem(gerar_lfcc(y), "lfcc")
salvar_imagem(gerar_cqcc(y), "cqcc")

print("Finalizado.")
print("Arquivos salvos em:", OUT_DIR.resolve())