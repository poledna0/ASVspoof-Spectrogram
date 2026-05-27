from pathlib import Path
import subprocess
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# =========================================================
# CONFIGURAÇÕES
# =========================================================

# caminho do dataset PA
DATASET_ROOT = Path("/mnt/ssd/pibic/PA").resolve()

# saída dos espectrogramas extremos
OUT_ROOT = Path("/mnt/ssd/pibic/espectogramas_extremos").resolve()

# extensões aceitas
AUDIO_EXTS = [".flac", ".wav"]

# parâmetros de processamento
SR = 16000
HOP = 160
WIN_25 = 400
WIN_20 = 320

# tipos de espectrograma
TIPOS = ["stft", "logmel", "cqt", "lfcc", "cqcc"]

# keep top X%
PERCENTILE = 95.0

# True = threshold por frame
# False = threshold global
PER_FRAME = True

# =========================================================

if not DATASET_ROOT.exists():
    raise RuntimeError(f"DATASET_ROOT não existe: {DATASET_ROOT}")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

for t in TIPOS:
    (OUT_ROOT / t / "PA").mkdir(parents=True, exist_ok=True)


def salvar_imagem(spec, caminho):

    spec = np.clip(spec, -80, 0)
    spec = (spec + 80) / 80.0

    caminho = caminho.with_suffix(".png")
    caminho.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3), frameon=False)

    ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
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

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg falhou: {path}")

    return np.frombuffer(result.stdout, dtype=np.float32)


def gerar_stft(y):

    D = librosa.stft(
        y,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN_25,
        window="hann"
    )

    return librosa.amplitude_to_db(np.abs(D), ref=np.max)


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

    freqs = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    edges = np.linspace(0, sr / 2, n_filters + 2)

    fb = np.zeros((n_filters, len(freqs)))

    for i in range(1, n_filters + 1):

        left = edges[i - 1]
        center = edges[i]
        right = edges[i + 1]

        fb[i - 1] = np.maximum(
            0,
            np.minimum(
                (freqs - left) / (center - left),
                (right - freqs) / (right - center)
            )
        )

    return fb


LFCC_FB = linear_filterbank(SR, 1024, 20)


def gerar_lfcc(y):

    S = np.abs(
        librosa.stft(
            y,
            n_fft=1024,
            hop_length=HOP,
            win_length=WIN_20,
            window="hann"
        )
    ) ** 2

    linear_spec = np.dot(LFCC_FB, S)

    log_spec = np.log(linear_spec + 1e-6)

    return librosa.feature.mfcc(
        S=log_spec,
        n_mfcc=20
    )


def gerar_cqcc(y):

    C = np.abs(
        librosa.cqt(
            y,
            sr=SR,
            hop_length=HOP
        )
    )

    logC = np.log(C + 1e-6)

    return librosa.feature.mfcc(
        S=logC,
        n_mfcc=30
    )


def keep_extremes(
    S_db,
    percentile=95,
    per_frame=False,
    min_db=-80.0
):

    if S_db.size == 0:
        return S_db

    if per_frame:

        S2 = np.full_like(S_db, min_db)

        for i in range(S_db.shape[1]):

            col = S_db[:, i]

            thr = np.percentile(col, percentile)

            mask = col >= thr

            S2[mask, i] = col[mask]

        return S2

    else:

        thr = np.percentile(S_db, percentile)

        return np.where(S_db >= thr, S_db, min_db)


def process_file(path_audio):

    try:

        y = load_audio(path_audio)

        # preserva train/dev/eval/flac/arquivo
        rel = path_audio.relative_to(DATASET_ROOT)

        salvar_imagem(
            keep_extremes(
                gerar_stft(y),
                percentile=PERCENTILE,
                per_frame=PER_FRAME
            ),
            OUT_ROOT / "stft" / "PA" / rel
        )

        salvar_imagem(
            keep_extremes(
                gerar_logmel(y),
                percentile=PERCENTILE,
                per_frame=PER_FRAME
            ),
            OUT_ROOT / "logmel" / "PA" / rel
        )

        salvar_imagem(
            keep_extremes(
                gerar_cqt(y),
                percentile=PERCENTILE,
                per_frame=PER_FRAME
            ),
            OUT_ROOT / "cqt" / "PA" / rel
        )

        salvar_imagem(
            keep_extremes(
                gerar_lfcc(y),
                percentile=PERCENTILE,
                per_frame=PER_FRAME
            ),
            OUT_ROOT / "lfcc" / "PA" / rel
        )

        salvar_imagem(
            keep_extremes(
                gerar_cqcc(y),
                percentile=PERCENTILE,
                per_frame=PER_FRAME
            ),
            OUT_ROOT / "cqcc" / "PA" / rel
        )

    except Exception as e:

        print(f"ERRO: {path_audio} -> {e}")


def main():

    inicio = datetime.now()

    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"Início: {inicio}")

    audio_paths = []

    for ext in AUDIO_EXTS:
        audio_paths.extend(DATASET_ROOT.rglob(f"*{ext}"))

    audio_paths = sorted(audio_paths)

    print(f"Total de arquivos: {len(audio_paths)}")

    for audio in tqdm(audio_paths):
        process_file(audio)

    fim = datetime.now()

    print(f"Fim: {fim}")
    print(f"Duração: {fim - inicio}")


if __name__ == "__main__":
    main()