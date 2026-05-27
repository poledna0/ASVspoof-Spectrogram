from pathlib import Path
import subprocess
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Keep the same specs as existing scripts
SR = 16000
HOP = 160
WIN_25 = 400
WIN_20 = 320
TIPOS = ["stft", "logmel", "cqt", "lfcc", "cqcc"]

DATASET_ROOT = next((Path(p) for p in [
    '/home/henrique/ssd/pibic/PA/PA',
    '/mnt/ssd/pibic/PA/PA',
    '/home/henrique/ssd/pibic/PA',
    '/mnt/ssd/pibic/PA'
] if Path(p).exists()), None)

if DATASET_ROOT is None:
    raise RuntimeError('Dataset PA não encontrado em /home/henrique/ssd/pibic/PA/PA ou /mnt/ssd/pibic/PA/PA')

OUT_ROOT = Path('/mnt/ssd/pibic/espectogramas_extremos')
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def salvar_imagem(spec, caminho):

    # spec assumed to be in dB and clipped to [-80, 0]
    S = np.clip(spec, -80, 0)
    S = (S + 80) / 80.0

    caminho = caminho.with_suffix('.png')
    caminho.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3), frameon=False)
    ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
    plt.gcf().add_axes(ax)
    ax.imshow(S, origin='lower', aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_axis_off()
    plt.savefig(caminho, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_audio(path):
    cmd = [
        'ffmpeg',
        '-loglevel', 'quiet',
        '-i', str(path),
        '-f', 'f32le',
        '-ac', '1',
        '-ar', str(SR),
        '-'
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f'ffmpeg falhou: {path}')
    return np.frombuffer(result.stdout, dtype=np.float32)


def gerar_stft(y):
    D = librosa.stft(y, n_fft=512, hop_length=HOP, win_length=WIN_25, window='hann')
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S


def gerar_logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP, win_length=WIN_25, n_mels=128, fmin=0, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max)


def gerar_cqt(y):
    C = librosa.cqt(y, sr=SR, hop_length=HOP, bins_per_octave=12, n_bins=84)
    return librosa.amplitude_to_db(np.abs(C), ref=np.max)


def linear_filterbank(sr, n_fft, n_filters):
    freqs = np.linspace(0, sr/2, int(1 + n_fft//2))
    edges = np.linspace(0, sr/2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(1, n_filters + 1):
        left = edges[i-1]
        center = edges[i]
        right = edges[i+1]
        fb[i-1] = np.maximum(0, np.minimum((freqs-left)/(center-left), (right-freqs)/(right-center)))
    return fb


LFCC_FB = linear_filterbank(SR, 1024, 20)


def gerar_lfcc(y):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP, win_length=WIN_20, window='hann'))**2
    linear_spec = np.dot(LFCC_FB, S)
    log_spec = np.log(linear_spec + 1e-6)
    return librosa.feature.mfcc(S=log_spec, n_mfcc=20)


def gerar_cqcc(y):
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP))
    logC = np.log(C + 1e-6)
    return librosa.feature.mfcc(S=logC, n_mfcc=30)


def keep_extremes(S_db, percentile=95, per_frame=False, min_db=-80.0):
    """
    Keep only the extreme (peak) values of a spectrogram in dB.
    - percentile: percent of values to keep (e.g. 95 keeps top 5%).
    - per_frame: if True, compute percentile per time-frame (column), else global.
    Returns a spectrogram in dB where non-extreme values are set to min_db.
    """
    if S_db.size == 0:
        return S_db

    if per_frame:
        S2 = np.full_like(S_db, min_db)
        # iterate columns
        for i in range(S_db.shape[1]):
            col = S_db[:, i]
            thr = np.percentile(col, percentile)
            mask = col >= thr
            S2[mask, i] = col[mask]
        return S2
    else:
        thr = np.percentile(S_db, percentile)
        S2 = np.where(S_db >= thr, S_db, min_db)
        return S2


def process_file(path_audio, out_root, percentile=95, per_frame=False):
    # load resampled to SR to keep same specs
    y = load_audio(path_audio)

    rel = Path(path_audio).stem

    # generate each type and save extreme-only image
    try:
        s_stft = gerar_stft(y)
        salvar_imagem(keep_extremes(s_stft, percentile, per_frame), out_root / 'stft' / f"{rel}_extremo")

        s_logmel = gerar_logmel(y)
        salvar_imagem(keep_extremes(s_logmel, percentile, per_frame), out_root / 'logmel' / f"{rel}_extremo")

        s_cqt = gerar_cqt(y)
        salvar_imagem(keep_extremes(s_cqt, percentile, per_frame), out_root / 'cqt' / f"{rel}_extremo")

        s_lfcc = gerar_lfcc(y)
        salvar_imagem(keep_extremes(s_lfcc, percentile, per_frame), out_root / 'lfcc' / f"{rel}_extremo")

        s_cqcc = gerar_cqcc(y)
        salvar_imagem(keep_extremes(s_cqcc, percentile, per_frame), out_root / 'cqcc' / f"{rel}_extremo")

    except Exception as e:
        print(f"ERRO ao processar {path_audio}: {e}")


def main():
    audio_paths = sorted(list(DATASET_ROOT.rglob('*.flac')) + list(DATASET_ROOT.rglob('*.wav')))
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"Total de arquivos: {len(audio_paths)}")
    print(f"Início: {datetime.now()}")

    for audio in tqdm(audio_paths):
        process_file(audio, OUT_ROOT, percentile=95.0, per_frame=True)

    print(f"Fim: {datetime.now()}")


if __name__ == '__main__':
    main()
