from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import subprocess

SR = 16000
HOP = 160
WIN = 400


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

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg falhou: {path}")

    y = np.frombuffer(result.stdout, np.float32)

    return y


def gerar_logmel(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=512,
        hop_length=HOP,
        win_length=WIN,
        n_mels=128,
        fmin=0,
        fmax=8000
    )

    return librosa.power_to_db(mel, ref=np.max)


def salvar_png(spec, out_path):

    spec = np.clip(spec, -80, 0)
    spec = (spec + 80) / 80

    plt.figure(figsize=(3,3), frameon=False)

    ax = plt.Axes(plt.gcf(), [0,0,1,1])
    plt.gcf().add_axes(ax)

    ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_axis_off()

    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":

    AUDIO = Path("/home/henrique/pibic/data-set-asv/PA/ASVspoof2019_PA_eval/flac/PA_E_0067638.flac")
    OUT = Path("teste_spec.png")

    print("Carregando áudio...")
    y = load_audio(AUDIO)

    print("Gerando log-mel...")
    spec = gerar_logmel(y)

    print("Salvando imagem...")
    salvar_png(spec, OUT)

    print("OK -> arquivo salvo em:", OUT.resolve())
