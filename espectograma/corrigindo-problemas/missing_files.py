from pathlib import Path


DATASET_ROOT = Path("/home/heniruqe/b/dataset/DATA-SET-ORIGINAL").resolve()
OUT_ROOT = Path("/home/heniruqe/b/dataset").resolve()

AUDIO_EXT = ".flac"
IMG_EXT = ".png"

TIPO_ESPECTRO = "logstft"  # mel | logmel | logstft
OUT_SPEC_ROOT = OUT_ROOT / TIPO_ESPECTRO

RELATORIO = Path("faltantes_espectrogramas.txt").resolve()


def coletar_audios(root: Path):
    return list(root.rglob(f"*{AUDIO_EXT}"))

def espectrograma_esperado(audio_path: Path) -> Path:
    rel = audio_path.relative_to(DATASET_ROOT)
    return (OUT_SPEC_ROOT / rel).with_suffix(IMG_EXT)


if __name__ == "__main__":

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)

    if not OUT_SPEC_ROOT.exists():
        raise FileNotFoundError(OUT_SPEC_ROOT)

    audios = coletar_audios(DATASET_ROOT)
    print(f"Total de áudios encontrados: {len(audios)}")

    faltantes = []

    for audio in audios:
        esperado = espectrograma_esperado(audio)
        if not esperado.exists():
            faltantes.append(audio.resolve())

    print(f"Total de espectrogramas faltantes: {len(faltantes)}")

    with open(RELATORIO, "w") as f:
        for path in faltantes:
            f.write(str(path) + "\n")

    print(f"Relatório salvo em: {RELATORIO}")
