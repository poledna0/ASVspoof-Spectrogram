import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def carregar_audio(caminho_arquivo):
    print(f"Carregando áudio: {caminho_arquivo}")
    # y: Array com o áudio carregado (sinal de onda) cada valor representa a amplitude em um ponto no tempo. por exemplo, y[0] é a amplitude no primeiro ponto de amostragem. que seria o frame 0.
    # sr: Taxa de amostragem do áudio (quantas amostras por segundo).
    # sr=None significa que o áudio será carregado com a taxa de amostragem original. ou seja uam coisa crua sem nenhuma alteração.
    y, sr = librosa.load(caminho_arquivo, sr=None)
    print(f"Taxa de amostragem: {sr} Hz")
    print(f"Duração: {len(y)/sr:.2f} segundos")
    return y, sr


def gerar_espectrograma_stft(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Calcular STFT
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plotar
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma STFT (Fourier)', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"STFT salvo em: {caminho_saida}")


def gerar_mel_espectrograma(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Calcular Mel-espectrograma
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Plotar
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Espectrograma', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência Mel')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mel-espectrograma salvo em: {caminho_saida}")


def gerar_log_mel_espectrograma(y, sr, caminho_saida):

    plt.figure(figsize=(12, 8))
    
    # Calcular Mel-espectrograma com escala logarítmica
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Plotar com escala log
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Espectrograma', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência Mel (escala log)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Log-Mel espectrograma salvo em: {caminho_saida}")


def gerar_cochleagrama(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Aproximação usando CQT (Constant-Q Transform) que simula melhor a audição humana
    C = np.abs(librosa.cqt(y, sr=sr))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    
    # Plotar
    librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Cochleagrama (CQT)', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cochleagrama salvo em: {caminho_saida}")


def gerar_log_stft_espectrograma(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Calcular STFT com escala logarítmica
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plotar com eixo Y logarítmico
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-STFT Espectrograma', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz, escala log)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Log-STFT salvo em: {caminho_saida}")


def gerar_chroma_espectrograma(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Calcular Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Plotar
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chroma Espectrograma', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Classes de Altura (Chroma)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chroma espectrograma salvo em: {caminho_saida}")


def gerar_gabor_espectrograma(y, sr, caminho_saida):
    plt.figure(figsize=(12, 8))
    
    # Aproximação usando STFT com janela gaussiana (similar ao Gabor)
    D = librosa.stft(y, window='hann', n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plotar
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Gabor Espectrograma (STFT otimizado)', fontsize=16, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Frequência (Hz)')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gabor espectrograma salvo em: {caminho_saida}")


def gerar_todos_espectrogramas(caminho_audio, diretorio_saida="espectrogramas_saida"):
    # Criar diretório de saída
    Path(diretorio_saida).mkdir(exist_ok=True)
    
    # Carregar áudio
    y, sr = carregar_audio(caminho_audio)
    
    # Nome base do arquivo
    nome_base = Path(caminho_audio).stem
    
    print("\n" + "="*60)
    print("GERANDO ESPECTROGRAMAS")
    print("="*60 + "\n")
    
    # Gerar cada tipo de espectrograma
    gerar_espectrograma_stft(y, sr, f"{diretorio_saida}/{nome_base}_01_STFT.png")
    gerar_mel_espectrograma(y, sr, f"{diretorio_saida}/{nome_base}_02_Mel.png")
    gerar_log_mel_espectrograma(y, sr, f"{diretorio_saida}/{nome_base}_03_LogMel.png")
    gerar_cochleagrama(y, sr, f"{diretorio_saida}/{nome_base}_04_Cochleagrama.png")
    gerar_log_stft_espectrograma(y, sr, f"{diretorio_saida}/{nome_base}_05_LogSTFT.png")
    gerar_chroma_espectrograma(y, sr, f"{diretorio_saida}/{nome_base}_06_Chroma.png")
    gerar_gabor_espectrograma(y, sr, f"{diretorio_saida}/{nome_base}_07_Gabor.png")
    
    print("\n" + "="*60)
    print(f"✓ TODOS OS ESPECTROGRAMAS FORAM GERADOS COM SUCESSO!")
    print(f"✓ Imagens salvas em: {diretorio_saida}/")
    print("="*60)


if __name__ == "__main__":
    # Caminho para o arquivo de áudio
    caminho_audio = "audio-teste/PA_T_0000001.flac"
    
    # Gerar todos os espectrogramas
    gerar_todos_espectrogramas(caminho_audio)
