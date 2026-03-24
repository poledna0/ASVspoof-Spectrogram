# Detecção de Audio Replay com Deep Learning

## Descrição

Projeto focado na detecção de ataques de **Audio Replay** em sistemas de autenticação por voz, utilizando técnicas de Deep Learning aplicadas a espectrogramas.

---

## Objetivo

Desenvolver modelos capazes de diferenciar áudios reais (bonafide) de áudios reproduzidos (spoof), aumentando a segurança de sistemas biométricos.

---

## Dataset

- ASVspoof 2019 (Physical Access)

---

## Abordagem

- Conversão de áudio em espectrogramas:
  - Log-Mel
  - STFT
  - CQT
  - LFCC
  - CQCC

- Modelo:
  - EfficientNet-B0 (PyTorch)
  - CNN custom
  - ResNet-18

- Métrica principal:
  - EER (Equal Error Rate)

---

## Resultados

- Melhor desempenho: **Log-Mel (EER ~1.54%)**
- Representações espectrais completas superaram coeficientes cepstrais

---

## Autor

Henrique Poledna