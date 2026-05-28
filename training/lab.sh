#!/bin/bash

# =========================================================
# EfficientNet-B0
# =========================================================

# -------------------------
# EfficientNet-B0 - CQCC
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-EfficientNet-B0.py \
--img_dir_train /home/henrique/pibic/data-set-asv/cqcc/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/cqcc/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name EfficientNet-B0_cqcc 2>&1 | tee log_EfficientNet-B0_cqcc.txt


# -------------------------
# EfficientNet-B0 - LogMel
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-EfficientNet-B0.py \
--img_dir_train /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name EfficientNet-B0_logmel 2>&1 | tee log_EfficientNet-B0_logmel.txt


# -------------------------
# EfficientNet-B0 - STFT
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-EfficientNet-B0.py \
--img_dir_train /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name EfficientNet-B0_stft 2>&1 | tee log_EfficientNet-B0_stft.txt


# -------------------------
# EfficientNet-B0 - CQT
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-EfficientNet-B0.py \
--img_dir_train /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name EfficientNet-B0_cqt 2>&1 | tee log_EfficientNet-B0_cqt.txt


# =========================================================
# ResNet-18v2
# =========================================================

# -------------------------
# ResNet-18v2 - CQT
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-ResNet-18.py \
--img_dir_train /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name ResNet-18v2_cqt 2>&1 | tee log_ResNet-18v2_cqt.txt


# -------------------------
# ResNet-18v2 - STFT
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-ResNet-18.py \
--img_dir_train /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name ResNet-18v2_stft 2>&1 | tee log_ResNet-18v2_stft.txt