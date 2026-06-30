#!/bin/bash

# =========================================================
# Modelos Faltantes
# =========================================================

# -------------------------
# EfficientNet-B0 - LFCC
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-EfficientNet-B0.py \
--img_dir_train /home/henrique/pibic/data-set-asv/lfcc/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/lfcc/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name EfficientNet-B0_lfcc 2>&1 | tee log_EfficientNet-B0_lfcc.txt


# -------------------------
# ResNet-18v2 - LogMel
# -------------------------

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/metade-ResNet-18.py \
--img_dir_train /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_dev/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--model_name ResNet-18v2_logmel 2>&1 | tee log_ResNet-18v2_logmel.txt
