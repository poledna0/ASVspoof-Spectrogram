#!/bin/bash
python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/padrao-ResNet-18.py \
--img_dir_train /home/henrique/pibic/espectogramas_extremos/logmel \
--img_dir_dev /home/henrique/pibic/espectogramas_extremos/logmel \
--img_dir_eval /home/henrique/pibic/espectogramas_extremos/logmel \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name logmel 2>&1 | tee log_logmel.txt
