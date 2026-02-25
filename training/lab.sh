#!/bin/bash

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py \
--img_dir_train /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_dev/flac \
--img_dir_eval /home/henrique/pibic/data-set-asv/logmel/PA/ASVspoof2019_PA_eval/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name logmel 2>&1 | tee log_logmel.txt


python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py \
--img_dir_train /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_dev/flac \
--img_dir_eval /home/henrique/pibic/data-set-asv/stft/PA/ASVspoof2019_PA_eval/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name stft 2>&1 | tee log_stft.txt


python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py \
--img_dir_train /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_dev/flac \
--img_dir_eval /home/henrique/pibic/data-set-asv/cqt/PA/ASVspoof2019_PA_eval/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name cqt 2>&1 | tee log_cqt.txt


python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py \
--img_dir_train /home/henrique/pibic/data-set-asv/cqcc/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/cqcc/PA/ASVspoof2019_PA_dev/flac \
--img_dir_eval /home/henrique/pibic/data-set-asv/cqcc/PA/ASVspoof2019_PA_eval/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name cqcc 2>&1 | tee log_cqcc.txt


python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py \
--img_dir_train /home/henrique/pibic/data-set-asv/lfcc/PA/ASVspoof2019_PA_train/flac \
--img_dir_dev /home/henrique/pibic/data-set-asv/lfcc/PA/ASVspoof2019_PA_dev/flac \
--img_dir_eval /home/henrique/pibic/data-set-asv/lfcc/PA/ASVspoof2019_PA_eval/flac \
--protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt \
--protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt \
--protocol_eval /home/henrique/pibic/git/ASVspoof-Spectrogram/PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt \
--model_name lfcc 2>&1 | tee log_lfcc.txt