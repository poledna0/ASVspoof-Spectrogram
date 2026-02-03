#!/bin/bash

python3 /home/henrique/pibic/git/ASVspoof-Spectrogram/training/main.py --img_dir_train /home/henrique/pibic/data-set-asv/logmel/LA/ASVspoof2019_LA_train/flac --img_dir_dev /home/henrique/pibic/data-set-asv/logmel/LA/ASVspoof2019_LA_dev/flac --protocol_train /home/henrique/pibic/git/ASVspoof-Spectrogram/LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --protocol_dev /home/henrique/pibic/git/ASVspoof-Spectrogram/LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --model_name logmel
