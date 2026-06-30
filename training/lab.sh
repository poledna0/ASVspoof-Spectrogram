#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/henrique/pibic/data-set-asv}"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts/training_runs"
LOG_DIR="${ARTIFACTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# =========================================================
# Modelos Faltantes
# =========================================================

# -------------------------
# EfficientNet-B0 - LFCC
# -------------------------

python3 "${ROOT_DIR}/training/metade-EfficientNet-B0.py" \
--img_dir_train "${DATA_ROOT}/lfcc/PA/ASVspoof2019_PA_train/flac" \
--img_dir_dev "${DATA_ROOT}/lfcc/PA/ASVspoof2019_PA_dev/flac" \
--protocol_train "${ROOT_DIR}/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt" \
--protocol_dev "${ROOT_DIR}/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt" \
--model_name EfficientNet-B0_lfcc 2>&1 | tee "${LOG_DIR}/log_EfficientNet-B0_lfcc.txt"


# -------------------------
# ResNet-18v2 - LogMel
# -------------------------

python3 "${ROOT_DIR}/training/metade-ResNet-18.py" \
--img_dir_train "${DATA_ROOT}/logmel/PA/ASVspoof2019_PA_train/flac" \
--img_dir_dev "${DATA_ROOT}/logmel/PA/ASVspoof2019_PA_dev/flac" \
--protocol_train "${ROOT_DIR}/PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt" \
--protocol_dev "${ROOT_DIR}/PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt" \
--model_name ResNet-18v2_logmel 2>&1 | tee "${LOG_DIR}/log_ResNet-18v2_logmel.txt"
