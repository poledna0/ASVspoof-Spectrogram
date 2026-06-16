import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, accuracy_score, precision_recall_fscore_support
)
from PIL import Image
import torchvision.transforms as T

# Adiciona o diretório tdcf ao path para importar as métricas
sys.path.insert(0, str(Path(__file__).parent / 'tdcf'))
try:
    import eval_metrics as em
except ImportError:
    em = None
    print("Aviso: Módulo eval_metrics não encontrado em tdcf/. min-tDCF não será calculado.")

# Configuração de Logs
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("EnsembleInference")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

# ==========================================
# Definição dos Ensembles
# ==========================================
ENSEMBLES = [
    {
        "id": 1,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_logmel', 'EfficientNet-B0_stft', 'ResNet-18v2_cqt'],
        "weights": [0.170, 0.461, 0.191, 0.178]
    },
    {
        "id": 2,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_cqt', 'EfficientNet-B0_logmel', 'ResNet-18v2_stft'],
        "weights": [0.234, 0.268, 0.287, 0.211]
    },
    {
        "id": 3,
        "models": ['EfficientNet-B0_logmel', 'EfficientNet-B0_stft', 'ResNet-18v2_cqt', 'ResNet-18v2_stft'],
        "weights": [0.275, 0.300, 0.223, 0.201]
    },
    {
        "id": 4,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_stft', 'ResNet-18v2_cqt', 'ResNet-18v2_logmel'],
        "weights": [0.181, 0.213, 0.186, 0.420]
    },
    {
        "id": 5,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_cqt', 'EfficientNet-B0_stft', 'ResNet-18v2_logmel'],
        "weights": [0.194, 0.227, 0.196, 0.383]
    },
    {
        "id": 6,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_cqt', 'EfficientNet-B0_logmel', 'ResNet-18v2_logmel'],
        "weights": [0.207, 0.226, 0.389, 0.178]
    },
    {
        "id": 7,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_logmel', 'ResNet-18v2_cqt', 'ResNet-18v2_stft'],
        "weights": [0.252, 0.289, 0.260, 0.199]
    },
    {
        "id": 8,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_cqt', 'EfficientNet-B0_logmel', 'EfficientNet-B0_stft'],
        "weights": [0.143, 0.256, 0.449, 0.152]
    },
    {
        "id": 9,
        "models": ['EfficientNet-B0_logmel', 'EfficientNet-B0_stft', 'ResNet-18v2_cqt', 'ResNet-18v2_logmel'],
        "weights": [0.191, 0.355, 0.225, 0.229]
    },
    {
        "id": 10,
        "models": ['EfficientNet-B0_cqcc', 'EfficientNet-B0_cqt', 'EfficientNet-B0_stft', 'ResNet-18v2_stft'],
        "weights": [0.225, 0.294, 0.224, 0.257]
    }
]

COST_MODEL = {
    'Pspoof': 0.05,
    'Ptar': (1 - 0.05) * 0.99,
    'Pnon': (1 - 0.05) * 0.01,
    'Cmiss_asv': 1,
    'Cfa_asv': 10,
    'Cmiss_cm': 1,
    'Cfa_cm': 10,
}

# ==========================================
# Componentes do Pipeline
# ==========================================

class ASVspoofCMDataset(Dataset):
    def __init__(self, img_dir, protocol_file, arch):
        self.img_dir = img_dir
        self.arch = arch
        self.samples = []

        if "EfficientNet" in arch:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485], std=[0.229])
            ])
            self.convert_mode = "L"
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.convert_mode = "RGB"

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    utt_id = parts[1]
                    label = 0 if parts[-1] == "bonafide" else 1
                    self.samples.append((utt_id, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, utt_id + ".png")

        if not os.path.exists(img_path):
            return torch.zeros((1 if self.convert_mode == "L" else 3, 224, 224)), torch.tensor(-1), utt_id

        img = Image.open(img_path).convert(self.convert_mode)
        spec = self.transform(img)

        return spec, torch.tensor(label), utt_id

def get_model(arch):
    if "EfficientNet" in arch:
        model = models.efficientnet_b0()
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif "ResNet" in arch:
        model = models.resnet18()
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 2)
        )
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch}")
    return model

def find_checkpoint(root_dir, model_name):
    for path in Path(root_dir).rglob(f"{model_name}_best.pth"):
        return str(path)
    return None

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold

def evaluate_metrics(y_true, scores, eer_threshold):
    y_pred = (scores >= eer_threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# ==========================================
# Classe Principal de Inferência
# ==========================================

class EnsemblePipeline:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Usando device: {self.device}")
        
        self.results_dir = Path("ensemble_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        (self.results_dir / "models_info").mkdir(exist_ok=True)
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        
        self.models_to_load = self._get_unique_models()
        self.model_predictions = {}
        self.ground_truth = {}
        self.loaded_checkpoints = {}
        
        self.asv_scores = None
        if em is not None and self.args.asv_scores_file:
            self._load_asv_scores()
            
    def _get_unique_models(self):
        models_set = set()
        for ensemble in ENSEMBLES:
            for model in ensemble['models']:
                models_set.add(model)
        return list(models_set)

    def _load_asv_scores(self):
        try:
            asv_data = np.genfromtxt(self.args.asv_scores_file, dtype=str)
            asv_keys = asv_data[:, 1]
            asv_scores = asv_data[:, 2].astype(float)
            
            self.asv_scores = {
                'tar': asv_scores[asv_keys == 'target'],
                'non': asv_scores[asv_keys == 'nontarget'],
                'spoof': asv_scores[asv_keys == 'spoof']
            }
            self.logger.info("Scores ASV carregados para cálculo de min-tDCF.")
        except Exception as e:
            self.logger.error(f"Falha ao carregar scores ASV: {e}")
            self.asv_scores = None

    def run_inference_for_model(self, model_name):
        checkpoint_path = find_checkpoint(self.args.checkpoints_dir, model_name)
        if not checkpoint_path:
            self.logger.error(f"Checkpoint não encontrado para {model_name}.")
            return False
            
        self.loaded_checkpoints[model_name] = checkpoint_path
        self.logger.info(f"Carregando {model_name} de {checkpoint_path}")
        
        arch = "EfficientNet-B0" if "EfficientNet" in model_name else "ResNet-18v2"
        feature = model_name.split('_')[1]
        
        img_dir = os.path.join(self.args.data_root, feature, "PA", "ASVspoof2019_PA_eval", "flac")
        
        try:
            dataset = ASVspoofCMDataset(img_dir, self.args.protocol_file, arch)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        except Exception as e:
            self.logger.error(f"Erro ao carregar dataset para {model_name}: {e}")
            return False
            
        model = get_model(arch)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        except Exception as e:
            self.logger.error(f"Erro ao carregar pesos de {checkpoint_path}: {e}")
            return False
            
        model.to(self.device)
        model.eval()
        
        scores_dict = {}
        
        start_time = time.time()
        valid_samples = 0
        
        with torch.no_grad():
            for x, y, utt in dataloader:
                valid_mask = y != -1
                if not valid_mask.any():
                    continue
                    
                x = x[valid_mask].to(self.device)
                y = y[valid_mask]
                utt = [u for u, v in zip(utt, valid_mask) if v]
                
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                spoof_scores = probs[:, 1].cpu().numpy()
                
                for u, s, l in zip(utt, spoof_scores, y.numpy()):
                    scores_dict[u] = float(s)
                    if u not in self.ground_truth:
                        self.ground_truth[u] = l
                        
                valid_samples += len(utt)
                
        elapsed = time.time() - start_time
        avg_time = (elapsed / valid_samples) * 1000 if valid_samples > 0 else 0
        
        self.logger.info(f"Inferência concluída para {model_name}: {valid_samples} amostras processadas em {elapsed:.2f}s ({avg_time:.2f} ms/amostra)")
        self.model_predictions[model_name] = scores_dict
        return True

    def calculate_ensemble_scores(self):
        if not self.model_predictions:
            self.logger.error("Nenhuma predição de modelo disponível.")
            return

        all_metrics = {}
        
        common_utts = set(self.ground_truth.keys())
        for preds in self.model_predictions.values():
            common_utts = common_utts.intersection(preds.keys())
            
        common_utts = sorted(list(common_utts))
        if not common_utts:
            self.logger.error("Nenhuma amostra comum encontrada entre as predições.")
            return
            
        y_true = np.array([self.ground_truth[u] for u in common_utts])
        bonafide_mask = y_true == 0
        spoof_mask = y_true == 1
        
        for ensemble in ENSEMBLES:
            eid = ensemble['id']
            models_list = ensemble['models']
            weights = ensemble['weights']
            
            missing_models = [m for m in models_list if m not in self.model_predictions]
            if missing_models:
                self.logger.warning(f"Ensemble {eid} pulado. Modelos faltando: {missing_models}")
                continue
                
            ensemble_scores = np.zeros(len(common_utts))
            
            for m, w in zip(models_list, weights):
                m_scores = np.array([self.model_predictions[m][u] for u in common_utts])
                ensemble_scores += w * m_scores
                
            eer, eer_th = compute_eer(y_true, ensemble_scores)
            metrics_dict = evaluate_metrics(y_true, ensemble_scores, eer_th)
            
            min_tdcf = float('inf')
            if self.asv_scores is not None and em is not None:
                try:
                    tar_asv, non_asv, spoof_asv = self.asv_scores['tar'], self.asv_scores['non'], self.asv_scores['spoof']
                    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = em.obtain_asv_error_rates(
                        tar_asv, non_asv, spoof_asv,
                        em.compute_eer(tar_asv, non_asv)[1]
                    )
                    
                    tDCF_curve, _ = em.compute_tDCF(
                        ensemble_scores[bonafide_mask], ensemble_scores[spoof_mask],
                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                        COST_MODEL, print_cost=False
                    )
                    min_tdcf = np.min(tDCF_curve)
                except Exception as e:
                    self.logger.error(f"Erro no t-DCF para Ensemble {eid}: {e}")
            
            metrics_dict.update({
                "eer_percent": float(eer * 100),
                "eer_threshold": float(eer_th),
                "min_tdcf": float(min_tdcf)
            })
            
            all_metrics[f"Ensemble_{eid:02d}"] = metrics_dict
            
            self.logger.info(f"\n--- Resultados Ensemble {eid} ---")
            self.logger.info(f"Modelos: {models_list}")
            self.logger.info(f"Pesos: {weights}")
            self.logger.info(f"EER: {eer*100:.4f}%")
            if min_tdcf != float('inf'):
                self.logger.info(f"min-tDCF: {min_tdcf:.6f}")
                
            preds_df = pd.DataFrame({
                'utt_id': common_utts,
                'label': y_true,
                'score': ensemble_scores
            })
            preds_df.to_csv(self.results_dir / "predictions" / f"Ensemble_{eid:02d}_scores.csv", index=False)
            
            with open(self.results_dir / "metrics" / f"Ensemble_{eid:02d}_metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)
                
        with open(self.results_dir / "models_info" / "loaded_models.json", "w") as f:
            json.dump({
                "total_samples": len(common_utts),
                "checkpoints": self.loaded_checkpoints
            }, f, indent=4)

    def execute(self):
        self.logger.info("Iniciando pipeline de inferência de ensemble...")
        for model in self.models_to_load:
            self.run_inference_for_model(model)
        self.calculate_ensemble_scores()
        self.logger.info("Pipeline concluído. Resultados salvos em 'ensemble_results'.")

def main():
    parser = argparse.ArgumentParser(description="Inference para Ensembles - ASVspoof 2019")
    parser.add_argument("--data_root", type=str, default="/home/henrique/pibic/data-set-asv", help="Diretório raiz dos datasets gerados")
    parser.add_argument("--protocol_file", type=str, default="PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt", help="Caminho para o protocolo de eval")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Diretório para buscar pesos .pth")
    parser.add_argument("--asv_scores_file", type=str, default="PA_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt", help="Scores ASV para min-tDCF")
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    os.makedirs("ensemble_results/logs", exist_ok=True)
    logger = setup_logger("ensemble_results/logs/execution.log")
    
    with open("ensemble_results/logs/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        
    pipeline = EnsemblePipeline(args, logger)
    pipeline.execute()

if __name__ == "__main__":
    main()
