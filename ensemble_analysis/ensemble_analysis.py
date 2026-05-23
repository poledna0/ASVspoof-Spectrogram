
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

# Importar eval_metrics do diretório tdcf
sys.path.insert(0, str(Path(__file__).parent / 'tdcf'))
import eval_metrics as em


@dataclass
class EnsembleResult:
    models: List[str]
    weights: List[float]
    eer: float
    eer_threshold: float
    min_tdcf: float
    min_tdcf_threshold: float
    correlation_avg: float
    error_disagreement: float  # Quão bem se complementam
    description: str = ""


class EnsembleAnalyzer:
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.scores_root = self.workspace_root / 'scores'
        self.protocols_root = self.workspace_root / 'PA_cm_protocols'
        self.asv_scores_file = self.workspace_root / 'PA_scores' / 'ASVspoof2019.PA.asv.eval.gi.trl.scores.txt'
        
        # Configuração do t-DCF (oficial ASVspoof 2019)
        self.cost_model = {
            'Pspoof': 0.05,
            'Ptar': (1 - 0.05) * 0.99,
            'Pnon': (1 - 0.05) * 0.01,
            'Cmiss_asv': 1,
            'Cfa_asv': 10,
            'Cmiss_cm': 1,
            'Cfa_cm': 10,
        }
        
        self.results = []
        self.loaded_scores = {}
        self.protocol = None
        self.asv_data = None
        
    def load_protocol(self, split: str = 'eval') -> pd.DataFrame:
        protocol_file = self.protocols_root / f'ASVspoof2019.PA.cm.{split}.trl.txt'
        
        protocol = pd.read_csv(
            protocol_file,
            sep=' ',
            header=None,
            names=['speaker', 'utt_id', 'environment', 'attack', 'label']
        )
        
        print(f"✓ Protocolo carregado: {len(protocol)} samples ({split.upper()})")
        return protocol
    
    def load_model_scores(self, model: str, feature: str, split: str = 'EVAL') -> pd.DataFrame:
        """Carrega scores de um modelo+feature específico"""
        if self.protocol is None:
            self.protocol = self.load_protocol('eval')
        
        score_file = self.scores_root / model / f'{feature}_{split}_scores.txt'
        
        if not score_file.exists():
            print(f"Arquivo não encontrado: {score_file}")
            return None
        
        scores_df = pd.read_csv(
            score_file,
            sep=' ',
            header=None,
            names=['utt_id', 'score']
        )
        
        merged = self.protocol[['utt_id', 'label']].merge(scores_df, on='utt_id', how='inner')
        if merged.empty:
            print(f"Falha ao alinhar protocolo com scores do modelo {model} {feature}")
            return None
        
        mean_bonafide = merged.loc[merged['label'] == 'bonafide', 'score'].mean()
        mean_spoof = merged.loc[merged['label'] == 'spoof', 'score'].mean()
        spoof_higher = mean_bonafide < mean_spoof
        
        scores_df['score'] = self._normalize_scores(scores_df['score'].values, spoof_higher)
        
        return scores_df
    
    def _normalize_scores(self, scores: np.ndarray, spoof_higher: bool) -> np.ndarray:

        if spoof_higher:
            if np.all((scores >= 0) & (scores <= 1)):
                normalized = 1.0 - scores
            else:
                normalized = 1.0 / (1.0 + np.exp(scores))
        else:
            if np.all((scores >= 0) & (scores <= 1)):
                normalized = scores
            else:
                normalized = 1.0 / (1.0 + np.exp(-scores))
        return normalized
    
    def load_all_models(self, split: str = 'EVAL') -> Dict:
        models_info = {
            'EfficientNet-B0': ['cqcc', 'cqt', 'lfcc', 'logmel', 'stft'],
            'ResNet-18v2': ['cqcc', 'cqt', 'logmel', 'stft'],
        }
        
        all_scores = {}
        
        for model, features in models_info.items():
            print(f"\n{'='*60}")
            print(f"Carregando {model}")
            print(f"{'='*60}")
            
            model_scores = {}
            for feature in features:
                scores_df = self.load_model_scores(model, feature, split)
                if scores_df is not None:
                    model_scores[feature] = scores_df
                    print(f"  ✓ {feature:10s}: {len(scores_df)} scores")
            
            all_scores[model] = model_scores
        
        self.loaded_scores = all_scores
        return all_scores
    
    def merge_scores_with_protocol(self, split: str = 'eval') -> pd.DataFrame:
        self.protocol = self.load_protocol(split)
        
        # Começar com protocolo
        merged = self.protocol.copy()
        
        # Agregar scores de cada modelo+feature
        for model_name, model_features in self.loaded_scores.items():
            for feature, scores_df in model_features.items():
                col_name = f'{model_name}_{feature}'
                
                # Merge
                merged = merged.merge(
                    scores_df,
                    on='utt_id',
                    how='left'
                )
                merged.rename(columns={'score': col_name}, inplace=True)
        
        # Verificar se há NaNs (amostra que não tem scores)
        nan_count = merged.isna().sum().sum()
        if nan_count > 0:
            print(f"\n⚠ Aviso: {nan_count} valores NaN encontrados (amostras sem scores)")
            print("Descartando linhas incompletas...")
            merged = merged.dropna()
        
        print(f"\n✓ Merged dataset: {len(merged)} samples com scores completos")
        return merged
    
    def compute_complementarity_matrix(self, scores_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        score_cols = [col for col in scores_df.columns 
                      if col not in ['speaker', 'utt_id', 'environment', 'attack', 'label']]
        
        correlation_matrix = pd.DataFrame(
            np.zeros((len(score_cols), len(score_cols))),
            index=score_cols,
            columns=score_cols
        )
        
        error_disagreement = pd.DataFrame(
            np.zeros((len(score_cols), len(score_cols))),
            index=score_cols,
            columns=score_cols
        )
        
        # Para cada par de modelos
        for i, col1 in enumerate(score_cols):
            for j, col2 in enumerate(score_cols):
                if i == j:
                    correlation_matrix.iloc[i, j] = 1.0
                    error_disagreement.iloc[i, j] = 0.0
                else:
                    # Correlação de scores
                    corr = np.corrcoef(scores_df[col1], scores_df[col2])[0, 1]
                    correlation_matrix.iloc[i, j] = corr if not np.isnan(corr) else 0
                    
                    # Erro de cada modelo (usando threshold 0.5)
                    threshold = 0.5
                    errors1 = (scores_df[col1] >= threshold) != (scores_df['label'] == 'spoof')
                    errors2 = (scores_df[col2] >= threshold) != (scores_df['label'] == 'spoof')
                    
                    # Desacordo = quando um erra e outro acerta
                    disagreement = (errors1 != errors2).sum() / len(scores_df)
                    error_disagreement.iloc[i, j] = disagreement
        
        return correlation_matrix, error_disagreement
    
    def test_ensemble(self, scores_df: pd.DataFrame, 
                     model_features: List[str], 
                     weights: List[float]) -> EnsembleResult:
        
        # Validar entrada
        if len(model_features) != len(weights):
            raise ValueError("model_features e weights devem ter o mesmo tamanho")
        
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError(f"Pesos devem somar 1.0, somou {sum(weights)}")
        
        # Combinar scores via weighted average
        ensemble_score = np.zeros(len(scores_df))
        for feature, weight in zip(model_features, weights):
            if feature not in scores_df.columns:
                raise ValueError(f"Feature {feature} não encontrada")
            ensemble_score += weight * scores_df[feature].values
        
        # Separar por label
        bonafide_mask = scores_df['label'] == 'bonafide'
        spoof_mask = scores_df['label'] == 'spoof'
        
        bonafide_scores = ensemble_score[bonafide_mask]
        spoof_scores = ensemble_score[spoof_mask]
        
        # Computar EER
        try:
            eer, eer_threshold = em.compute_eer(bonafide_scores, spoof_scores)
        except Exception as e:
            print(f"Erro ao computar EER: {e}")
            return None
        
        # Computar t-DCF
        try:
            # Precisamos de ASV scores para computar t-DCF
            # Usar dados do ASV score file
            tar_asv, non_asv, spoof_asv = self._load_asv_scores()
            Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = em.obtain_asv_error_rates(
                tar_asv, non_asv, spoof_asv,
                em.compute_eer(tar_asv, non_asv)[1]
            )
            
            tDCF_curve, CM_thresholds = em.compute_tDCF(
                bonafide_scores, spoof_scores,
                Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                self.cost_model,
                print_cost=False
            )
            
            min_tdcf_idx = np.argmin(tDCF_curve)
            min_tdcf = tDCF_curve[min_tdcf_idx]
            min_tdcf_threshold = CM_thresholds[min_tdcf_idx]
        except Exception as e:
            print(f"Erro ao computar t-DCF: {e}")
            min_tdcf = float('inf')
            min_tdcf_threshold = 0.5
        
        # Calcular complementaridade média
        score_cols = model_features
        correlation_sum = 0
        disagreement_sum = 0
        
        pair_count = 0
        for i in range(len(score_cols)):
            for j in range(i+1, len(score_cols)):
                col1, col2 = score_cols[i], score_cols[j]
                
                # Correlação
                corr = np.corrcoef(scores_df[col1], scores_df[col2])[0, 1]
                if not np.isnan(corr):
                    correlation_sum += corr
                
                # Desacordo de erro
                threshold = 0.5
                errors1 = (scores_df[col1] >= threshold) != (scores_df['label'] == 'spoof')
                errors2 = (scores_df[col2] >= threshold) != (scores_df['label'] == 'spoof')
                disagreement = (errors1 != errors2).sum() / len(scores_df)
                disagreement_sum += disagreement
                
                pair_count += 1
        
        correlation_avg = correlation_sum / pair_count if pair_count > 0 else 0
        error_disagreement_avg = disagreement_sum / pair_count if pair_count > 0 else 0
        
        result = EnsembleResult(
            models=model_features,
            weights=weights,
            eer=eer,
            eer_threshold=eer_threshold,
            min_tdcf=min_tdcf,
            min_tdcf_threshold=min_tdcf_threshold,
            correlation_avg=correlation_avg,
            error_disagreement=error_disagreement_avg
        )
        
        return result
    
    def _load_asv_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.asv_data is None:
            asv_data = np.genfromtxt(
                self.asv_scores_file,
                dtype=str
            )
            asv_sources = asv_data[:, 0]
            asv_keys = asv_data[:, 1]
            asv_scores = asv_data[:, 2].astype(float)
            
            tar_asv = asv_scores[asv_keys == 'target']
            non_asv = asv_scores[asv_keys == 'nontarget']
            spoof_asv = asv_scores[asv_keys == 'spoof']
            
            self.asv_data = (tar_asv, non_asv, spoof_asv)
        
        return self.asv_data
    
    def search_best_ensemble_combinations(self, scores_df: pd.DataFrame,
                                         max_models: int = 4,
                                         num_weight_combinations: int = 100) -> List[EnsembleResult]:

        
        score_cols = [col for col in scores_df.columns 
                      if col not in ['speaker', 'utt_id', 'environment', 'attack', 'label']]
        
        results = []
        
        print(f"\n{'='*70}")
        print("BUSCA DE MELHORES ENSEMBLES")
        print(f"{'='*70}")
        print(f"Modelos disponíveis: {len(score_cols)}")
        print(f"Testando: 2 a {min(max_models, len(score_cols))} modelos por ensemble")
        
        for n_models in range(2, min(max_models + 1, len(score_cols) + 1)):
            print(f"\n--- Testando combinações com {n_models} modelos ---")
            
            # Todas as combinações de n_models
            for model_combo in combinations(score_cols, n_models):
                
                # Gerar combinações de pesos
                weight_combos = self._generate_weight_combinations(
                    n_models, num_weight_combinations
                )
                
                best_for_combo = None
                best_tdcf = float('inf')
                
                for weights in weight_combos:
                    try:
                        result = self.test_ensemble(scores_df, list(model_combo), list(weights))
                        
                        if result and result.min_tdcf < best_tdcf:
                            best_tdcf = result.min_tdcf
                            best_for_combo = result
                    except Exception as e:
                        continue
                
                if best_for_combo:
                    results.append(best_for_combo)
                    print(f"  {' + '.join([m.split('_')[1] for m in model_combo]):30s} "
                          f"| EER={best_for_combo.eer*100:6.3f}% | min-tDCF={best_for_combo.min_tdcf:.6f} | "
                          f"CompScore={1-best_for_combo.correlation_avg:.3f}")
        
        # Ranquear por min-tDCF
        results.sort(key=lambda x: x.min_tdcf)
        
        return results
    
    def _generate_weight_combinations(self, n_models: int, 
                                     num_combinations: int) -> List[List[float]]:
        """Gera combinações de pesos que somam 1.0"""
        
        combinations_list = []
        
        # Caso especial: 2 modelos - testar incrementos de 10%
        if n_models == 2:
            for w1 in np.linspace(0.0, 1.0, 11):
                w2 = 1.0 - w1
                combinations_list.append([w1, w2])
        
        # Caso geral: amostragem aleatória
        else:
            for _ in range(num_combinations):
                # Usar Dirichlet para gerar pesos que somam 1
                weights = np.random.dirichlet(np.ones(n_models))
                combinations_list.append(list(weights))
        
        return combinations_list
    
    def print_complementarity_report(self, correlation_matrix: pd.DataFrame,
                                     error_disagreement: pd.DataFrame):
        """Imprime relatório de complementaridade entre modelos"""
        
        print(f"\n{'='*70}")
        print("ANÁLISE DE COMPLEMENTARIDADE ENTRE MODELOS")
        print(f"{'='*70}")
        
        print("\n1. CORRELAÇÃO DE SCORES (mais próximo de 0 = melhor)")
        print("   (Modelos com baixa correlação tendem a capturar aspectos diferentes)")
        print(correlation_matrix.to_string())
        
        print("\n\n2. DESACORDO DE ERROS (mais próximo de 1 = melhor)")
        print("   (Quando um modelo erra, o outro acerta?)")
        print(error_disagreement.to_string())
        
        print("\n\n3. PARES COM MELHOR COMPLEMENTARIDADE")
        
        # Calcular score de complementaridade
        complementarity_pairs = []
        
        score_cols = correlation_matrix.columns.tolist()
        for i in range(len(score_cols)):
            for j in range(i+1, len(score_cols)):
                col1, col2 = score_cols[i], score_cols[j]
                
                # Score = baixa correlação + alto desacordo
                comp_score = (1 - abs(correlation_matrix.loc[col1, col2])) * error_disagreement.loc[col1, col2]
                
                complementarity_pairs.append({
                    'model1': col1,
                    'model2': col2,
                    'correlation': correlation_matrix.loc[col1, col2],
                    'error_disagreement': error_disagreement.loc[col1, col2],
                    'complementarity_score': comp_score
                })
        
        # Ordenar por complementarity score
        complementarity_pairs.sort(key=lambda x: x['complementarity_score'], reverse=True)
        
        print("\n   Ranking (por complementaridade):")
        for i, pair in enumerate(complementarity_pairs[:10], 1):
            print(f"   {i}. {pair['model1']:30s} + {pair['model2']:30s}")
            print(f"      Correlação={pair['correlation']:7.4f}, Desacordo={pair['error_disagreement']:.4f}, "
                  f"ComplementScore={pair['complementarity_score']:.4f}")
    
    def print_top_ensembles(self, results: List[EnsembleResult], top_n: int = 10):
        """Imprime os top-N melhores ensembles"""
        
        print(f"\n{'='*90}")
        print(f"TOP {top_n} MELHORES ENSEMBLES (Ranqueados por min-tDCF)")
        print(f"{'='*90}")
        
        for i, result in enumerate(results[:top_n], 1):
            model_names = [m.split('_')[1] for m in result.models]
            print(f"\n{i}. Ensemble: {' + '.join(model_names)}")
            print(f"   Modelos:         {result.models}")
            print(f"   Pesos:           {[f'{w:.3f}' for w in result.weights]}")
            print(f"   EER:             {result.eer*100:.4f}%")
            print(f"   min-tDCF:        {result.min_tdcf:.6f}")
            print(f"   Correlação Avg:  {result.correlation_avg:.4f}")
            print(f"   Desacordo Erro:  {result.error_disagreement:.4f} (complementaridade)")
    
    def run_full_analysis(self):
        """Executa análise completa"""
        
        print("="*70)
        print("FRAMEWORK DE ENSEMBLE PARA ASVSPOOOF 2019")
        print("Análise de Complementaridade + Weighted Score Fusion")
        print("="*70)
        
        # 1. Carregar dados
        print("\n[1/5] Carregando modelos e scores...")
        self.load_all_models()
        
        # 2. Merge com protocolo
        print("\n[2/5] Merging scores com protocolo...")
        scores_df = self.merge_scores_with_protocol()
        
        # 3. Análise de complementaridade
        print("\n[3/5] Análise de complementaridade entre modelos...")
        corr_matrix, err_disagreement = self.compute_complementarity_matrix(scores_df)
        self.print_complementarity_report(corr_matrix, err_disagreement)
        
        # 4. Busca de melhores ensembles
        print("\n[4/5] Busca de melhores combinações de ensemble...")
        results = self.search_best_ensemble_combinations(scores_df, max_models=4, num_weight_combinations=100)
        
        # 5. Relatório final
        print("\n[5/5] Gerando relatório final...")
        self.print_top_ensembles(results, top_n=15)
        
        # Salvar resultados em JSON
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[EnsembleResult]):
        """Salva resultados em arquivo JSON"""
        output_file = self.workspace_root / 'ensemble_results.json'
        
        results_dict = []
        for r in results:
            results_dict.append({
                'models': r.models,
                'weights': r.weights,
                'eer': float(r.eer),
                'eer_threshold': float(r.eer_threshold),
                'min_tdcf': float(r.min_tdcf),
                'min_tdcf_threshold': float(r.min_tdcf_threshold),
                'correlation_avg': float(r.correlation_avg),
                'error_disagreement': float(r.error_disagreement),
            })
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Resultados salvos em: {output_file}")


if __name__ == '__main__':
    workspace_root = str(Path(__file__).parent)
    analyzer = EnsembleAnalyzer(workspace_root)
    results = analyzer.run_full_analysis()
