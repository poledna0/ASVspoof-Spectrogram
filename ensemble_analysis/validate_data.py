#!/usr/bin/env python3
"""
Script de validação dos dados
Verifica formato, alinhamento, e integridade dos scores
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'tdcf'))

def validate_workspace():
    """Valida estrutura do workspace"""
    workspace = Path(__file__).parent
    
    print("="*70)
    print("VALIDAÇÃO DO WORKSPACE")
    print("="*70)
    
    required_dirs = [
        'scores/EfficientNet-B0',
        'scores/ResNet-18v2',
        'PA_cm_protocols',
        'PA_scores',
        'tdcf'
    ]
    
    for dir_path in required_dirs:
        full_path = workspace / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
    
    # Verificar ASV scores
    asv_file = workspace / 'PA_scores' / 'ASVspoof2019.PA.asv.eval.gi.trl.scores.txt'
    if asv_file.exists():
        print(f"✓ ASV scores file exists")
    else:
        print(f"✗ ASV scores file NOT FOUND: {asv_file}")
        return False
    
    return True


def validate_scores():
    """Valida integridade dos scores"""
    workspace = Path(__file__).parent
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DE SCORES")
    print("="*70)
    
    models_info = {
        'EfficientNet-B0': ['cqcc', 'cqt', 'lfcc', 'logmel', 'stft'],
        'ResNet-18v2': ['cqcc', 'cqt', 'logmel', 'stft'],
    }
    
    all_utts = None
    
    for model, features in models_info.items():
        print(f"\n{model}:")
        
        model_utts = None
        for feature in features:
            score_file = workspace / 'scores' / model / f'{feature}_EVAL_scores.txt'
            
            if not score_file.exists():
                print(f"  ✗ {feature}: arquivo não encontrado")
                continue
            
            scores_df = pd.read_csv(score_file, sep=' ', header=None, names=['utt_id', 'score'])
            
            print(f"  ✓ {feature:10s}: {len(scores_df):5d} samples, "
                  f"score range=[{scores_df['score'].min():.4f}, {scores_df['score'].max():.4f}]")
            
            if model_utts is None:
                model_utts = set(scores_df['utt_id'])
            else:
                model_utts = model_utts.intersection(set(scores_df['utt_id']))
        
        if model_utts is not None:
            print(f"  → Total common UTTs in {model}: {len(model_utts)}")
            
            if all_utts is None:
                all_utts = model_utts
            else:
                all_utts = all_utts.intersection(model_utts)
    
    if all_utts:
        print(f"\n✓ Todos os modelos têm scores para {len(all_utts)} amostras comuns")
    
    return True


def validate_protocol():
    """Valida protocolo"""
    workspace = Path(__file__).parent
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DE PROTOCOLO")
    print("="*70)
    
    protocol_file = workspace / 'PA_cm_protocols' / 'ASVspoof2019.PA.cm.eval.trl.txt'
    
    if not protocol_file.exists():
        print(f"✗ Protocolo não encontrado: {protocol_file}")
        return False
    
    protocol = pd.read_csv(
        protocol_file,
        sep=' ',
        header=None,
        names=['speaker', 'utt_id', 'environment', 'attack', 'label']
    )
    
    print(f"✓ Protocolo carregado: {len(protocol)} samples")
    
    # Estatísticas
    label_counts = protocol['label'].value_counts()
    print(f"\n  Labels:")
    for label, count in label_counts.items():
        print(f"    - {label}: {count} ({count/len(protocol)*100:.1f}%)")
    
    print(f"\n  Ambientes únicos: {protocol['environment'].nunique()}")
    print(f"  Ataques únicos: {protocol['attack'].nunique()}")
    print(f"  Speakers únicos: {protocol['speaker'].nunique()}")
    
    return True


def validate_asv_scores():
    """Valida scores ASV"""
    workspace = Path(__file__).parent
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DE ASV SCORES")
    print("="*70)
    
    asv_file = workspace / 'PA_scores' / 'ASVspoof2019.PA.asv.eval.gi.trl.scores.txt'
    
    asv_data = np.genfromtxt(asv_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)
    
    print(f"✓ ASV scores carregados: {len(asv_scores)} samples")
    
    for key in ['target', 'nontarget', 'spoof']:
        count = (asv_keys == key).sum()
        key_scores = asv_scores[asv_keys == key]
        print(f"\n  {key}:")
        print(f"    - Count: {count}")
        print(f"    - Range: [{key_scores.min():.4f}, {key_scores.max():.4f}]")
        print(f"    - Mean: {key_scores.mean():.4f}, Std: {key_scores.std():.4f}")
    
    return True


def test_tdcf_import():
    """Testa se eval_metrics pode ser importado"""
    workspace = Path(__file__).parent
    sys.path.insert(0, str(workspace / 'tdcf'))
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DE MÓDULO TDCF")
    print("="*70)
    
    try:
        import eval_metrics as em
        print("✓ eval_metrics importado com sucesso")
        
        # Test básico
        tar_scores = np.array([1.0, 2.0, 3.0])
        non_scores = np.array([0.0, 0.5, 0.8])
        
        eer, threshold = em.compute_eer(tar_scores, non_scores)
        print(f"✓ compute_eer funciona: EER={eer:.4f}, threshold={threshold:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Erro ao importar eval_metrics: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("VALIDAÇÃO COMPLETA DO WORKSPACE")
    print("="*70)
    
    checks = [
        ("Estrutura", validate_workspace),
        ("Scores", validate_scores),
        ("Protocolo", validate_protocol),
        ("ASV Scores", validate_asv_scores),
        ("TDCF Module", test_tdcf_import),
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n✗ Erro durante validação de {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ VALIDAÇÃO COMPLETA - Tudo OK!")
        print("Você pode agora executar: python run_ensemble_analysis.py")
    else:
        print("✗ Alguns testes falharam - verifique os erros acima")
    print("="*70)


if __name__ == '__main__':
    main()
