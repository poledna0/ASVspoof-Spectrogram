# Top 10 Ensembles Reais (Checkpoint + Score Fusion)

## Como foi feito

- Modelos considerados: apenas modelos com `*_best.pth` em `checkpoints/` e scores DEV/EVAL válidos em `scores/`.
- Total de modelos válidos: 8.
- Geração de ensembles: combinações de 4 modelos (70 combinações), com busca de pesos via Dirichlet + peso uniforme.
- Seleção Top 10: ranking por `min-tDCF` no DEV.
- Avaliação final: EVAL com métricas oficiais (`eval_metrics.py`) e validação com script oficial `tdcf/evaluate_tDCF_asvspoof19.py`.

## Ranking final no EVAL

1. ensemble_04 | EER=0.7839% | min-tDCF=0.020921
2. ensemble_01 | EER=0.9224% | min-tDCF=0.022189
3. ensemble_07 | EER=1.1053% | min-tDCF=0.024164
4. ensemble_05 | EER=1.1995% | min-tDCF=0.025511
5. ensemble_03 | EER=1.2106% | min-tDCF=0.025739
6. ensemble_02 | EER=1.1843% | min-tDCF=0.026385
7. ensemble_06 | EER=1.3385% | min-tDCF=0.027966
8. ensemble_08 | EER=1.0720% | min-tDCF=0.028229
9. ensemble_09 | EER=1.2204% | min-tDCF=0.028561
10. ensemble_10 | EER=1.2492% | min-tDCF=0.029001

## Por que o top 1 venceu

- `ensemble_04` combina 3 visões EfficientNet (cqcc, logmel, stft) com 1 visão ResNet (cqt), reduzindo sobreposição de erro entre arquiteturas.
- Os pesos não ficaram concentrados em um único modelo, o que reduz risco de overfitting de fusão.
- Teve boa consistência DEV->EVAL: já era forte no DEV e manteve liderança no EVAL.

## Artefatos importantes

- Score final para submissão tDCF: `final/CM_EVAL_TOP1.txt`
- Configuração e métricas do melhor ensemble: `final/best_ensemble.json`
- Tabela consolidada dos Top 10: `top10/summary.csv`
- Saída oficial do script tDCF por ensemble: `top10/ensemble_xx/official_tdcf_stdout.txt`
- Log completo da execução: `logs/pipeline.log`
