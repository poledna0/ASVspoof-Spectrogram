# Top 10 Ensembles Reais (Base DEV dos Checkpoints)

## Fonte dos scores

- Base de seleção (DEV): scores dentro das subpastas de `checkpoints/runs-Jun-1/*/scores/`.
- Avaliação final (EVAL): `scores/<arquitetura>/<feature>_EVAL_scores.txt`.
- Origem detalhada por modelo: `metadata/score_sources.json`.

## Melhor ensemble final

- ID: `ensemble_07`
- Modelos: `EfficientNet-B0_cqcc + EfficientNet-B0_logmel + ResNet-18v2_cqt + ResNet-18v2_stft`
- EER (EVAL): `0.9122%`
- min-tDCF (EVAL): `0.024222`
- Score final tDCF: `final/CM_EVAL_TOP1.txt`

## Artefatos principais

- `top10/summary.csv`
- `top10/ensemble_07/metrics.json`
- `top10/ensemble_07/official_tdcf_stdout.txt`
- `final/best_ensemble.json`
- `final/CM_EVAL_TOP1.txt`
