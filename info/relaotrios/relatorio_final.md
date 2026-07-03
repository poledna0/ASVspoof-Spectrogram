# RELATÓRIO FINAL

## ABORDAGENS DEEP LEARNING PARA MITIGAÇÃO DE ATAQUES POR REPLAY EM SISTEMAS DE AUTENTICAÇÃO BIOMÉTRICA

Relatório Final apresentado ao Programa Institucional de Bolsas de Iniciação Científica e Tecnológica, Pró-Reitoria de Pesquisa, Pós-Graduação e Inovação da Pontifícia Universidade Católica do Paraná.

Curso: CiberSegurança  
Modalidade: PIBIC  
Instituição: Pontifícia Universidade Católica do Paraná  
Cidade: Curitiba  
Ano: 2026

---

## RESUMO

Este trabalho investigou contramedidas de Deep Learning para detecção de ataques de replay no cenário Physical Access (PA) da base ASVspoof 2019, com ênfase na comparação entre representações espectrais, arquiteturas convolucionais e fusão por ensemble com avaliação oficial por min-tDCF. O problema de pesquisa foi estruturado a partir da necessidade de aumentar a robustez de sistemas de autenticação biométrica por voz frente a ataques físicos de reprodução de áudio, mantendo capacidade de generalização para condições não vistas no treinamento. Metodologicamente, o projeto contemplou: (i) extração sistemática de espectrogramas em múltiplas representações (STFT, Log-Mel, CQT, LFCC, CQCC); (ii) treinamento de modelos EfficientNet-B0 e ResNet-18v2 com early stopping e ajuste de taxa de aprendizado; (iii) geração e padronização de arquivos de score para os conjuntos DEV e EVAL; (iv) correção de problemas de extração causados por arquivos FLAC com inconsistências de frame; e (v) construção de pipeline de ensemble ranqueado por min-tDCF, com validação no script oficial ASVspoof. Os resultados individuais indicaram melhor desempenho para representações espectrais completas, destacando-se EfficientNet-B0 Log-Mel (EER EVAL = 1,54%) e ResNet-18v2 STFT (EER EVAL = 1,45%; min-tDCF = 0,039776). Em fase avançada do projeto, a estratégia de ensemble de 4 modelos, selecionada em DEV e validada em EVAL com scores no formato oficial CM, atingiu desempenho superior ao dos modelos individuais: o melhor ensemble (EfficientNet-B0_cqcc + EfficientNet-B0_logmel + ResNet-18v2_cqt + ResNet-18v2_stft) obteve EER EVAL de 0,9122% e min-tDCF de 0,024222. A análise consolidada mostra que a complementaridade entre arquiteturas e representações reduz erro em relação ao uso isolado de uma única feature. Conclui-se que o objetivo de desenvolver contramedidas mais robustas para replay foi atingido, com entrega de pipeline reprodutível, score final compatível com avaliação oficial e documentação técnica consolidada para continuidade em nível de pesquisa aplicada.

**Palavras-chave:** replay attack; ASVspoof 2019; deep learning; espectrogramas; ensemble; min-tDCF.

---

## LISTA DE FIGURAS

1. Figura 1. Exemplo de espectrograma STFT do arquivo de teste PA_T_0000001, utilizado para validação visual da etapa de pré-processamento. Fonte: [espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_01_STFT.png](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_01_STFT.png).
2. Figura 2. Exemplo de espectrograma Log-Mel do mesmo arquivo de teste, evidenciando redistribuição de energia em escala perceptual. Fonte: [espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_03_LogMel.png](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_03_LogMel.png).
3. Figura 3. Exemplo de representação Cochleagrama/CQT no conjunto de teste de espectrogramas. Fonte: [espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_04_Cochleagrama.png](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_04_Cochleagrama.png).
4. Figura 4. Esquema de cenário PA e figura de referência metodológica do relatório parcial (inserir no Word a partir do PDF). Fonte: [info/relaotrios/relatorio-parcial.pdf](info/relaotrios/relatorio-parcial.pdf).
5. Figura 5. Arquitetura simplificada da EfficientNet-B0 utilizada no projeto (inserir no Word a partir do PDF parcial). Fonte: [info/relaotrios/relatorio-parcial.pdf](info/relaotrios/relatorio-parcial.pdf).

---

## LISTA DE TABELAS

1. Tabela 1. Distribuição de amostras por partição no ASVspoof 2019 PA.
2. Tabela 2. Configuração de treinamento dos modelos.
3. Tabela 3. Resultados históricos iniciais (resultadoextremo).
4. Tabela 4. Resultados individuais finais (DEV/EVAL) dos modelos principais.
5. Tabela 5. Resultados oficiais de min-tDCF para arquivos históricos de score (make-scores).
6. Tabela 6. Ranking dos Top 10 ensembles reais no EVAL.
7. Tabela 7. Informações de infraestrutura e artefatos de execução.
8. Tabela 8. Trabalhos relacionados e relação com o projeto.

---

## LISTA DE ABREVIATURAS E SIGLAS

- ASV: Automatic Speaker Verification
- CM: Countermeasure
- DEV: Development set
- EER: Equal Error Rate
- EVAL: Evaluation set
- LFCC: Linear Frequency Cepstral Coefficients
- CQCC: Constant-Q Cepstral Coefficients
- CQT: Constant-Q Transform
- STFT: Short-Time Fourier Transform
- t-DCF: tandem Detection Cost Function
- PA: Physical Access

---

## SUMÁRIO

1. Introdução  
2. Objetivos  
2.1 Objetivo Geral  
2.2 Objetivos Específicos  
3. Material e Métodos  
4. Resultados  
5. Discussão  
6. Considerações Finais  
6.1 Recomendações para Trabalhos Futuros  
7. Uso de Inteligência Artificial Generativa  
8. Outras Atividades Realizadas  
Referências  
Anexo A. Inventário técnico dos artefatos

---

## 1 INTRODUÇÃO

A autenticação biométrica por voz tornou-se componente relevante em aplicações de segurança digital, mas sua adoção amplia a superfície de ataque para fraudes de spoofing. No cenário de acesso físico, ataques de replay simulam tentativas legítimas por meio de gravação e reprodução em ambiente real, incorporando efeitos de canal, reverberação e características de dispositivo de reprodução. Esse tipo de ataque representa risco operacional significativo para sistemas de ASV.

No contexto científico, o desafio ASVspoof consolidou protocolos, dados e métricas para avaliação de contramedidas, com evolução explícita para cenários mais realistas e para avaliação conjunta com ASV por meio da métrica t-DCF. O plano oficial de avaliação ASVspoof 2019 enfatiza generalização para ataques não vistos e avaliação de impacto real da contramedida em cascata com ASV.

Este projeto partiu dessa motivação e desenvolveu um pipeline completo no domínio PA da base ASVspoof 2019, cobrindo extração de representações espectrais, treinamento de modelos convolucionais, geração de scores padronizados, avaliação por EER e min-tDCF, e fusão por ensemble orientada por métrica oficial. A abordagem foi guiada por evidências internas do projeto (scripts, logs, scores e resultados finais) e por literatura técnica disponibilizada no repositório.

---

## 2 OBJETIVOS

### 2.1 OBJETIVO GERAL

Desenvolver e avaliar contramedidas baseadas em Deep Learning para detecção de ataques de replay em autenticação biométrica por voz, visando reduzir EER e min-tDCF no protocolo ASVspoof 2019 PA.

### 2.2 OBJETIVOS ESPECÍFICOS

1. Implementar pipeline de extração de representações espectrais para dados PA.
2. Treinar e comparar arquiteturas EfficientNet-B0 e ResNet-18v2 em múltiplas features.
3. Produzir e validar scores DEV/EVAL compatíveis com protocolo oficial.
4. Corrigir inconsistências de extração e padronizar processamento de arquivos problemáticos.
5. Avaliar desempenho com métricas EER e min-tDCF.
6. Desenvolver ensemble de modelos com fusão por pesos e seleção dos melhores arranjos.
7. Consolidar resultados em formato reprodutível para análise técnica e submissão.

---

## 3 MATERIAL E MÉTODOS

### 3.1 Base de dados e protocolo

Foi utilizado o ASVspoof 2019 no cenário Physical Access, com partições train/dev/eval, conforme protocolos em [PA_cm_protocols](PA_cm_protocols).

**Tabela 1 - Distribuição da base ASVspoof 2019 PA**

| Partição | Bonafide | Spoof | Total |
|---|---:|---:|---:|
| Train | 5.400 | 48.600 | 54.000 |
| Dev | 5.400 | 24.300 | 29.700 |
| Eval | 18.090 | 116.640 | 134.730 |

Fonte: protocolos em [PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt](PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt), [PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt](PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt), [PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt](PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt).

### 3.2 Extração de características espectrais

A extração foi implementada principalmente em [espectograma/main.py](espectograma/main.py), com geração de cinco representações: STFT, Log-Mel, CQT, LFCC e CQCC. O pipeline inclui:

- leitura robusta via ffmpeg para contornar falhas de decodificação;
- padronização para 16 kHz e canal único;
- geração de imagens PNG em paleta contínua para entrada em CNN.

Também foram desenvolvidos scripts auxiliares para depuração de arquivos faltantes e reprocessamento pontual de áudios com falhas:

- [espectograma/corrigindo-problemas/missing_files.py](espectograma/corrigindo-problemas/missing_files.py)
- [espectograma/corrigindo-problemas/espectogramas-entrada-txt-faltante.py](espectograma/corrigindo-problemas/espectogramas-entrada-txt-faltante.py)

O arquivo [espectograma/corrigindo-problemas/faltantes_espectrogramas.txt](espectograma/corrigindo-problemas/faltantes_espectrogramas.txt) contém 85.728 ocorrências listadas de faltantes na verificação histórica.

### 3.3 Arquiteturas e treinamento

Foram utilizadas duas arquiteturas principais:

- EfficientNet-B0 com adaptação para 1 canal (features em escala cinza);
- ResNet-18v2 com entrada RGB e softmax para score de spoof.

Scripts principais de treinamento completo:

- [training/completo-EfficientNet-B0.py](training/completo-EfficientNet-B0.py)
- [training/completo-ResNet-18.py](training/completo-ResNet-18.py)

Scripts de treinamento parcial por execução (runs com timestamp):

- [training/metade-EfficientNet-B0.py](training/metade-EfficientNet-B0.py)
- [training/metade-ResNet-18.py](training/metade-ResNet-18.py)

Script de execução das etapas faltantes:

- [training/lab.sh](training/lab.sh)

**Tabela 2 - Hiperparâmetros e configurações relevantes**

| Item | EfficientNet-B0 | ResNet-18v2 |
|---|---|---|
| Batch size | 32 | 32 |
| Épocas máximas | 100 | 100 |
| Otimizador | AdamW | AdamW |
| Learning rate inicial | 1e-4 | 1e-4 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Early stopping | Paciência 10 | Paciência 10 |
| Entrada | 1 canal (L) | 3 canais (RGB) |
| Métrica de seleção | menor EER em DEV | menor EER em DEV |
| Dispositivo registrado em logs | CUDA | CUDA |

### 3.4 Métricas e avaliação oficial

A avaliação utilizou:

- EER pelo módulo [tdcf/eval_metrics.py](tdcf/eval_metrics.py);
- t-DCF no formato oficial ASVspoof pelo script [tdcf/evaluate_tDCF_asvspoof19.py](tdcf/evaluate_tDCF_asvspoof19.py), com score CM no formato utt attack label score.

Scores ASV de referência:

- [PA_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt](PA_scores/ASVspoof2019.PA.asv.eval.gi.trl.scores.txt)

### 3.5 Estratégias de ensemble

Foram implementadas duas fases:

1. Fase inicial de análise e busca (script histórico): [ensemble_analysis/ensemble_analysis.py](ensemble_analysis/ensemble_analysis.py).
2. Fase final reprodutível e consolidada (pipeline real): [ensemble_analysis/real_ensemble_pipeline.py](ensemble_analysis/real_ensemble_pipeline.py), executada via [ensemble_analysis/run_ensemble_analysis.py](ensemble_analysis/run_ensemble_analysis.py).

No pipeline final:

- combinações de 4 modelos (70 combinações candidatas);
- amostragem de pesos via Dirichlet + baseline uniforme;
- seleção por menor min-tDCF em DEV;
- validação em EVAL com geração de Top 10;
- geração de score final CM para submissão.

### 3.6 Sobre rastreabilidade dos scores

A origem dos scores usados no ensemble final foi registrada em:

- [final_ensemble_results/metadata/score_sources.json](final_ensemble_results/metadata/score_sources.json)

A seleção foi feita prioritariamente com scores DEV das subpastas de checkpoints (quando disponíveis), com fallback documentado para arquivos na pasta geral de scores.

---

## 4 RESULTADOS

### 4.1 Resultados históricos de etapa preliminar (resultadoextremo)

Os logs em [resultadoextremo](resultadoextremo) representam uma etapa anterior, com desempenho inferior à etapa final.

**Tabela 3 - Desempenho histórico (resultadoextremo)**

| Feature | Melhor EER (histórico no log) | EER EVAL | Accuracy EVAL |
|---|---:|---:|---:|
| CQCC | 5,89% | 8,43% | 0,9157 |
| CQT | 4,00% | 5,61% | 0,9439 |
| LogMel | 3,15% | 5,11% | 0,9489 |
| STFT | 2,50% | 3,49% | 0,9651 |

Fonte: [resultadoextremo/log_cqcc.txt](resultadoextremo/log_cqcc.txt), [resultadoextremo/log_cqt.txt](resultadoextremo/log_cqt.txt), [resultadoextremo/log_logmel.txt](resultadoextremo/log_logmel.txt), [resultadoextremo/log_stft.txt](resultadoextremo/log_stft.txt).

### 4.2 Modelos individuais consolidados (fase principal)

Os resultados abaixo combinam evidências de logs finais e cálculo oficial de min-tDCF a partir dos arquivos de score EVAL.

**Tabela 4 - Resultados individuais finais dos modelos (fase principal)**

| Modelo | Melhor EER DEV (log) | EER EVAL (log/score) | Accuracy EVAL (log) | min-tDCF EVAL |
|---|---:|---:|---:|---:|
| EfficientNet-B0_cqcc | 1,63% | 2,92% | 0,9707 | 0,077788 |
| EfficientNet-B0_cqt | 1,20% | 1,97% | 0,9803 | 0,049519 |
| EfficientNet-B0_lfcc | 4,96% | 5,36% | 0,9464 | 0,144926 |
| EfficientNet-B0_logmel | 1,34% | 1,54% | 0,9846 | 0,041093 |
| EfficientNet-B0_stft | 1,61% | 1,70% | 0,9830 | 0,046347 |
| ResNet-18v2_cqcc | 1,65% | 3,12% | 0,9688 | 0,079356 |
| ResNet-18v2_cqt | 0,85% | 1,71% | 0,9829 | 0,041958 |
| ResNet-18v2_logmel | 1,41% | 1,58% | 0,9842 | 0,040645 |
| ResNet-18v2_stft | 1,34% | 1,45% | 0,9855 | 0,039776 |

Fontes: logs em [log/EfficientNet-B0](log/EfficientNet-B0) e [log/ResNet-18v2](log/ResNet-18v2), além de score files em [scores](scores).

### 4.3 Resultados oficiais em arquivos históricos make-scores

A pasta [make-scores](make-scores) contém arquivos CM prontos para avaliação oficial, associados ao fluxo de correção histórica de score.

**Tabela 5 - Avaliação oficial dos arquivos de score históricos (make-scores)**

| Arquivo CM | CM EER (%) | min-tDCF |
|---|---:|---:|
| cqccscore.txt | 3,122850255 | 0,079355790 |
| cqtsocre.txt | 1,708401153 | 0,041958331 |
| logmelscore.txt | 1,575622402 | 0,040645257 |
| stftscore.txt | 1,453944270 | 0,039776189 |

Fonte: execução de [tdcf/evaluate_tDCF_asvspoof19.py](tdcf/evaluate_tDCF_asvspoof19.py) sobre os arquivos de [make-scores](make-scores).

### 4.4 Resultados de ensemble (Top 10 reais)

A rodada final consolidada gerou 10 ensembles reais em [final_ensemble_results/top10](final_ensemble_results/top10), com seleção por DEV e validação em EVAL.

**Tabela 6 - Top 10 ensembles (EVAL)**

| Rank | Ensemble | EER DEV (%) | min-tDCF DEV | EER EVAL (%) | min-tDCF EVAL |
|---:|---|---:|---:|---:|---:|
| 1 | ensemble_07 | 0,6121 | 0,015775 | 0,9122 | 0,024222 |
| 2 | ensemble_04 | 0,5772 | 0,015536 | 0,9188 | 0,024924 |
| 3 | ensemble_08 | 0,6121 | 0,015807 | 1,0005 | 0,025294 |
| 4 | ensemble_10 | 0,6667 | 0,015988 | 1,2887 | 0,027064 |
| 5 | ensemble_06 | 0,6471 | 0,015741 | 1,4527 | 0,027255 |
| 6 | ensemble_05 | 0,6296 | 0,015667 | 1,2106 | 0,029495 |
| 7 | ensemble_09 | 0,6667 | 0,015875 | 1,2669 | 0,031337 |
| 8 | ensemble_02 | 0,6883 | 0,014960 | 1,4255 | 0,032293 |
| 9 | ensemble_01 | 0,6862 | 0,014747 | 1,2879 | 0,032930 |
| 10 | ensemble_03 | 0,7366 | 0,015043 | 1,5308 | 0,035379 |

Fonte: [final_ensemble_results/top10/summary.csv](final_ensemble_results/top10/summary.csv).

### 4.5 Melhor ensemble final e score de submissão

Melhor ensemble final (ranqueamento EVAL):

- ID: ensemble_07
- Modelos: EfficientNet-B0_cqcc + EfficientNet-B0_logmel + ResNet-18v2_cqt + ResNet-18v2_stft
- Pesos: 0,3686; 0,1041; 0,2358; 0,2914
- EER EVAL oficial: 0,912157320%
- min-tDCF oficial: 0,024221643

Arquivos principais:

- [final_ensemble_results/final/best_ensemble.json](final_ensemble_results/final/best_ensemble.json)
- [final_ensemble_results/final/CM_EVAL_TOP1.txt](final_ensemble_results/final/CM_EVAL_TOP1.txt)
- [final_ensemble_results/top10/ensemble_07/official_tdcf_stdout.txt](final_ensemble_results/top10/ensemble_07/official_tdcf_stdout.txt)

### 4.6 Infraestrutura, artefatos e cobertura de execução

**Tabela 7 - Infraestrutura e artefatos observados no projeto**

| Item | Valor observado |
|---|---|
| Dispositivo de treino/inferência | CUDA (registrado nos logs) |
| Checkpoints *_best.pth encontrados | 8 |
| Arquivos de score DEV | 9 |
| Arquivos de score EVAL | 9 |
| Combinações candidatas no ensemble final | 70 |
| Amostras comuns usadas no ensemble final | DEV: 29.700; EVAL: 134.730 |
| Tempo de execução da rodada final do pipeline | ~2 min 31 s (conforme pipeline.log) |

Fontes: [final_ensemble_results/logs/pipeline.log](final_ensemble_results/logs/pipeline.log), [checkpoints](checkpoints), [scores](scores).

### 4.7 Inserção recomendada de figuras no texto

**Figura 1 - STFT de amostra de áudio de referência (inserir na seção 3.2, após descrição do STFT).**  
![Figura 1](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_01_STFT.png)

Legenda sugerida: Distribuição tempo-frequência da amostra PA_T_0000001 no domínio STFT, destacando energia harmônica e segmentos de silêncio utilizados no pré-processamento para treinamento.

**Figura 2 - Log-Mel da mesma amostra (inserir na seção 3.2, comparação com STFT).**  
![Figura 2](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_03_LogMel.png)

Legenda sugerida: Projeção Log-Mel da amostra PA_T_0000001, utilizada para aproximar a percepção auditiva e aumentar robustez de representação para classificação spoof/bonafide.

**Figura 3 - Cochleagrama/CQT (inserir na seção 3.2, após descrição de CQT).**  
![Figura 3](espectograma/test-espectograma/espectrogramas_saida/PA_T_0000001_04_Cochleagrama.png)

Legenda sugerida: Representação CQT com ênfase em resolução espectral de baixas frequências, relevante para padrões de replay e artefatos de reprodução.

**Figura 4 - Cenário PA do relatório parcial (inserir na seção 3.1, no Word a partir do PDF parcial).**

Legenda sugerida: Cenário de ataque de replay no ASVspoof 2019 PA, com dispositivo de reprodução e aquisição no ambiente físico.

**Figura 5 - Arquitetura EfficientNet-B0 adaptada (inserir na seção 3.3, no Word a partir do PDF parcial).**

Legenda sugerida: Estrutura simplificada da EfficientNet-B0 com adaptação para classificação binária de espectrogramas.

---

## 5 DISCUSSÃO

### 5.1 Interpretação dos resultados de modelos individuais

Os resultados finais dos modelos individuais confirmam o comportamento já indicado no relatório parcial: representações espectrais mais ricas (Log-Mel, STFT e CQT) tendem a superar representações cepstrais mais comprimidas (especialmente LFCC) no cenário PA deste projeto. O pior desempenho de LFCC foi consistente em DEV e EVAL, com aumento de erro e min-tDCF elevado, sugerindo menor capacidade de generalização para as condições de replay avaliadas.

No bloco ResNet-18v2, STFT e LogMel apresentaram os menores min-tDCF entre os modelos individuais avaliados, com vantagem concreta em custo normalizado de operação tandem. Esse resultado reforça a importância de avaliar não apenas EER, mas a métrica oficial orientada a aplicação ASV.

### 5.2 Evolução do projeto e impacto das correções

A comparação entre a etapa histórica em [resultadoextremo](resultadoextremo) e a fase principal indica melhora substancial. Além do ganho de modelagem, houve ganho operacional por correção de fluxo:

- tratamento de falhas de leitura de FLAC;
- padronização de geração de score;
- ajuste de score para formato oficial de t-DCF;
- revisão da origem dos scores usados no ensemble (prioridade para subpastas de checkpoints em DEV).

Esse último ponto foi crítico: a rastreabilidade em [final_ensemble_results/metadata/score_sources.json](final_ensemble_results/metadata/score_sources.json) eliminou ambiguidade entre scores da raiz e scores de runs específicos.

### 5.3 Ensemble e complementaridade

O melhor desempenho global foi obtido com ensemble de quatro modelos heterogêneos. A combinação vencedora não corresponde ao melhor modelo isolado repetido, mas ao equilíbrio de decisões entre arquiteturas e features distintas. Em termos práticos, isso significa que erros residuais de um modelo tendem a ser parcialmente compensados por outro.

A queda de EER de aproximadamente 1,45%-1,54% (melhores modelos individuais) para 0,9122% (melhor ensemble) e a redução de min-tDCF para 0,024222 mostram benefício real da fusão ponderada.

### 5.4 Limitações observadas

1. Nem todos os modelos em score possuem checkpoint correspondente em runs finais (exemplo: ResNet-18v2_cqcc não aparece no conjunto de 8 checkpoints best mapeados no pipeline final).
2. Metadados completos de hardware (modelo de GPU, memória, tempo por época) não estão padronizados nos logs.
3. Parte das figuras conceituais está em PDFs, exigindo reconstrução manual no Word para versão final formatada.
4. Há arquivos históricos de experimentação (make-scores, resultadoextremo) com nomenclatura heterogênea, demandando normalização documental.

### 5.5 Comparação com literatura e aderência metodológica

Os resultados do projeto dialogam com achados da literatura presente no repositório:

- o uso de espectrogramas e CNNs em spoof detection é consistente com estudos de ResNet/LCNN em ASVspoof;
- a avaliação por t-DCF segue o plano oficial ASVspoof 2019;
- a discussão sobre informações discriminativas em silêncio e regiões espectrais é coerente com estudos sobre energia regional e artefatos de síntese/replay.

**Tabela 8 - Trabalhos relacionados e relação com este projeto**

| Referência | Contribuição principal | Relação com este projeto |
|---|---|---|
| ASVspoof 2019 Evaluation Plan | protocolo e métrica t-DCF | base oficial de avaliação adotada |
| Disken (2024) | energia regional complementar | reforça relevância de pistas de energia e regiões específicas |
| Chakravarty & Dua (2024) | análise de datasets e espectrogramas com ResNet | aproximação metodológica por espectrograma + CNN |
| Cuccovillo et al. (2023) | AST para detecção de fala sintética | referência para futuras extensões com Transformer |

---

## 6 CONSIDERAÇÕES FINAIS

O projeto atingiu o objetivo geral ao desenvolver uma cadeia completa de contramedidas para replay no ASVspoof 2019 PA, desde pré-processamento até avaliação oficial e fusão por ensemble. O principal resultado técnico foi a obtenção de desempenho superior por fusão de modelos, com EER de 0,9122% e min-tDCF de 0,024222 no melhor ensemble final, ambos verificados com ferramentas oficiais.

Além do desempenho, o trabalho produziu contribuição metodológica relevante em reprodutibilidade: organização de scores, rastreabilidade de origem de dados, padronização do formato CM e consolidação de resultados em estrutura de artefatos final. A análise mostrou que decisões de engenharia de experimento (qual score usar, em qual etapa, com qual protocolo) influenciam diretamente a validade dos resultados.

Portanto, as evidências apontam que a combinação de representações espectrais complementares, arquiteturas distintas e avaliação orientada por min-tDCF constitui estratégia robusta para mitigação de ataques de replay no cenário estudado.

### 6.1 RECOMENDAÇÕES PARA TRABALHOS FUTUROS

1. Incluir arquiteturas baseadas em Transformer (AST e variantes) no mesmo protocolo de avaliação.
2. Incorporar embeddings pré-treinados (ex.: wav2vec2, ECAPA-TDNN) para comparação direta com o pipeline atual.
3. Implementar calibração de score (ex.: logística) antes da fusão, para avaliar impacto em t-DCF.
4. Padronizar logging de infraestrutura (GPU, RAM, tempo por época) para análise de custo computacional.
5. Reexecutar todos os modelos com versionamento único de dataset e seeds múltiplas, reportando intervalo de confiança.

---

## 7 USO DE INTELIGÊNCIA ARTIFICIAL GENERATIVA

### Perguntas sobre uso de IA generativa

1. Para escrita deste relatório, alguma ferramenta de IA generativa foi utilizada?  
Sim.

2. Qual(is) ferramenta(s) de IA generativa foi(foram) utilizada(s)?  
- GitHub Copilot (assistente de apoio técnico e organização textual).  
- Gemini (conforme declarado no relatório parcial, para apoio textual).

3. Indique os usos aplicados neste relatório final:  
- Correção gramatical e revisão de fluidez textual.  
- Organização de estrutura e consolidação de resultados.  
- Apoio na padronização descritiva de seções técnicas.  
- Geração automática de tabelas a partir de resultados computados no projeto.

4. Declaração:  
Durante a preparação deste Relatório Final, ferramentas de IA generativa foram utilizadas para apoio de revisão linguística, organização textual e estruturação técnica. Todo o conteúdo foi revisado criticamente, validado com evidências do próprio projeto e é de responsabilidade do autor.

---

## 8 OUTRAS ATIVIDADES REALIZADAS

1. Produção de apresentações de acompanhamento técnico (arquivos em [info/slides](info/slides)).
2. Levantamento e estudo de literatura especializada em spoof detection, incluindo plano oficial ASVspoof e artigos recentes sobre espectrogramas, CNNs e Transformers.
3. Consolidação de artefatos finais de ensemble e validação oficial com script t-DCF.
4. Organização de scripts de execução e automação para treinamentos faltantes, correções de caminho e rastreabilidade de resultados.

---

## REFERÊNCIAS

AGGARWAL, R. K.; DAVE, M. Speech emotion recognition using composite MFCC and imbfcc features. In: Signal and Information Processing, 2011.

CHAKRAVARTY, N.; DUA, M. Publicly available datasets analysis and spectrogram-ResNet41 based improved features extraction for audio spoof attack detection. International Journal of System Assurance Engineering and Management, 2024.

CUCCOVILLO, L.; GERHARDT, M.; AICHROTH, P. Audio Spectrogram Transformer for Synthetic Speech Detection via Speech Formant Analysis. IEEE WIFS, 2023.

DELGADO, H. et al. ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan, 2021.

DIŞKEN, G. Complementary regional energy features for spoofed speech detection. Computer Speech & Language, v. 85, 2024.

HE, K. et al. Deep residual learning for image recognition. CVPR, 2016.

KINNUNEN, T. et al. ASVspoof 2019: Selected Topics in Antispoofing for Speaker Recognition. Interspeech/ArXiv, 2019.

NAUTSCH, A. et al. The ASVspoof 2019 Challenge: spoofing countermeasures for the detection of synthesized, converted and replayed speech. IEEE JSTSP, 2021.

TODISCO, M. et al. t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and ASV. Odyssey, 2018.

YAMAGISHI, J. et al. ASVspoof 2019 Evaluation Plan. ASVspoof Consortium, 2019.

Referências adicionais utilizadas no relatório parcial, mantidas por continuidade metodológica:

ALAN V. OPPENHEIM. Spectrogram and the STFT. Aalto University.  
HOCHULI, A. G. Projeto de Pesquisa PIBIC 2025/2026.  
SUNG, B. EfficientNet Implementation.  
ALI, H. et al. Fine tuned EfficientNet-B0 for classification tasks. Scientific Reports, 2025.

---

## ANEXO A - INVENTÁRIO TÉCNICO DOS ARTEFATOS

### A.1 Estruturas-chave do projeto analisadas

- Relatórios e modelo institucional: [info/relaotrios](info/relaotrios)
- Slides de acompanhamento: [info/slides](info/slides)
- Literatura técnica em PDF: [info/estado-arte](info/estado-arte)
- Extração de espectrogramas: [espectograma](espectograma)
- Treinamento de modelos: [training](training)
- Métricas oficiais t-DCF: [tdcf](tdcf)
- Scores e protocolos: [scores](scores), [PA_cm_protocols](PA_cm_protocols), [PA_scores](PA_scores)
- Logs de execução: [log](log), [resultadoextremo](resultadoextremo), [checkpoints/runs-Jun-1/log1-6](checkpoints/runs-Jun-1/log1-6)
- Resultados finais de ensemble: [final_ensemble_results](final_ensemble_results)

### A.2 Artefatos finais para uso imediato

1. Score final para avaliação oficial: [final_ensemble_results/final/CM_EVAL_TOP1.txt](final_ensemble_results/final/CM_EVAL_TOP1.txt)
2. Configuração do melhor ensemble: [final_ensemble_results/final/best_ensemble.json](final_ensemble_results/final/best_ensemble.json)
3. Ranking consolidado dos Top 10: [final_ensemble_results/top10/summary.csv](final_ensemble_results/top10/summary.csv)
4. Logs oficiais t-DCF por ensemble: [final_ensemble_results/top10](final_ensemble_results/top10)

---

**Nota de consistência:** todas as métricas apresentadas foram extraídas de arquivos do próprio projeto (logs, score files, CSV/JSON de resultado e execução oficial do script de t-DCF). Onde não há metadado explícito (por exemplo, modelo exato da GPU), a ausência foi indicada como limitação documental.