#!/usr/bin/env python3
"""
Script para executar análise de ensemble
"""

import sys
from pathlib import Path

# Adicionar o diretório ao path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

from ensemble_analysis import EnsembleAnalyzer

if __name__ == '__main__':
    analyzer = EnsembleAnalyzer(str(workspace_root))
    results = analyzer.run_full_analysis()
