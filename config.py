"""
Genome Sequencing and Reconstruction Project
===========================================

A comprehensive Deep Learning framework for reconstructing incomplete DNA sequences
of extinct species using multi-modal neural networks and evolutionary algorithms.

Features:
- LSTM-based sequence prediction
- Transformer models (DNA-BERT style)
- Graph Neural Networks for phylogenetic analysis
- Autoencoder-based denoising
- Real scientific dataset integration
- Mutation analysis and health optimization

Author: AI Research Lab
Version: 1.0.0
License: MIT
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)
    
# Model configurations
MODEL_CONFIGS = {
    'lstm': {
        'hidden_size': 512,
        'num_layers': 3,
        'dropout': 0.2,
        'bidirectional': True
    },
    'transformer': {
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 6,
        'ff_dim': 3072,
        'max_seq_len': 2048
    },
    'autoencoder': {
        'encoding_dim': 256,
        'hidden_dims': [1024, 512, 256],
        'dropout': 0.1
    },
    'gnn': {
        'node_features': 128,
        'edge_features': 64,
        'hidden_dim': 256,
        'num_layers': 4
    }
}

# DNA encoding
DNA_ALPHABET = ['A', 'T', 'G', 'C', 'N']  # N for unknown/missing
DNA_TO_INT = {base: i for i, base in enumerate(DNA_ALPHABET)}
INT_TO_DNA = {i: base for base, i in DNA_TO_INT.items()}

# Codon table for protein synthesis
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Disease-associated mutations (examples)
DISEASE_MUTATIONS = {
    'BRCA1': {
        'position': 185,
        'healthy': 'G',
        'pathogenic': 'A',
        'disease': 'Breast Cancer',
        'impact': 'High'
    },
    'CFTR': {
        'position': 508,
        'healthy': 'CTT',
        'pathogenic': 'deletion',
        'disease': 'Cystic Fibrosis',
        'impact': 'High'
    },
    'APOE4': {
        'position': 112,
        'healthy': 'T',
        'pathogenic': 'C',
        'disease': 'Alzheimer\'s Disease',
        'impact': 'Medium'
    }
}

print(f"Genome Sequencing and Reconstruction Framework Initialized")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Available Models: {list(MODEL_CONFIGS.keys())}")
