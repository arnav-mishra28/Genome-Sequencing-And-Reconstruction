"""
phase1_pretrain.py
==================
Phase 1: Pre-train transformer on modern genomes.
Task: Masked language model prediction (15% mask rate).
Data: Modern human + elephant + wolf + pigeon genomes from NCBI.
Output: Pre-trained DNABERT-2 weights.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from typing import List, Dict
from config.settings import PHASE1_EPOCHS, BATCH_SIZE, MODERN_SPECIES


def run_phase1(
    sequences_raw: Dict[str, str],
    vocab:         Dict[str, int],
    epochs:        int = None,
    batch_size:    int = None,
) -> object:
    """
    Pre-train DNABERT-2 on modern genome sequences using MLM.
    Returns the trained model.
    """
    from models.dnabert2_transformer import train_dnabert2

    epochs     = epochs or PHASE1_EPOCHS
    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 65)
    print("  PHASE 1 — Pre-training Transformer on Modern Genomes")
    print("  Task: Masked Language Model prediction")
    print("=" * 65)

    # Use modern species for pre-training
    modern_seqs = [
        seq for name, seq in sequences_raw.items()
        if name in MODERN_SPECIES
    ]

    if not modern_seqs:
        print("  [WARN] No modern sequences found — using all sequences.")
        modern_seqs = list(sequences_raw.values())

    print(f"  Pre-training on {len(modern_seqs)} modern sequences")

    model = train_dnabert2(
        sequences=modern_seqs,
        vocab=vocab,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("  ✅ Phase 1 complete — transformer pre-trained.\n")
    return model
