"""
phase2_corruption.py
====================
Phase 2: Corruption Training.
Feed corrupted DNA → train model to reconstruct original.
Multi-task: deamination repair + gap filling + base correction.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from typing import List, Dict
from config.settings import PHASE2_EPOCHS, BATCH_SIZE, AE_EPOCHS


def run_phase2(
    sequences_raw: Dict[str, str],
    simulated:     Dict[str, Dict],
    epochs:        int = None,
    ae_epochs:     int = None,
    batch_size:    int = None,
) -> object:
    """
    Train denoising autoencoder on corrupted → original pairs.
    Returns the trained AE model.
    """
    from models.denoising_autoencoder import train_autoencoder
    from preprocessing.corruption import create_corruption_pairs

    epochs     = epochs or PHASE2_EPOCHS
    ae_epochs  = ae_epochs or AE_EPOCHS
    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 65)
    print("  PHASE 2 — Corruption Training")
    print("  Task: Reconstruct original from corrupted DNA")
    print("=" * 65)

    # Build clean/noisy pairs from simulated data + corruption engine
    clean_seqs = list(sequences_raw.values())
    noisy_seqs = []

    for k, v in sequences_raw.items():
        if k in simulated:
            noisy_seqs.append(simulated[k]["damaged_sequence"])
        else:
            noisy_seqs.append(v)

    # Add extra corruption variants from the engine
    print("  Generating additional corruption variants …")
    extra_pairs = create_corruption_pairs(clean_seqs[:4], n_variants=2, seed=42)
    for clean, corrupted in extra_pairs:
        clean_seqs.append(clean)
        noisy_seqs.append(corrupted)

    # Ensure equal length
    min_len    = min(len(clean_seqs), len(noisy_seqs))
    clean_seqs = clean_seqs[:min_len]
    noisy_seqs = noisy_seqs[:min_len]

    print(f"  Training on {len(clean_seqs)} clean/noisy pairs")

    model = train_autoencoder(
        clean_seqs=clean_seqs,
        noisy_seqs=noisy_seqs,
        epochs=ae_epochs,
        batch_size=batch_size,
    )

    print("  ✅ Phase 2 complete — corruption model trained.\n")
    return model
