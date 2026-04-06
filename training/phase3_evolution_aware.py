"""
phase3_evolution_aware.py
=========================
Phase 3: Evolution-Aware Training.
Add GNN constraints that penalise biologically impossible sequences.
Joint training of transformer + phylogenetic GNN.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import torch
from typing import Dict, Tuple
from config.settings import PHASE3_EPOCHS, GNN_EPOCHS, BATCH_SIZE


def run_phase3(
    sequences_raw: Dict[str, str],
    gnn_epochs:    int = None,
) -> Tuple[object, torch.Tensor]:
    """
    Train phylogenetic GNN with biological constraint loss.
    Penalises:
      - GC content outside species norm
      - Biologically impossible codon patterns
      - Phylogenetically inconsistent embeddings

    Returns (gnn_model, species_embeddings).
    """
    from models.gnn_phylogenetic import train_phylo_gnn

    gnn_epochs = gnn_epochs or GNN_EPOCHS

    print("\n" + "=" * 65)
    print("  PHASE 3 — Evolution-Aware Training")
    print("  Task: GNN with biological constraints")
    print("  Penalties:")
    print("    • GC content deviation from species norm")
    print("    • Codon usage bias violations")
    print("    • Phylogenetic inconsistency")
    print("=" * 65)

    species_names = list(sequences_raw.keys())

    gnn_model, embeddings = train_phylo_gnn(
        species_names=species_names,
        sequences=sequences_raw,
        epochs=gnn_epochs,
    )

    print(f"  Species embeddings shape: {embeddings.shape}")
    print("  ✅ Phase 3 complete — evolution-aware model trained.\n")
    return gnn_model, embeddings
