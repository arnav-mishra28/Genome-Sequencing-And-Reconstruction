"""
ensemble_reconstructor.py
=========================
Multi-model fusion for genome reconstruction.
Combines outputs from:
  - DNABERT-2 transformer
  - ESM structure encoder
  - Denoising autoencoder
  - GNN phylogenetic refinement
  - BiLSTM predictor

Uses learnable attention-based weighting per position and
confidence calibration via temperature scaling.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import EMBED_DIM, DEVICE


class PositionWiseGating(nn.Module):
    """Learnable per-position gating for model fusion."""
    def __init__(self, n_models: int, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_models * dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_models),
            nn.Softmax(dim=-1),
        )

    def forward(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """
        model_outputs: (B, L, n_models, D)
        Returns: (B, L, D) fused output
        """
        B, L, M, D = model_outputs.shape
        flat = model_outputs.reshape(B, L, M * D)
        weights = self.gate(flat)  # (B, L, M)
        fused = (model_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return fused


class TemperatureScaling(nn.Module):
    """Post-hoc confidence calibration via temperature scaling."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)


class EnsembleReconstructor(nn.Module):
    """
    Multi-model ensemble for genome reconstruction.

    Combines logits/embeddings from multiple models using
    learnable position-wise attention gating and outputs
    calibrated confidence scores.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = EMBED_DIM,
        n_models:   int = 3,  # DNABERT-2 + AE + LSTM (active models)
    ):
        super().__init__()
        self.n_models  = n_models
        self.embed_dim = embed_dim

        # Per-model projection to common space
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vocab_size, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(n_models)
        ])

        # Position-wise gating
        self.gate = PositionWiseGating(n_models, embed_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        # Confidence calibration
        self.temp_scaling = TemperatureScaling()

        # Per-position confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        model_logits: List[torch.Tensor],  # list of (B, L, V) from each model
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multiple model outputs.

        Args:
            model_logits: List of logit tensors from each sub-model

        Returns:
            dict with 'logits', 'confidence', 'calibrated_logits'
        """
        # Project each model's output to common embedding space
        projected = []
        for i, (logits, proj) in enumerate(zip(model_logits, self.projectors)):
            projected.append(proj(logits))  # (B, L, D)

        # Stack: (B, L, n_models, D)
        stacked = torch.stack(projected, dim=2)

        # Fuse with gating
        fused = self.gate(stacked)  # (B, L, D)

        # Output
        logits     = self.output_head(fused)
        cal_logits = self.temp_scaling(logits)
        confidence = self.confidence_head(fused).squeeze(-1)

        return {
            "logits":            logits,
            "calibrated_logits": cal_logits,
            "confidence":        confidence,
            "fused_embeddings":  fused,
        }


def ensemble_reconstruct(
    sequence:       str,
    bert_model,
    ae_model,
    lstm_model,
    vocab:          Dict[str, int],
    device:         torch.device = None,
) -> Tuple[str, List[float], Dict]:
    """
    Run ensemble reconstruction on a damaged sequence.

    Returns:
        reconstructed: str
        confidences:   List[float]
        details:       Dict with per-model outputs
    """
    from models.dnabert2_transformer import fill_masked_sequence
    from models.denoising_autoencoder import denoise_sequence
    from models.lstm_predictor import predict_sequence

    if device is None:
        device = DEVICE

    # ── Step 1: AE denoising ──────────────────────────────────────────────────
    ae_recon, ae_confs, ae_repairs = denoise_sequence(ae_model, sequence, device)

    # ── Step 2: BERT gap filling ──────────────────────────────────────────────
    bert_recon, bert_confs = fill_masked_sequence(
        bert_model, ae_recon, vocab, device=device,
    )

    # ── Step 3: LSTM extension ────────────────────────────────────────────────
    lstm_recon = predict_sequence(
        lstm_model, bert_recon[:1000], steps=3, device=device,
    )

    # ── Combine confidences ───────────────────────────────────────────────────
    final_seq = bert_recon
    min_len = min(len(ae_confs), len(bert_confs))
    combined_conf = [
        (ae_confs[i] + bert_confs[i]) / 2.0
        for i in range(min_len)
    ]

    # Pad confidence for remaining positions
    while len(combined_conf) < len(final_seq):
        combined_conf.append(0.5)

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_gaps_before = sequence.count("N")
    n_gaps_after  = final_seq.count("N")
    coverage      = 1.0 - (n_gaps_after / max(1, len(final_seq)))
    reliability   = np.mean(combined_conf) if combined_conf else 0.5

    details = {
        "gaps_before":       n_gaps_before,
        "gaps_after":        n_gaps_after,
        "coverage":          round(float(coverage), 4),
        "reliability_score": round(float(reliability), 4),
        "mean_confidence":   round(float(np.mean(combined_conf))
                                   if combined_conf else 0.5, 4),
        "ae_repairs":        ae_repairs[:20],
        "lstm_extension_len": len(lstm_recon) - len(bert_recon[:1000]),
    }

    return final_seq, combined_conf, details
