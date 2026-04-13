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
  - Transformer+GNN Fusion model (NEW)

Uses learnable attention-based weighting per position and
confidence calibration via temperature scaling.

v3.0: Added multi-species fusion reconstruction using the unified
TransformerGNNFusion model with per-base confidence scoring.
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Ensemble reconstruction (original pipeline path)
# ═══════════════════════════════════════════════════════════════════════════════
def ensemble_reconstruct(
    sequence:       str,
    bert_model,
    ae_model,
    lstm_model,
    vocab:          Dict[str, int],
    fusion_model    = None,
    species_feats   = None,
    adjacency       = None,
    species_idx:    int = 0,
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

    # ── Step 0: Fusion model (if available) ───────────────────────────────────
    fusion_recon, fusion_confs, fusion_reliability = None, None, 0.5
    if fusion_model is not None and species_feats is not None:
        try:
            from models.fusion_model import multi_species_reconstruct
            fusion_results = multi_species_reconstruct(
                model=fusion_model,
                sequences={"target": sequence},
                vocab=vocab,
                species_names=["target"],
                species_feats=species_feats,
                adjacency=adjacency,
                device=device,
            )
            if "target" in fusion_results:
                fr = fusion_results["target"]
                fusion_recon = fr["reconstructed_seq"]
                fusion_confs = fr["confidences"]
                fusion_reliability = fr["reliability_score"]
        except Exception as e:
            print(f"  [ENSEMBLE] Fusion fallback: {e}")

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

    # ── Combine: prefer fusion if available ───────────────────────────────────
    if fusion_recon is not None and fusion_confs is not None:
        # Use fusion as primary, BERT as fallback for remaining gaps
        final_seq = list(fusion_recon)
        for i in range(len(final_seq)):
            if final_seq[i] == "N" and i < len(bert_recon) and bert_recon[i] != "N":
                final_seq[i] = bert_recon[i]
        final_seq = "".join(final_seq)

        # Combine confidences: weighted average favoring fusion
        min_len = min(len(fusion_confs), len(bert_confs), len(ae_confs))
        combined_conf = []
        for i in range(min_len):
            fc = fusion_confs[i] if i < len(fusion_confs) else 0.5
            bc = bert_confs[i] if i < len(bert_confs) else 0.5
            ac = ae_confs[i] if i < len(ae_confs) else 0.5
            # Weighted: fusion 50%, BERT 30%, AE 20%
            combined_conf.append(0.5 * fc + 0.3 * bc + 0.2 * ac)
    else:
        final_seq = bert_recon
        min_len = min(len(ae_confs), len(bert_confs))
        combined_conf = [
            (ae_confs[i] + bert_confs[i]) / 2.0
            for i in range(min_len)
        ]

    # Pad confidence for remaining positions
    while len(combined_conf) < len(final_seq):
        combined_conf.append(0.5)

    # ── Confidence scoring ────────────────────────────────────────────────────
    try:
        from models.confidence_scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        # Create mock logits from confidences for scoring
        conf_tensor = torch.tensor(combined_conf[:len(final_seq)])
        # Build a simple 5-class logit tensor from confidences
        logits = torch.zeros(len(conf_tensor), 5)
        for i, c in enumerate(conf_tensor):
            base = final_seq[i] if i < len(final_seq) else "N"
            base_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}.get(base, 4)
            logits[i, base_idx] = c * 3  # scale confidence to logit
        conf_result = scorer.score_sequence(logits)
        combined_conf = conf_result["per_base_confidence"]
        reliability = conf_result["reliability_score"]
    except Exception:
        reliability = float(np.mean(combined_conf)) if combined_conf else 0.5

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_gaps_before = sequence.count("N")
    n_gaps_after  = final_seq.count("N")
    coverage      = 1.0 - (n_gaps_after / max(1, len(final_seq)))

    # Confidence classification
    conf_arr = np.array(combined_conf[:len(final_seq)])
    high_conf_pct = float((conf_arr >= 0.85).mean()) if len(conf_arr) > 0 else 0
    low_conf_count = int((conf_arr < 0.5).sum()) if len(conf_arr) > 0 else 0

    details = {
        "gaps_before":         n_gaps_before,
        "gaps_after":          n_gaps_after,
        "coverage":            round(float(coverage), 4),
        "reliability_score":   round(float(reliability), 4),
        "mean_confidence":     round(float(np.mean(combined_conf))
                                     if combined_conf else 0.5, 4),
        "high_confidence_pct": round(high_conf_pct, 4),
        "low_confidence_count": low_conf_count,
        "fusion_used":         fusion_recon is not None,
        "fusion_reliability":  round(float(fusion_reliability), 4),
        "ae_repairs":          ae_repairs[:20],
        "lstm_extension_len":  len(lstm_recon) - len(bert_recon[:1000]),
    }

    return final_seq, combined_conf, details


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-Species Ensemble Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════
def multi_species_ensemble_reconstruct(
    simulated:      Dict[str, Dict],
    sequences_raw:  Dict[str, str],
    bert_model,
    ae_model,
    lstm_model,
    vocab:          Dict[str, int],
    fusion_model    = None,
    species_names:  List[str] = None,
    device:         torch.device = None,
) -> Dict[str, Dict]:
    """
    Reconstruct all species simultaneously, using multi-species context
    from the fusion model when available.

    Returns: {species_name: {reconstructed_seq, confidences, details}}
    """
    if device is None:
        device = DEVICE

    # Build phylo graph for fusion model
    species_feats, adjacency = None, None
    if fusion_model is not None and species_names:
        try:
            from models.fusion_model import build_fusion_phylo_graph
            species_feats, adjacency = build_fusion_phylo_graph(
                species_names, sequences_raw,
            )
            species_feats = species_feats.to(device)
            adjacency = adjacency.to(device)
        except Exception as e:
            print(f"  [MULTI] Phylo graph error: {e}")

    results = {}
    for sp_name, sim_result in simulated.items():
        print(f"\n  Reconstructing: {sp_name}")
        damaged = sim_result["damaged_sequence"]

        sp_idx = 0
        if species_names and sp_name in species_names:
            sp_idx = species_names.index(sp_name)

        recon_seq, confidences, details = ensemble_reconstruct(
            sequence=damaged,
            bert_model=bert_model,
            ae_model=ae_model,
            lstm_model=lstm_model,
            vocab=vocab,
            fusion_model=fusion_model,
            species_feats=species_feats,
            adjacency=adjacency,
            species_idx=sp_idx,
            device=device,
        )

        results[sp_name] = {
            "original_length":    len(sequences_raw.get(sp_name, damaged)),
            "damaged_length":     len(damaged),
            "reconstructed_seq":  recon_seq[:500] + "…",
            "full_length":        len(recon_seq),
            "confidences":        confidences[:500],
            **details,
            "mutation_log":       sim_result.get("mutation_log", [])[:20],
            "mutation_summary":   sim_result.get("mutation_summary", {}),
        }

        print(f"    Coverage: {details['coverage']:.2%}  "
              f"Reliability: {details['reliability_score']:.4f}  "
              f"Confidence: {details['mean_confidence']:.4f}  "
              f"Gaps: {details['gaps_before']} → {details['gaps_after']}"
              f"  Fusion: {'✓' if details.get('fusion_used') else '✗'}")

    return results
