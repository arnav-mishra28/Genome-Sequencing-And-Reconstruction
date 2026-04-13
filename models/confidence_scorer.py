"""
confidence_scorer.py
====================
Advanced Confidence Scoring Module for DNA Reconstruction.

This is the startup differentiator — every reconstructed base gets
a rigorous, calibrated confidence score.

Three uncertainty signals are combined:
  1. Softmax confidence — max probability from the prediction distribution
  2. Monte Carlo dropout — epistemic uncertainty via stochastic forward passes
  3. Prediction entropy — information-theoretic uncertainty

The final confidence is:
  conf = w1 * softmax_conf + w2 * (1 - mc_variance) + w3 * (1 - norm_entropy)

Outputs:
  - Per-base confidence:  float in [0, 1] for every position
  - Sequence reliability:  aggregated score for the entire sequence
  - Confidence classification:  HIGH / MEDIUM / LOW per region
  - Calibration metrics:  ECE and reliability diagram data
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Dropout Estimator
# ═══════════════════════════════════════════════════════════════════════════════
def mc_dropout_estimate(
    model:      nn.Module,
    input_fn,
    n_samples:  int = 10,
    device:     torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run multiple stochastic forward passes with dropout enabled
    to estimate epistemic uncertainty.

    Args:
        model:    any model with dropout layers
        input_fn: callable that returns model inputs (called each pass)
        n_samples: number of MC samples
        device:   torch device

    Returns:
        mean_probs:  (L, V) — mean prediction probabilities
        variance:    (L,)   — per-position prediction variance
        entropy:     (L,)   — per-position prediction entropy
    """
    if device is None:
        device = DEVICE

    # Enable dropout during inference for MC estimation
    model.train()  # enables dropout

    all_probs = []
    for _ in range(n_samples):
        with torch.no_grad():
            inputs = input_fn()
            if isinstance(inputs, dict):
                outputs = model(**{k: v.to(device) if torch.is_tensor(v) else v
                                   for k, v in inputs.items()})
            else:
                outputs = model(inputs.to(device))

            # Extract logits
            if isinstance(outputs, dict):
                logits = outputs.get("mlm_logits",
                         outputs.get("recon_logits",
                         outputs.get("logits")))
            else:
                logits = outputs

            probs = torch.softmax(logits[0], dim=-1)  # (L, V)
            all_probs.append(probs.cpu())

    model.eval()  # restore eval mode

    stacked = torch.stack(all_probs)        # (N, L, V)
    mean_probs = stacked.mean(dim=0)        # (L, V)

    # Variance: mean of per-class variances across positions
    variance = stacked.var(dim=0).mean(dim=-1)  # (L,)

    # Entropy of mean prediction
    entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)  # (L,)

    return mean_probs, variance, entropy


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence Scorer
# ═══════════════════════════════════════════════════════════════════════════════
class ConfidenceScorer:
    """
    Combines multiple uncertainty signals into calibrated confidence scores.

    Usage:
        scorer = ConfidenceScorer()
        result = scorer.score_sequence(
            logits=model_logits,       # (L, V) or (B, L, V)
            mc_variance=mc_var,        # (L,) from MC dropout
            mc_entropy=mc_ent,         # (L,) from MC dropout
            model_confidence=conf,     # (L,) from confidence head
        )
    """

    def __init__(
        self,
        w_softmax:  float = 0.4,
        w_mc:       float = 0.3,
        w_entropy:  float = 0.2,
        w_model:    float = 0.1,
        temperature: float = 1.5,
        high_thresh: float = 0.85,
        low_thresh:  float = 0.5,
    ):
        self.w_softmax  = w_softmax
        self.w_mc       = w_mc
        self.w_entropy  = w_entropy
        self.w_model    = w_model
        self.temperature = temperature
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh

    def score_sequence(
        self,
        logits:           torch.Tensor,           # (L, V)
        mc_variance:      Optional[torch.Tensor] = None,  # (L,)
        mc_entropy:       Optional[torch.Tensor] = None,  # (L,)
        model_confidence: Optional[torch.Tensor] = None,  # (L,)
    ) -> Dict:
        """
        Compute calibrated confidence scores for every position.

        Returns dict with:
          - per_base_confidence:  List[float]
          - reliability_score:    float
          - confidence_classes:   List[str]  (HIGH/MEDIUM/LOW)
          - mean_confidence:      float
          - min_confidence:       float
          - low_confidence_count: int
          - high_confidence_pct:  float
        """
        # Handle batched input
        if logits.dim() == 3:
            logits = logits[0]  # (L, V)

        L = logits.shape[0]

        # 1. Softmax confidence
        probs = torch.softmax(logits.detach() / self.temperature, dim=-1)
        softmax_conf = probs.max(dim=-1).values  # (L,)

        # 2. MC dropout variance → confidence (invert: low var = high conf)
        if mc_variance is not None:
            mc_conf = 1.0 - torch.clamp(mc_variance * 10, 0, 1)
        else:
            mc_conf = torch.ones(L)

        # 3. Entropy → confidence (invert: low entropy = high conf)
        if mc_entropy is not None:
            max_entropy = math.log(logits.shape[-1])
            norm_entropy = mc_entropy / max_entropy
            entropy_conf = 1.0 - norm_entropy
        else:
            entropy_conf = torch.ones(L)

        # 4. Model's own confidence prediction
        if model_confidence is not None:
            model_conf = model_confidence
        else:
            model_conf = torch.ones(L)

        # Ensure all on same device, detached
        softmax_conf = softmax_conf.detach().float().cpu()
        mc_conf = mc_conf.detach().float().cpu() if torch.is_tensor(mc_conf) else mc_conf.float().cpu()
        entropy_conf = entropy_conf.detach().float().cpu() if torch.is_tensor(entropy_conf) else entropy_conf.float().cpu()
        model_conf = model_conf.detach().float().cpu() if torch.is_tensor(model_conf) else model_conf.float().cpu()

        # 5. Weighted combination
        combined = (
            self.w_softmax  * softmax_conf +
            self.w_mc       * mc_conf +
            self.w_entropy  * entropy_conf +
            self.w_model    * model_conf
        )

        # Clamp to [0, 1]
        combined = torch.clamp(combined, 0.0, 1.0)

        per_base = combined.detach().numpy().tolist()

        # Classify each position
        classes = []
        for c in per_base:
            if c >= self.high_thresh:
                classes.append("HIGH")
            elif c >= self.low_thresh:
                classes.append("MEDIUM")
            else:
                classes.append("LOW")

        # Aggregate metrics
        conf_array = np.array(per_base)
        high_pct = float((conf_array >= self.high_thresh).mean())
        low_count = int((conf_array < self.low_thresh).sum())

        # Reliability score: geometric mean of positional confidences
        # (penalizes low-confidence positions more than arithmetic mean)
        log_conf = np.log(conf_array.clip(1e-6))
        reliability = float(np.exp(log_conf.mean()))

        return {
            "per_base_confidence":  [round(c, 4) for c in per_base],
            "reliability_score":    round(reliability, 4),
            "confidence_classes":   classes,
            "mean_confidence":      round(float(conf_array.mean()), 4),
            "min_confidence":       round(float(conf_array.min()), 4),
            "max_confidence":       round(float(conf_array.max()), 4),
            "low_confidence_count": low_count,
            "high_confidence_pct":  round(high_pct, 4),
            "n_positions":          L,
        }

    def score_region(
        self,
        per_base_conf: List[float],
        window_size:   int = 50,
    ) -> List[Dict]:
        """
        Score confidence for sliding windows across the sequence.
        Useful for identifying low-confidence regions.
        """
        confs = np.array(per_base_conf)
        regions = []

        for start in range(0, len(confs) - window_size + 1, window_size // 2):
            end = min(start + window_size, len(confs))
            window = confs[start:end]

            region = {
                "start":           start,
                "end":             end,
                "mean_confidence": round(float(window.mean()), 4),
                "min_confidence":  round(float(window.min()), 4),
                "low_count":       int((window < self.low_thresh).sum()),
                "classification":  (
                    "HIGH" if window.mean() >= self.high_thresh else
                    "MEDIUM" if window.mean() >= self.low_thresh else
                    "LOW"
                ),
            }
            regions.append(region)

        return regions


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: score a full reconstruction
# ═══════════════════════════════════════════════════════════════════════════════
def score_reconstruction(
    reconstructed:  str,
    logits:         torch.Tensor,   # (L, V) — prediction logits
    mc_variance:    torch.Tensor = None,
    mc_entropy:     torch.Tensor = None,
    model_conf:     torch.Tensor = None,
    reference:      str = None,     # if available, compute calibration
) -> Dict:
    """
    Full confidence scoring for a reconstructed sequence.
    Returns confidence scores + optional calibration metrics.
    """
    scorer = ConfidenceScorer()

    result = scorer.score_sequence(
        logits=logits,
        mc_variance=mc_variance,
        mc_entropy=mc_entropy,
        model_confidence=model_conf,
    )

    # Regional scores
    result["regions"] = scorer.score_region(
        result["per_base_confidence"], window_size=50,
    )

    # Calibration vs reference (if available)
    if reference is not None:
        from evaluation.metrics import confidence_calibration
        min_len = min(len(reconstructed), len(reference))
        actual_correct = [
            reconstructed[i].upper() == reference[i].upper()
            for i in range(min_len)
        ]
        confs = result["per_base_confidence"][:min_len]
        cal = confidence_calibration(confs, actual_correct)
        result["calibration"] = cal

    return result
