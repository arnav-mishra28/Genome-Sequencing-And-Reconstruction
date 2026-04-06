"""
corruption.py
=============
Learnable corruption engine for Phase 2 training.
Applies configurable damage profiles to DNA sequences for
training the denoising / reconstruction models.

Damage types mirror real ancient DNA degradation patterns:
  - Deamination (C→T, G→A) with position-dependent rates
  - Oxidative damage (G→T)
  - Random substitutions
  - Gap insertion (N regions)
  - Small indels
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional


class CorruptionEngine:
    """
    Configurable DNA corruption engine.
    Each call to .corrupt() applies layered damage.
    """
    def __init__(
        self,
        deamination_rate:  float = 0.06,
        oxidation_rate:    float = 0.01,
        substitution_rate: float = 0.02,
        gap_rate:          float = 0.005,
        indel_rate:        float = 0.003,
        gap_size_range:    Tuple[int, int] = (10, 100),
        seed:              Optional[int] = None,
    ):
        self.deamination_rate  = deamination_rate
        self.oxidation_rate    = oxidation_rate
        self.substitution_rate = substitution_rate
        self.gap_rate          = gap_rate
        self.indel_rate        = indel_rate
        self.gap_size_range    = gap_size_range
        self.rng               = random.Random(seed)

    def corrupt(self, sequence: str,
                return_mask: bool = False) -> Tuple[str, ...]:
        """
        Apply layered corruption to a clean DNA sequence.

        Returns:
            corrupted_sequence: str
            corruption_mask:    np.ndarray  (1=corrupted, 0=original)
                                Only if return_mask=True.
        """
        seq  = list(sequence.upper())
        mask = np.zeros(len(seq), dtype=np.float32)
        L    = len(seq)

        # ── Deamination ───────────────────────────────────────────────────────
        for i in range(L):
            dist_from_end = min(i, L - 1 - i)
            rate_mult = 3.0 if dist_from_end < 10 else 1.0
            effective_rate = self.deamination_rate * rate_mult

            if seq[i] == "C" and self.rng.random() < effective_rate:
                seq[i] = "T"
                mask[i] = 1.0
            elif seq[i] == "G" and self.rng.random() < effective_rate * 0.8:
                seq[i] = "A"
                mask[i] = 1.0

        # ── Oxidation ─────────────────────────────────────────────────────────
        for i in range(L):
            if seq[i] == "G" and self.rng.random() < self.oxidation_rate:
                seq[i] = "T"
                mask[i] = 1.0

        # ── Random substitutions ──────────────────────────────────────────────
        for i in range(L):
            if self.rng.random() < self.substitution_rate:
                orig = seq[i]
                choices = [b for b in "ACGT" if b != orig]
                seq[i] = self.rng.choice(choices)
                mask[i] = 1.0

        # ── Gap insertion ─────────────────────────────────────────────────────
        n_gaps = max(1, int(L * self.gap_rate))
        for _ in range(n_gaps):
            start = self.rng.randint(0, L - 1)
            size  = self.rng.randint(*self.gap_size_range)
            end   = min(start + size, L)
            for j in range(start, end):
                seq[j] = "N"
                mask[j] = 1.0

        result = "".join(seq)
        if return_mask:
            return result, mask
        return (result,)

    def batch_corrupt(self, sequences: List[str],
                      return_masks: bool = False) -> List:
        """Corrupt a batch of sequences."""
        results = []
        for seq in sequences:
            results.append(self.corrupt(seq, return_mask=return_masks))
        return results


def create_corruption_pairs(
    clean_sequences: List[str],
    n_variants: int = 3,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Create multiple corruption variants for each clean sequence.
    Returns list of (clean, corrupted) pairs.
    """
    pairs = []
    for v in range(n_variants):
        engine = CorruptionEngine(
            deamination_rate=0.04 + 0.03 * v,
            oxidation_rate=0.005 + 0.005 * v,
            substitution_rate=0.01 + 0.01 * v,
            seed=seed + v,
        )
        for seq in clean_sequences:
            corrupted = engine.corrupt(seq)[0]
            pairs.append((seq, corrupted))
    return pairs
