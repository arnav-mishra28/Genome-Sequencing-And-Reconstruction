"""
metrics.py
==========
All 5 required evaluation metrics for genome reconstruction:

  1. Sequence Accuracy     — per-base correctness vs reference
  2. Edit Distance         — Levenshtein distance
  3. Reconstruction Similarity — alignment identity percentage
  4. Phylogenetic Consistency  — evolutionary distance preservation
  5. Confidence Calibration    — Expected Calibration Error (ECE)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Sequence Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
def sequence_accuracy(reconstructed: str, reference: str) -> Dict:
    """
    Per-base accuracy: fraction of positions matching the reference.
    """
    r = reconstructed.upper()
    ref = reference.upper()
    min_len = min(len(r), len(ref))

    if min_len == 0:
        return {"accuracy": 0.0, "matches": 0, "total": 0}

    matches = sum(1 for i in range(min_len) if r[i] == ref[i])

    return {
        "accuracy":   round(matches / min_len, 6),
        "matches":    matches,
        "total":      min_len,
        "mismatches": min_len - matches,
        "length_diff": abs(len(r) - len(ref)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Edit Distance (Levenshtein)
# ═══════════════════════════════════════════════════════════════════════════════
def edit_distance(seq1: str, seq2: str, max_len: int = 5000) -> Dict:
    """
    Levenshtein edit distance between two sequences.
    Truncated to max_len for performance.
    """
    s1 = seq1.upper()[:max_len]
    s2 = seq2.upper()[:max_len]
    m, n = len(s1), len(s2)

    # Optimised: two-row DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[j] = min(prev[j] + 1,       # deletion
                          curr[j-1] + 1,       # insertion
                          prev[j-1] + cost)    # substitution
        prev, curr = curr, [0] * (n + 1)

    dist = prev[n]
    normalised = dist / max(m, n) if max(m, n) > 0 else 0.0

    return {
        "edit_distance":    dist,
        "normalised":       round(normalised, 6),
        "similarity":       round(1.0 - normalised, 6),
        "seq1_len":         len(seq1),
        "seq2_len":         len(seq2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Reconstruction Similarity  (BLAST-like alignment identity)
# ═══════════════════════════════════════════════════════════════════════════════
def reconstruction_similarity(reconstructed: str, reference: str,
                               window: int = 100) -> Dict:
    """
    Sliding-window identity percentage (BLAST-like).
    """
    r = reconstructed.upper()
    ref = reference.upper()
    min_len = min(len(r), len(ref))

    if min_len == 0:
        return {"overall_identity": 0.0, "window_identities": []}

    # Overall identity
    overall_matches = sum(1 for i in range(min_len) if r[i] == ref[i])
    overall_id = overall_matches / min_len

    # Window identities
    window_ids = []
    for start in range(0, min_len - window + 1, window // 2):
        end = min(start + window, min_len)
        w_matches = sum(1 for i in range(start, end)
                        if r[i] == ref[i])
        w_id = w_matches / (end - start)
        window_ids.append({
            "start":    start,
            "end":      end,
            "identity": round(w_id, 4),
        })

    # Per-base type accuracy
    base_acc = {}
    for base in "ACGT":
        positions = [i for i in range(min_len) if ref[i] == base]
        if positions:
            correct = sum(1 for i in positions if r[i] == base)
            base_acc[base] = round(correct / len(positions), 4)
        else:
            base_acc[base] = None

    return {
        "overall_identity":  round(overall_id, 6),
        "window_identities": window_ids[:50],  # cap for JSON size
        "per_base_accuracy": base_acc,
        "n_gaps_remaining":  r.count("N"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Phylogenetic Consistency
# ═══════════════════════════════════════════════════════════════════════════════
def _kmer_freq(seq: str, k: int = 4) -> np.ndarray:
    """Compute normalised k-mer frequency vector."""
    from itertools import product
    bases = "ACGT"
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    idx = {km: i for i, km in enumerate(all_kmers)}
    vec = np.zeros(len(all_kmers), dtype=np.float64)
    s = seq.upper().replace("N", "")
    for i in range(len(s) - k + 1):
        km = s[i:i+k]
        if km in idx:
            vec[idx[km]] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def phylogenetic_consistency(
    reconstructed:    str,
    relative_seq:     str,
    expected_distance: float,
    species_name:     str = "",
) -> Dict:
    """
    Measures whether the reconstructed sequence maintains the correct
    evolutionary distance to its modern relative.

    Uses k-mer frequency cosine similarity as a proxy for evolutionary
    distance, then compares to expected phylogenetic distance.
    """
    rec_vec = _kmer_freq(reconstructed)
    rel_vec = _kmer_freq(relative_seq)

    # Cosine similarity
    dot  = np.dot(rec_vec, rel_vec)
    norm = (np.linalg.norm(rec_vec) * np.linalg.norm(rel_vec)) + 1e-9
    cosine_sim = float(dot / norm)

    # Convert expected distance to expected similarity
    # Empirical: similarity ≈ exp(-distance / scale)
    expected_sim = float(np.exp(-expected_distance / 50.0))

    # Consistency score: how close is measured vs expected
    deviation = abs(cosine_sim - expected_sim)
    consistency = max(0.0, 1.0 - deviation)

    return {
        "species":            species_name,
        "cosine_similarity":  round(cosine_sim, 6),
        "expected_similarity": round(expected_sim, 6),
        "expected_distance":  expected_distance,
        "deviation":          round(deviation, 6),
        "consistency_score":  round(consistency, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Confidence Calibration  (Expected Calibration Error — ECE)
# ═══════════════════════════════════════════════════════════════════════════════
def confidence_calibration(
    confidences:  List[float],
    actual_correct: List[bool],
    n_bins: int   = 10,
) -> Dict:
    """
    Expected Calibration Error (ECE).
    Measures how well model confidence matches actual accuracy.

    Args:
        confidences:    per-position confidence scores [0, 1]
        actual_correct: whether each position was correctly reconstructed
        n_bins:         number of calibration bins
    """
    confs  = np.array(confidences, dtype=np.float64)
    accs   = np.array(actual_correct, dtype=np.float64)

    # Defensive: ensure equal length
    min_n  = min(len(confs), len(accs))
    confs  = confs[:min_n]
    accs   = accs[:min_n]
    n      = len(confs)

    if n == 0:
        return {"ece": 0.0, "bins": [], "mean_confidence": 0.0,
                "mean_accuracy": 0.0}

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins_info = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)

        count = mask.sum()
        if count == 0:
            bins_info.append({
                "bin": f"[{lo:.1f}, {hi:.1f})",
                "count": 0, "avg_conf": 0.0, "avg_acc": 0.0,
            })
            continue

        avg_conf = float(confs[mask].mean())
        avg_acc  = float(accs[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)

        bins_info.append({
            "bin":      f"[{lo:.1f}, {hi:.1f})",
            "count":    int(count),
            "avg_conf": round(avg_conf, 4),
            "avg_acc":  round(avg_acc, 4),
            "gap":      round(abs(avg_acc - avg_conf), 4),
        })

    return {
        "ece":             round(ece, 6),
        "bins":            bins_info,
        "mean_confidence": round(float(confs.mean()), 4),
        "mean_accuracy":   round(float(accs.mean()), 4),
        "n_samples":       n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  All-in-one evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_reconstruction(
    reconstructed:     str,
    reference:         str,
    relative_seq:      str,
    expected_distance: float,
    confidences:       List[float],
    species_name:      str = "",
) -> Dict:
    """Run all 5 metrics on a single species reconstruction."""
    # Per-position correctness for calibration
    min_len = min(len(reconstructed), len(reference))
    actual_correct = [
        reconstructed[i].upper() == reference[i].upper()
        for i in range(min_len)
    ]

    # Trim confidences to match
    confs = confidences[:min_len]

    result = {
        "species":     species_name,
        "accuracy":    sequence_accuracy(reconstructed, reference),
        "edit_dist":   edit_distance(reconstructed, reference),
        "similarity":  reconstruction_similarity(reconstructed, reference),
        "phylo":       phylogenetic_consistency(
                           reconstructed, relative_seq,
                           expected_distance, species_name),
        "calibration": confidence_calibration(confs, actual_correct),
    }
    return result
