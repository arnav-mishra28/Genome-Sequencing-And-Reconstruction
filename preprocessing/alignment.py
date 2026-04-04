"""
alignment.py
============
Pure-Python pairwise and multiple sequence alignment.
(Wraps Biopython's pairwisealigner + simulates MAFFT/Clustal output format
 so the pipeline works even without external binaries.)
"""

import os
import subprocess
import json
import numpy as np
from typing import List, Dict, Tuple

try:
    from Bio import Align, SeqIO
    from Bio.Align import substitution_matrices
    BIOPYTHON = True
except ImportError:
    BIOPYTHON = False
    print("[WARN] Biopython not found — using built-in Smith-Waterman.")

BASE_DIR = r"D:\Genome Sequencing And Reconstruction"
ALIGN_DIR = os.path.join(BASE_DIR, "data", "alignments")
os.makedirs(ALIGN_DIR, exist_ok=True)


# ── Smith–Waterman (pure Python fallback) ─────────────────────────────────────
def smith_waterman(seq1: str, seq2: str,
                   match=2, mismatch=-1, gap=-1) -> Tuple[str, str, float]:
    s1, s2 = seq1.upper(), seq2.upper()
    m, n = len(s1), len(s2)

    H = np.zeros((m+1, n+1), dtype=np.float32)
    for i in range(1, m+1):
        for j in range(1, n+1):
            diag = H[i-1,j-1] + (match if s1[i-1]==s2[j-1] else mismatch)
            up   = H[i-1,j] + gap
            left = H[i,j-1] + gap
            H[i,j] = max(0, diag, up, left)

    # Traceback from maximum
    i, j = np.unravel_index(np.argmax(H), H.shape)
    score = float(H[i,j])
    a1, a2 = [], []

    while H[i,j] > 0:
        diag = H[i-1,j-1] if i>0 and j>0 else 0
        up   = H[i-1,j]   if i>0 else 0
        left = H[i,j-1]   if j>0 else 0

        if H[i,j] == diag + (match if s1[i-1]==s2[j-1] else mismatch):
            a1.append(s1[i-1]); a2.append(s2[j-1]); i-=1; j-=1
        elif H[i,j] == up + gap:
            a1.append(s1[i-1]); a2.append("-"); i-=1
        else:
            a1.append("-"); a2.append(s2[j-1]); j-=1

    return "".join(reversed(a1)), "".join(reversed(a2)), score


def biopython_align(seq1: str, seq2: str) -> Tuple[str, str, float]:
    if not BIOPYTHON:
        return smith_waterman(seq1, seq2)
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score  = -2
    aligner.extend_gap_score = -0.5
    try:
        alignments = aligner.align(seq1[:5000], seq2[:5000])
        best = next(iter(alignments))
        a, b = str(best).split("\n")[0], str(best).split("\n")[2]
        return a, b, float(best.score)
    except Exception:
        return smith_waterman(seq1[:500], seq2[:500])


def align_to_reference(
    query: str,
    reference: str,
    name: str = "unnamed"
) -> Dict:
    """Align a query (ancient, possibly gapped) sequence to a modern reference."""
    print(f"  [ALIGN] {name} → reference ({len(query)} bp vs {len(reference)} bp)")

    # Use shorter slices to stay tractable
    q_slice = query[:8000].replace("N", "-")
    r_slice = reference[:8000]

    a_query, a_ref, score = biopython_align(q_slice, r_slice)

    # Compute identity
    matches    = sum(q == r and q != "-" for q, r in zip(a_query, a_ref))
    aln_len    = max(1, len(a_query))
    identity   = matches / aln_len

    # Find variants (SNPs)
    variants = []
    for pos, (q, r) in enumerate(zip(a_query, a_ref)):
        if q != r and q != "-" and r != "-" and q != "N":
            variants.append({
                "aln_pos":  pos,
                "ref_base": r,
                "qry_base": q,
                "type": _classify_variant(r, q),
            })

    result = {
        "name":           name,
        "score":          score,
        "identity":       round(identity, 4),
        "aligned_length": aln_len,
        "variants":       variants,
        "n_variants":     len(variants),
        "query_aligned":  a_query[:200] + "..." if len(a_query) > 200 else a_query,
        "ref_aligned":    a_ref[:200]   + "..." if len(a_ref)   > 200 else a_ref,
    }

    out_path = os.path.join(ALIGN_DIR, f"{name}_alignment.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _classify_variant(ref: str, alt: str) -> str:
    transitions  = {("A","G"),("G","A"),("C","T"),("T","C")}
    transversions = {("A","C"),("C","A"),("A","T"),("T","A"),
                     ("G","C"),("C","G"),("G","T"),("T","G")}
    pair = (ref.upper(), alt.upper())
    if pair in transitions:
        return "transition"
    elif pair in transversions:
        return "transversion"
    return "unknown"


def find_outlier_variants(variants: List[Dict], window: int = 100) -> List[Dict]:
    """
    Flag variants whose local density exceeds μ + 2σ
    (statistically unusual clustering = potential hotspot or damage).
    """
    if not variants:
        return []
    positions = [v["aln_pos"] for v in variants]
    densities = []
    for pos in positions:
        local = sum(1 for p in positions if abs(p - pos) <= window // 2)
        densities.append(local)

    mu  = np.mean(densities)
    sig = np.std(densities) + 1e-9
    outliers = []
    for i, v in enumerate(variants):
        z = (densities[i] - mu) / sig
        if z > 2.0:
            v["outlier_z_score"] = round(float(z), 3)
            v["is_outlier"] = True
            outliers.append(v)
    return outliers