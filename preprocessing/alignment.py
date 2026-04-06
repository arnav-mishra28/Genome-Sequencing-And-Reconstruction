"""
alignment.py
============
Pure-Python pairwise and multiple sequence alignment with optional MAFFT
binary support.
"""

import os
import sys
import subprocess
import json
import tempfile
import numpy as np
from typing import List, Dict, Tuple

try:
    from Bio import Align
    BIOPYTHON = True
except ImportError:
    BIOPYTHON = False
    print("[WARN] Biopython not found — using built-in Smith-Waterman.")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import ALIGN_DIR


# ═══════════════════════════════════════════════════════════════════════════════
#  Smith–Waterman (pure Python fallback)
# ═══════════════════════════════════════════════════════════════════════════════
def smith_waterman(seq1: str, seq2: str,
                   match: int = 2, mismatch: int = -1,
                   gap: int = -1) -> Tuple[str, str, float]:
    s1, s2 = seq1.upper(), seq2.upper()
    m, n = len(s1), len(s2)

    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = H[i-1, j-1] + (match if s1[i-1] == s2[j-1] else mismatch)
            up   = H[i-1, j] + gap
            left = H[i, j-1] + gap
            H[i, j] = max(0, diag, up, left)

    i, j = np.unravel_index(np.argmax(H), H.shape)
    score = float(H[i, j])
    a1, a2 = [], []

    while H[i, j] > 0:
        diag = H[i-1, j-1] if i > 0 and j > 0 else 0
        up   = H[i-1, j]   if i > 0 else 0

        sc = match if s1[i-1] == s2[j-1] else mismatch
        if H[i, j] == diag + sc:
            a1.append(s1[i-1]); a2.append(s2[j-1]); i -= 1; j -= 1
        elif H[i, j] == up + gap:
            a1.append(s1[i-1]); a2.append("-"); i -= 1
        else:
            a1.append("-"); a2.append(s2[j-1]); j -= 1

    return "".join(reversed(a1)), "".join(reversed(a2)), score


# ═══════════════════════════════════════════════════════════════════════════════
#  Biopython wrapper
# ═══════════════════════════════════════════════════════════════════════════════
def biopython_align(seq1: str, seq2: str) -> Tuple[str, str, float]:
    if not BIOPYTHON:
        return smith_waterman(seq1, seq2)
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score      = 2
    aligner.mismatch_score   = -1
    aligner.open_gap_score   = -2
    aligner.extend_gap_score = -0.5
    try:
        alignments = aligner.align(seq1[:5000], seq2[:5000])
        best = next(iter(alignments))
        a, b = str(best).split("\n")[0], str(best).split("\n")[2]
        return a, b, float(best.score)
    except Exception:
        return smith_waterman(seq1[:500], seq2[:500])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAFFT wrapper  (multiple sequence alignment)
# ═══════════════════════════════════════════════════════════════════════════════
def _mafft_available() -> bool:
    try:
        subprocess.run(["mafft", "--version"],
                       capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def mafft_align(sequences: Dict[str, str],
                algorithm: str = "--auto") -> Dict[str, str]:
    """
    Run MAFFT multiple sequence alignment.
    Returns dict of {name: aligned_seq}.
    Falls back to pairwise alignment if MAFFT is not installed.
    """
    if not _mafft_available():
        print("  [ALIGN] MAFFT not found — falling back to pairwise.")
        names = list(sequences.keys())
        if len(names) < 2:
            return sequences
        ref = sequences[names[0]]
        result = {names[0]: ref}
        for n in names[1:]:
            a, b, _ = biopython_align(sequences[n], ref)
            result[n] = a
        return result

    # Write temp FASTA
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta",
                                     delete=False) as tmp:
        for name, seq in sequences.items():
            tmp.write(f">{name}\n{seq}\n")
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["mafft", algorithm, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        aligned = _parse_fasta_string(result.stdout)
        print(f"  [MAFFT] Aligned {len(aligned)} sequences")
        return aligned
    except Exception as e:
        print(f"  [MAFFT ERR] {e}")
        return sequences
    finally:
        os.unlink(tmp_path)


def _parse_fasta_string(text: str) -> Dict[str, str]:
    records: Dict[str, str] = {}
    header, parts = None, []
    for line in text.strip().split("\n"):
        if line.startswith(">"):
            if header:
                records[header] = "".join(parts)
            header, parts = line[1:].strip(), []
        else:
            parts.append(line.strip())
    if header:
        records[header] = "".join(parts)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════
def align_to_reference(
    query: str, reference: str, name: str = "unnamed",
) -> Dict:
    """Align a query (ancient, possibly gapped) sequence to a modern reference."""
    print(f"  [ALIGN] {name} → reference "
          f"({len(query)} bp vs {len(reference)} bp)")

    q_slice = query[:8000].replace("N", "-")
    r_slice = reference[:8000]

    a_query, a_ref, score = biopython_align(q_slice, r_slice)

    matches  = sum(q == r and q != "-" for q, r in zip(a_query, a_ref))
    aln_len  = max(1, len(a_query))
    identity = matches / aln_len

    variants = []
    for pos, (q, r) in enumerate(zip(a_query, a_ref)):
        if q != r and q != "-" and r != "-" and q != "N":
            variants.append({
                "aln_pos":  pos,
                "ref_base": r,
                "qry_base": q,
                "type":     _classify_variant(r, q),
            })

    result = {
        "name":           name,
        "score":          score,
        "identity":       round(identity, 4),
        "aligned_length": aln_len,
        "variants":       variants,
        "n_variants":     len(variants),
        "query_aligned":  (a_query[:200] + "…") if len(a_query) > 200
                          else a_query,
        "ref_aligned":    (a_ref[:200] + "…") if len(a_ref) > 200
                          else a_ref,
    }

    out_path = os.path.join(ALIGN_DIR, f"{name}_alignment.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _classify_variant(ref: str, alt: str) -> str:
    transitions  = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    transversions = {("A", "C"), ("C", "A"), ("A", "T"), ("T", "A"),
                     ("G", "C"), ("C", "G"), ("G", "T"), ("T", "G")}
    pair = (ref.upper(), alt.upper())
    if pair in transitions:
        return "transition"
    elif pair in transversions:
        return "transversion"
    return "unknown"


def find_outlier_variants(variants: List[Dict],
                          window: int = 100) -> List[Dict]:
    """Flag variants with statistically unusual local density."""
    if not variants:
        return []
    positions = [v["aln_pos"] for v in variants]
    densities = [
        sum(1 for p in positions if abs(p - pos) <= window // 2)
        for pos in positions
    ]
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