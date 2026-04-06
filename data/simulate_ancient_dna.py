"""
simulate_ancient_dna.py
=======================
Takes real sequences and simulates ancient DNA damage with biologically
accurate damage profiles:
  1. C→T / G→A deamination (position-dependent rate curves)
  2. Oxidative G→T damage (8-oxoG)
  3. Random substitutions
  4. Small insertions / deletions
  5. Large missing segments (gaps → 'N')
  6. Fragmentation into short reads
  7. Strand-bias simulation
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import SIM_DIR

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

MUTATION_INFO = {
    "deamination_C_to_T": {
        "change": "C→T",
        "mechanism": ("Hydrolytic deamination of cytosine removes amino group, "
                      "converting it to uracil (read as thymine). "
                      "Hallmark of ancient DNA."),
        "disease_relevance": ("Can create false KRAS/BRAF mutations if not "
                              "corrected; associated with UV-induced skin "
                              "cancer in living cells."),
    },
    "deamination_G_to_A": {
        "change": "G→A",
        "mechanism": ("Reverse-strand complement of C→T deamination. "
                      "Common at 3′ ends."),
        "disease_relevance": ("Similar to oncogenic point mutations seen in "
                              "RAS family genes."),
    },
    "oxidative_G_to_T": {
        "change": "G→T",
        "mechanism": ("8-oxoguanine formation via reactive oxygen species. "
                      "Oxidative DNA damage from ancient burial environment."),
        "disease_relevance": ("Transversion associated with TP53 mutations "
                              "in lung cancer."),
    },
    "random_substitution": {
        "change": "N→N",
        "mechanism": ("Random base substitution from background radiation, "
                      "chemical damage, or sequencing error."),
        "disease_relevance": "Depends on genomic context.",
    },
    "missing_segment": {
        "change": "deletion of region",
        "mechanism": ("Physical degradation of DNA backbone via hydrolysis "
                      "over millennia."),
        "disease_relevance": ("Large deletions can eliminate entire exons "
                              "or regulatory elements."),
    },
    "insertion": {
        "change": "insertion",
        "mechanism": ("Replication slippage or transposable element "
                      "insertion artefact."),
        "disease_relevance": ("Frameshift mutations; Huntington's disease "
                              "is caused by CAG repeat expansion."),
    },
    "deletion": {
        "change": "small deletion",
        "mechanism": ("Exonuclease activity or strand break re-ligation "
                      "error."),
        "disease_relevance": ("Frameshift; analogous to CFTR ΔF508 deletion "
                              "in cystic fibrosis."),
    },
}


# ── Position-dependent deamination curve (empirical) ──────────────────────────
def _deamination_rate_curve(pos: int, length: int,
                            base_rate: float) -> float:
    """
    Returns elevated deamination rate at fragment ends (5′ and 3′)
    following the empirical exponential decay observed in real aDNA papers
    (Briggs et al. 2007, Dabney et al. 2013).
    """
    dist_from_end = min(pos, length - 1 - pos)
    if dist_from_end < 5:
        return base_rate * 4.0
    elif dist_from_end < 15:
        return base_rate * 2.5 * np.exp(-0.1 * (dist_from_end - 5))
    elif dist_from_end < 30:
        return base_rate * 1.2
    return base_rate * 0.3


def simulate_ancient_damage(
    sequence: str,
    name: str,
    seed: int = 42,
    deamination_rate: float = 0.08,
    mutation_rate: float = 0.02,
    oxidation_rate: float = 0.01,
    deletion_rate: float = 0.005,
    insertion_rate: float = 0.003,
    gap_count: int = 5,
    gap_size_range: Tuple[int, int] = (50, 300),
    fragment: bool = True,
    fragment_size_range: Tuple[int, int] = (50, 150),
    strand_bias: float = 0.7,
) -> Dict:
    """Apply layered ancient DNA damage to a sequence."""
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    seq = list(sequence.upper().replace("N", rng.choice("ACGT")))
    mutation_log: List[Dict] = []
    length = len(seq)

    # ── 1. Position-dependent deamination ─────────────────────────────────────
    for i in range(length):
        rate = _deamination_rate_curve(i, length, deamination_rate)
        # Strand bias: C→T on forward strand, G→A on reverse
        if seq[i] == "C" and rng.random() < rate * strand_bias:
            mutation_log.append({
                "type": "deamination_C_to_T",
                "position": i, "original": "C", "mutated": "T",
                **MUTATION_INFO["deamination_C_to_T"],
            })
            seq[i] = "T"
        elif seq[i] == "G" and rng.random() < rate * (1 - strand_bias + 0.3):
            mutation_log.append({
                "type": "deamination_G_to_A",
                "position": i, "original": "G", "mutated": "A",
                **MUTATION_INFO["deamination_G_to_A"],
            })
            seq[i] = "A"

    # ── 2. Oxidative damage ───────────────────────────────────────────────────
    for i in range(length):
        if seq[i] == "G" and rng.random() < oxidation_rate:
            mutation_log.append({
                "type": "oxidative_G_to_T",
                "position": i, "original": "G", "mutated": "T",
                **MUTATION_INFO["oxidative_G_to_T"],
            })
            seq[i] = "T"

    # ── 3. Random substitutions ───────────────────────────────────────────────
    for i in range(length):
        if rng.random() < mutation_rate:
            orig = seq[i]
            choices = [b for b in "ACGT" if b != orig]
            mutated = rng.choice(choices)
            mutation_log.append({
                "type": "random_substitution",
                "position": i, "original": orig, "mutated": mutated,
                "change": f"{orig}→{mutated}",
                **MUTATION_INFO["random_substitution"],
            })
            seq[i] = mutated

    # ── 4. Small deletions ────────────────────────────────────────────────────
    del_positions = sorted(
        [i for i in range(length) if rng.random() < deletion_rate],
        reverse=True,
    )
    for i in del_positions:
        if i < len(seq):
            mutation_log.append({
                "type": "deletion",
                "position": i, "original": seq[i], "mutated": "",
                **MUTATION_INFO["deletion"],
            })
            seq.pop(i)

    # ── 5. Small insertions ───────────────────────────────────────────────────
    ins_positions = sorted(
        [i for i in range(len(seq)) if rng.random() < insertion_rate],
        reverse=True,
    )
    for i in ins_positions:
        ins_base = rng.choice("ACGT")
        mutation_log.append({
            "type": "insertion",
            "position": i, "original": "", "mutated": ins_base,
            **MUTATION_INFO["insertion"],
        })
        seq.insert(i, ins_base)

    # ── 6. Large missing segments (gaps → 'N') ───────────────────────────────
    seq_len_now = len(seq)
    gap_positions = []
    for _ in range(gap_count):
        gap_start = rng.randint(0, seq_len_now - 1)
        gap_size  = rng.randint(*gap_size_range)
        gap_end   = min(gap_start + gap_size, seq_len_now)
        orig_seg  = "".join(seq[gap_start:gap_end])
        for j in range(gap_start, gap_end):
            seq[j] = "N"
        mutation_log.append({
            "type": "missing_segment",
            "position": gap_start, "end": gap_end,
            "original": orig_seg[:20] + "…",
            "mutated": "N" * (gap_end - gap_start),
            **MUTATION_INFO["missing_segment"],
        })
        gap_positions.append((gap_start, gap_end))

    damaged_seq = "".join(seq)

    # ── 7. Fragmentation into short reads ─────────────────────────────────────
    fragments = []
    if fragment:
        pos = 0
        while pos < len(damaged_seq):
            frag_len = rng.randint(*fragment_size_range)
            frag = damaged_seq[pos : pos + frag_len]
            if len(frag) > 20:
                fragments.append({
                    "start":  pos,
                    "end":    pos + len(frag),
                    "seq":    frag,
                    "length": len(frag),
                })
            pos += frag_len

    return {
        "name":              name,
        "original_length":   length,
        "damaged_length":    len(damaged_seq),
        "damaged_sequence":  damaged_seq,
        "fragments":         fragments,
        "mutation_log":      mutation_log,
        "mutation_summary":  _summarize_mutations(mutation_log),
        "gap_positions":     gap_positions,
    }


def _summarize_mutations(log: List[Dict]) -> Dict:
    summary: Dict[str, int] = {}
    for m in log:
        t = m["type"]
        summary[t] = summary.get(t, 0) + 1
    return summary


def simulate_all(metadata_path: str):
    """Run simulation on every fetched sequence."""
    from data.fetch_sequences import load_fasta

    with open(metadata_path) as f:
        metadata = json.load(f)

    all_results = {}
    for name, info in metadata.items():
        print(f"\n[SIM] Simulating ancient damage → {name}")
        records = load_fasta(info["path"])
        seq = next(iter(records.values()))[:20_000]

        result = simulate_ancient_damage(
            sequence=seq, name=name,
            seed=hash(name) % (2**31),
            deamination_rate=0.07,
            mutation_rate=0.02,
            gap_count=6,
        )
        out_path = os.path.join(SIM_DIR, f"{name}_simulated.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        summary = result["mutation_summary"]
        total   = sum(summary.values())
        print(f"  → {total} mutations applied: {summary}")
        all_results[name] = out_path

    index_path = os.path.join(SIM_DIR, "simulation_index.json")
    with open(index_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


if __name__ == "__main__":
    from config.settings import SEQ_DIR
    meta_path = os.path.join(SEQ_DIR, "metadata.json")
    simulate_all(meta_path)