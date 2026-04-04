"""
simulate_ancient_dna.py
=======================
Takes real sequences and simulates ancient DNA damage:
  1. C→T / G→A deamination at fragment ends (most common ancient damage)
  2. Random point mutations (substitutions)
  3. Missing segments (gaps)
  4. Cytosine deamination interior
  5. Fragmentation (split into short reads)
  6. Insertions / Deletions

Every mutation is recorded with its type, position, original base,
mutated base, and biological significance.
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple

BASE_DIR = r"D:\Genome Sequencing And Reconstruction"
SIM_DIR  = os.path.join(BASE_DIR, "data", "simulated")
os.makedirs(SIM_DIR, exist_ok=True)

# Complement map
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

# ── Mutation type descriptions ─────────────────────────────────────────────────
MUTATION_INFO = {
    "deamination_C_to_T": {
        "change": "C→T",
        "mechanism": "Hydrolytic deamination of cytosine removes amino group, "
                     "converting it to uracil (read as thymine). Hallmark of ancient DNA.",
        "disease_relevance": "Can create false KRAS/BRAF mutations if not corrected; "
                             "associated with UV-induced skin cancer in living cells.",
    },
    "deamination_G_to_A": {
        "change": "G→A",
        "mechanism": "Reverse-strand complement of C→T deamination. Common at 3′ ends.",
        "disease_relevance": "Similar to oncogenic point mutations seen in RAS family genes.",
    },
    "oxidative_G_to_T": {
        "change": "G→T",
        "mechanism": "8-oxoguanine formation via reactive oxygen species. "
                     "Oxidative DNA damage from ancient burial environment.",
        "disease_relevance": "Transversion associated with TP53 mutations in lung cancer.",
    },
    "random_substitution": {
        "change": "N→N",
        "mechanism": "Random base substitution from background radiation, "
                     "chemical damage, or sequencing error.",
        "disease_relevance": "Depends on genomic context; may disrupt coding regions.",
    },
    "missing_segment": {
        "change": "deletion of region",
        "mechanism": "Physical degradation of DNA backbone via hydrolysis over millennia.",
        "disease_relevance": "Large deletions can eliminate entire exons or regulatory elements, "
                             "analogous to BRCA1/2 large deletions.",
    },
    "insertion": {
        "change": "insertion",
        "mechanism": "Replication slippage or transposable element insertion artefact.",
        "disease_relevance": "Frameshift mutations; Huntington's disease is caused by "
                             "CAG repeat expansion insertions.",
    },
    "deletion": {
        "change": "small deletion",
        "mechanism": "Exonuclease activity or strand break re-ligation error.",
        "disease_relevance": "Frameshift; analogous to CFTR ΔF508 deletion in cystic fibrosis.",
    },
}


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
) -> Dict:
    """
    Apply layered ancient DNA damage to a sequence.
    Returns dict with damaged sequence, fragments, and full mutation log.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    seq = list(sequence.upper().replace("N", rng.choice("ACGT")))
    mutation_log: List[Dict] = []
    length = len(seq)

    # ── 1. Deamination at fragment ends (5′ and 3′ bias) ─────────────────────
    end_zone = max(10, int(length * 0.05))
    for i in list(range(end_zone)) + list(range(length - end_zone, length)):
        if seq[i] == "C" and rng.random() < deamination_rate * 2:
            mutation_log.append({
                "type": "deamination_C_to_T",
                "position": i,
                "original": "C",
                "mutated": "T",
                **MUTATION_INFO["deamination_C_to_T"],
            })
            seq[i] = "T"
        elif seq[i] == "G" and rng.random() < deamination_rate * 1.5:
            mutation_log.append({
                "type": "deamination_G_to_A",
                "position": i,
                "original": "G",
                "mutated": "A",
                **MUTATION_INFO["deamination_G_to_A"],
            })
            seq[i] = "A"

    # ── 2. Interior deamination ────────────────────────────────────────────────
    for i in range(end_zone, length - end_zone):
        if seq[i] == "C" and rng.random() < deamination_rate * 0.3:
            mutation_log.append({
                "type": "deamination_C_to_T",
                "position": i,
                "original": "C",
                "mutated": "T",
                **MUTATION_INFO["deamination_C_to_T"],
            })
            seq[i] = "T"

    # ── 3. Oxidative damage ────────────────────────────────────────────────────
    for i in range(length):
        if seq[i] == "G" and rng.random() < oxidation_rate:
            mutation_log.append({
                "type": "oxidative_G_to_T",
                "position": i,
                "original": "G",
                "mutated": "T",
                **MUTATION_INFO["oxidative_G_to_T"],
            })
            seq[i] = "T"

    # ── 4. Random substitutions ────────────────────────────────────────────────
    for i in range(length):
        if rng.random() < mutation_rate:
            orig = seq[i]
            choices = [b for b in "ACGT" if b != orig]
            mutated = rng.choice(choices)
            mutation_log.append({
                "type": "random_substitution",
                "position": i,
                "original": orig,
                "mutated": mutated,
                "change": f"{orig}→{mutated}",
                **MUTATION_INFO["random_substitution"],
            })
            seq[i] = mutated

    # ── 5. Small deletions ─────────────────────────────────────────────────────
    del_positions = sorted(
        [i for i in range(length) if rng.random() < deletion_rate], reverse=True
    )
    for i in del_positions:
        if i < len(seq):
            mutation_log.append({
                "type": "deletion",
                "position": i,
                "original": seq[i],
                "mutated": "",
                **MUTATION_INFO["deletion"],
            })
            seq.pop(i)

    # ── 6. Small insertions ────────────────────────────────────────────────────
    ins_positions = sorted(
        [i for i in range(len(seq)) if rng.random() < insertion_rate], reverse=True
    )
    for i in ins_positions:
        ins_base = rng.choice("ACGT")
        mutation_log.append({
            "type": "insertion",
            "position": i,
            "original": "",
            "mutated": ins_base,
            **MUTATION_INFO["insertion"],
        })
        seq.insert(i, ins_base)

    # ── 7. Large missing segments (gaps → 'N') ─────────────────────────────────
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
            "position": gap_start,
            "end": gap_end,
            "original": orig_seg[:20] + "...",
            "mutated": "N" * (gap_end - gap_start),
            **MUTATION_INFO["missing_segment"],
        })
        gap_positions.append((gap_start, gap_end))

    damaged_seq = "".join(seq)

    # ── 8. Fragmentation into short reads ─────────────────────────────────────
    fragments = []
    if fragment:
        pos = 0
        while pos < len(damaged_seq):
            frag_len = rng.randint(*fragment_size_range)
            frag = damaged_seq[pos:pos + frag_len]
            if len(frag) > 20:   # minimum read length
                fragments.append({
                    "start": pos,
                    "end":   pos + len(frag),
                    "seq":   frag,
                    "length": len(frag),
                })
            pos += frag_len

    result = {
        "name": name,
        "original_length": length,
        "damaged_length": len(damaged_seq),
        "damaged_sequence": damaged_seq,
        "fragments": fragments,
        "mutation_log": mutation_log,
        "mutation_summary": _summarize_mutations(mutation_log),
        "gap_positions": gap_positions,
    }
    return result


def _summarize_mutations(log: List[Dict]) -> Dict:
    summary = {}
    for m in log:
        t = m["type"]
        summary[t] = summary.get(t, 0) + 1
    return summary


def simulate_all(metadata_path: str):
    """Run simulation on every fetched sequence."""
    from fetch_sequences import load_fasta

    with open(metadata_path) as f:
        metadata = json.load(f)

    all_results = {}
    for name, info in metadata.items():
        print(f"\n[SIM] Simulating ancient damage → {name}")
        records = load_fasta(info["path"])
        seq = next(iter(records.values()))
        # Use first 20,000 bp for speed (mtDNA is ~16 kbp anyway)
        seq = seq[:20_000]

        result = simulate_ancient_damage(
            sequence=seq,
            name=name,
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
    print(f"\n[INDEX] Simulation index → {index_path}")
    return all_results


if __name__ == "__main__":
    meta_path = os.path.join(
        BASE_DIR, "data", "sequences", "metadata.json"
    )
    simulate_all(meta_path)