"""
multi_species_loader.py
=======================
Download and prepare real genome datasets for multi-species evolutionary
training. Supports NCBI and Ensembl data sources.

Species supported:
  - Homo sapiens (human)
  - Pan troglodytes (chimpanzee)
  - Gorilla gorilla (gorilla)
  - Mus musculus (mouse)
  - Loxodonta africana (elephant)

Features:
  - FASTA download from NCBI/Ensembl
  - Sequence chunking with configurable window sizes
  - Corruption pipeline (deletions, mutations, missing chunks)
  - Multi-species merged dataset with species labels
  - Phylogenetic distance matrix builder
"""

import os
import sys
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import product

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import SEQ_DIR, SIM_DIR, DEVICE


# ═══════════════════════════════════════════════════════════════════════════════
#  Evolution Species Config
# ═══════════════════════════════════════════════════════════════════════════════
EVOLUTION_SPECIES = {
    "human":       {"accession": "NC_012920", "name": "Homo sapiens",           "taxon": "mammal"},
    "chimpanzee":  {"accession": "NC_001643", "name": "Pan troglodytes",        "taxon": "mammal"},
    "gorilla":     {"accession": "NC_011120", "name": "Gorilla gorilla",        "taxon": "mammal"},
    "mouse":       {"accession": "NC_005089", "name": "Mus musculus",           "taxon": "mammal"},
    "elephant":    {"accession": "NC_005129", "name": "Loxodonta africana",     "taxon": "mammal"},
}

# Phylogenetic distances (millions of years divergence)
EVOLUTION_DISTANCES = {
    ("human",      "chimpanzee"):  6.0,
    ("human",      "gorilla"):     9.0,
    ("chimpanzee", "gorilla"):     9.0,
    ("human",      "mouse"):      90.0,
    ("chimpanzee", "mouse"):      90.0,
    ("gorilla",    "mouse"):      90.0,
    ("human",      "elephant"):   90.0,
    ("chimpanzee", "elephant"):   90.0,
    ("gorilla",    "elephant"):   90.0,
    ("mouse",      "elephant"):   90.0,
}


def download_evolution_genomes(
    species: Dict[str, Dict] = None,
    output_dir: str = None,
) -> Dict[str, Dict]:
    """
    Download mtDNA genomes for evolutionary training from NCBI.

    Returns metadata dict: species_id → {path, length, accession, name}.
    """
    if species is None:
        species = EVOLUTION_SPECIES
    if output_dir is None:
        output_dir = os.path.join(SEQ_DIR, "evolution")
    os.makedirs(output_dir, exist_ok=True)

    from data.fetch_sequences import fetch_ncbi_sequence

    metadata = {}
    for sp_id, info in species.items():
        fasta_path = os.path.join(output_dir, f"{sp_id}_mtDNA.fasta")
        acc = info["accession"]

        if os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 100:
            print(f"  [EVO] {sp_id}: cached → {fasta_path}")
            # Read length
            from data.fetch_sequences import load_fasta
            recs = load_fasta(fasta_path)
            seq = next(iter(recs.values()), "")
            metadata[sp_id] = {
                "path": fasta_path,
                "length": len(seq),
                "accession": acc,
                "name": info["name"],
                "taxon": info.get("taxon", "mammal"),
            }
            continue

        print(f"  [EVO] Downloading {info['name']} ({acc})...")
        try:
            seq = fetch_ncbi_sequence(acc)
            if seq and len(seq) > 100:
                with open(fasta_path, "w") as f:
                    f.write(f">{sp_id}|{acc}|{info['name']}\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
                metadata[sp_id] = {
                    "path": fasta_path,
                    "length": len(seq),
                    "accession": acc,
                    "name": info["name"],
                    "taxon": info.get("taxon", "mammal"),
                }
                print(f"    ✅ {sp_id}: {len(seq):,} bp")
            else:
                print(f"    ⚠ {sp_id}: empty response, generating synthetic")
                seq = _generate_synthetic_genome(sp_id, 16500)
                with open(fasta_path, "w") as f:
                    f.write(f">{sp_id}|synthetic\n{seq}\n")
                metadata[sp_id] = {
                    "path": fasta_path, "length": len(seq),
                    "accession": "synthetic", "name": info["name"],
                    "taxon": info.get("taxon", "mammal"),
                }
        except Exception as e:
            print(f"    ⚠ {sp_id}: download failed ({e}), generating synthetic")
            seq = _generate_synthetic_genome(sp_id, 16500)
            with open(fasta_path, "w") as f:
                f.write(f">{sp_id}|synthetic\n{seq}\n")
            metadata[sp_id] = {
                "path": fasta_path, "length": len(seq),
                "accession": "synthetic", "name": info["name"],
                "taxon": info.get("taxon", "mammal"),
            }

    # Save metadata
    meta_path = os.path.join(output_dir, "evolution_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [EVO] Metadata → {meta_path}")

    return metadata


def _generate_synthetic_genome(species_id: str, length: int = 16500) -> str:
    """Generate a synthetic genome for fallback when download fails."""
    rng = random.Random(hash(species_id) % (2**31))
    # Use species-specific GC content to make it more realistic
    gc_content = {
        "human": 0.44, "chimpanzee": 0.44, "gorilla": 0.44,
        "mouse": 0.42, "elephant": 0.41,
    }.get(species_id, 0.43)

    seq = []
    for _ in range(length):
        if rng.random() < gc_content:
            seq.append(rng.choice("GC"))
        else:
            seq.append(rng.choice("AT"))
    return "".join(seq)


# ═══════════════════════════════════════════════════════════════════════════════
#  Sequence Chunking
# ═══════════════════════════════════════════════════════════════════════════════
def chunk_sequences(
    sequences: Dict[str, str],
    chunk_size: int = 512,
    stride: int = 256,
    min_chunk: int = 64,
) -> List[Dict]:
    """
    Chunk all species sequences into fixed-size windows with overlap.

    Returns list of dicts: {species, species_idx, chunk, start, end}
    """
    species_names = sorted(sequences.keys())
    chunks = []

    for sp_idx, sp_name in enumerate(species_names):
        seq = sequences[sp_name].upper()
        for start in range(0, len(seq) - min_chunk + 1, stride):
            end = min(start + chunk_size, len(seq))
            chunk = seq[start:end]
            if len(chunk) >= min_chunk:
                chunks.append({
                    "species": sp_name,
                    "species_idx": sp_idx,
                    "chunk": chunk,
                    "start": start,
                    "end": end,
                })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  Corruption Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def corrupt_sequence(
    sequence: str,
    deletion_rate: float = 0.05,
    mutation_rate: float = 0.08,
    gap_rate: float = 0.03,
    gap_max_len: int = 10,
    seed: int = None,
) -> Tuple[str, Dict]:
    """
    Apply biologically-inspired corruption to a DNA sequence.

    Corruption types:
      1. Point mutations (transitions favored 2:1 over transversions)
      2. Deletions (single base)
      3. Gap insertions (runs of N's)

    Returns: (corrupted_sequence, corruption_log)
    """
    rng = random.Random(seed)
    seq = list(sequence.upper())
    log = {"mutations": 0, "deletions": 0, "gaps": 0, "total_n": 0}

    TRANSITIONS = {"A": "G", "G": "A", "C": "T", "T": "C"}
    TRANSVERSIONS = {
        "A": ["C", "T"], "G": ["C", "T"],
        "C": ["A", "G"], "T": ["A", "G"],
    }

    for i in range(len(seq)):
        if seq[i] not in "ACGT":
            continue

        r = rng.random()

        # Point mutation
        if r < mutation_rate:
            if rng.random() < 0.67:  # 2:1 Ti/Tv ratio
                seq[i] = TRANSITIONS.get(seq[i], seq[i])
            else:
                seq[i] = rng.choice(TRANSVERSIONS.get(seq[i], ["A"]))
            log["mutations"] += 1

        # Deletion → N
        elif r < mutation_rate + deletion_rate:
            seq[i] = "N"
            log["deletions"] += 1

        # Gap insertion
        elif r < mutation_rate + deletion_rate + gap_rate:
            gap_len = rng.randint(1, gap_max_len)
            for j in range(i, min(i + gap_len, len(seq))):
                seq[j] = "N"
                log["gaps"] += 1

    log["total_n"] = seq.count("N")
    log["corruption_rate"] = round(log["total_n"] / max(1, len(seq)), 4)

    return "".join(seq), log


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-Species Merged Dataset
# ═══════════════════════════════════════════════════════════════════════════════
def build_multi_species_dataset(
    sequences: Dict[str, str],
    chunk_size: int = 512,
    stride: int = 256,
    corruption_rate: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[str]]:
    """
    Build a merged dataset from multiple species with corruption.

    Returns:
      - dataset: list of {species, species_idx, clean, corrupted, log}
      - species_names: ordered species list
    """
    species_names = sorted(sequences.keys())
    chunks = chunk_sequences(sequences, chunk_size, stride)

    dataset = []
    rng = random.Random(seed)

    for chunk_info in chunks:
        clean = chunk_info["chunk"]
        corrupted, log = corrupt_sequence(
            clean,
            deletion_rate=0.05,
            mutation_rate=corruption_rate,
            gap_rate=0.03,
            seed=rng.randint(0, 2**31),
        )
        dataset.append({
            "species": chunk_info["species"],
            "species_idx": chunk_info["species_idx"],
            "clean": clean,
            "corrupted": corrupted,
            "start": chunk_info["start"],
            "end": chunk_info["end"],
            "corruption_log": log,
        })

    print(f"  [MULTI] Built dataset: {len(dataset)} chunks from "
          f"{len(species_names)} species")
    return dataset, species_names


# ═══════════════════════════════════════════════════════════════════════════════
#  Phylogenetic Distance Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def build_phylo_distance_matrix(
    species_names: List[str],
    distances: Dict[Tuple[str, str], float] = None,
) -> np.ndarray:
    """
    Build NxN phylogenetic distance matrix.
    Uses EVOLUTION_DISTANCES by default.
    """
    if distances is None:
        distances = EVOLUTION_DISTANCES

    N = len(species_names)
    matrix = np.zeros((N, N), dtype=np.float32)

    for i, sp1 in enumerate(species_names):
        for j, sp2 in enumerate(species_names):
            if i == j:
                matrix[i, j] = 0.0
            else:
                d = distances.get((sp1, sp2),
                    distances.get((sp2, sp1), 100.0))
                matrix[i, j] = d

    return matrix


def build_phylo_adjacency(
    species_names: List[str],
    distances: Dict[Tuple[str, str], float] = None,
) -> np.ndarray:
    """
    Build adjacency matrix from phylogenetic distances.
    Weight = 1 / (1 + distance)
    """
    dist_matrix = build_phylo_distance_matrix(species_names, distances)
    N = len(species_names)
    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                adj[i, j] = 1.0
            else:
                adj[i, j] = 1.0 / (1.0 + dist_matrix[i, j])

    return adj


# ═══════════════════════════════════════════════════════════════════════════════
#  k-mer Frequency Vectors (for species features)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_species_features(
    sequences: Dict[str, str],
    k: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute k-mer frequency feature vectors for each species.
    Returns: (features_matrix, species_names)
    """
    species_names = sorted(sequences.keys())
    bases = "ACGT"
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_index = {km: i for i, km in enumerate(all_kmers)}
    n_features = len(all_kmers)

    features = np.zeros((len(species_names), n_features), dtype=np.float32)

    for i, sp in enumerate(species_names):
        seq = sequences[sp].upper().replace("N", "")
        vec = np.zeros(n_features, dtype=np.float32)
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_index:
                vec[kmer_index[kmer]] += 1
        total = vec.sum()
        if total > 0:
            vec /= total
        features[i] = vec

    return features, species_names


# ═══════════════════════════════════════════════════════════════════════════════
#  Load all evolution genomes into memory
# ═══════════════════════════════════════════════════════════════════════════════
def load_evolution_sequences(
    metadata: Dict[str, Dict] = None,
    max_length: int = 20000,
) -> Dict[str, str]:
    """Load all evolution species sequences from FASTA files."""
    if metadata is None:
        meta_path = os.path.join(SEQ_DIR, "evolution", "evolution_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            return {}

    from data.fetch_sequences import load_fasta

    sequences = {}
    for sp_id, info in metadata.items():
        path = info.get("path", "")
        if os.path.exists(path):
            recs = load_fasta(path)
            seq = next(iter(recs.values()), "")[:max_length]
            if seq:
                sequences[sp_id] = seq

    return sequences


if __name__ == "__main__":
    # Quick test
    print("Downloading evolution genomes...")
    meta = download_evolution_genomes()
    seqs = load_evolution_sequences(meta)
    print(f"\nLoaded {len(seqs)} species:")
    for sp, seq in seqs.items():
        print(f"  {sp}: {len(seq):,} bp")

    # Build dataset
    dataset, names = build_multi_species_dataset(seqs)
    print(f"\nDataset: {len(dataset)} chunks, species: {names}")
