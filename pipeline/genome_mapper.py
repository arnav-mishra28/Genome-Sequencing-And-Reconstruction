"""
genome_mapper.py
================
Maps short fragmented DNA reads to a reference genome.
Identifies:
  - Origin of each fragment
  - Chromosomal region (simulated for mtDNA)
  - Mutations / variants at each locus
  - Outlier variation clusters (hotspots)
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple

BASE_DIR  = r"D:\Genome Sequencing And Reconstruction"
MAP_DIR   = os.path.join(BASE_DIR, "data", "mappings")
os.makedirs(MAP_DIR, exist_ok=True)


# mtDNA "gene map" — approximate positions for human mtDNA NC_012920
MTDNA_GENES = [
    {"gene": "D-loop",       "start": 1,     "end": 576,   "type": "control"},
    {"gene": "tRNA-Phe",     "start": 577,   "end": 647,   "type": "tRNA"},
    {"gene": "12S rRNA",     "start": 648,   "end": 1601,  "type": "rRNA"},
    {"gene": "tRNA-Val",     "start": 1602,  "end": 1670,  "type": "tRNA"},
    {"gene": "16S rRNA",     "start": 1671,  "end": 3229,  "type": "rRNA"},
    {"gene": "ND1",          "start": 3307,  "end": 4262,  "type": "protein_coding"},
    {"gene": "ND2",          "start": 4470,  "end": 5511,  "type": "protein_coding"},
    {"gene": "COX1",         "start": 5904,  "end": 7445,  "type": "protein_coding"},
    {"gene": "COX2",         "start": 7586,  "end": 8269,  "type": "protein_coding"},
    {"gene": "ATP8",         "start": 8366,  "end": 8572,  "type": "protein_coding"},
    {"gene": "ATP6",         "start": 8527,  "end": 9207,  "type": "protein_coding"},
    {"gene": "COX3",         "start": 9207,  "end": 9990,  "type": "protein_coding"},
    {"gene": "ND4L",         "start": 10059, "end": 10404, "type": "protein_coding"},
    {"gene": "ND4",          "start": 10404, "end": 11935, "type": "protein_coding"},
    {"gene": "ND5",          "start": 11742, "end": 13565, "type": "protein_coding"},
    {"gene": "ND6",          "start": 13552, "end": 14070, "type": "protein_coding"},
    {"gene": "Cytochrome-b", "start": 14747, "end": 15887, "type": "protein_coding"},
    {"gene": "D-loop2",      "start": 15888, "end": 16569, "type": "control"},
]

DISEASE_MUTATIONS = {
    # Position : (mutation, disease association)
    3243:  ("A→G", "MELAS syndrome — mitochondrial encephalomyopathy"),
    8344:  ("A→G", "MERRF syndrome — myoclonic epilepsy"),
    11778: ("G→A", "Leber's Hereditary Optic Neuropathy (LHON)"),
    3460:  ("G→A", "LHON — ND1 subunit"),
    14484: ("T→C", "LHON — cytochrome b region"),
    4160:  ("T→C", "LHON — mild ND1 variant"),
    7444:  ("G→A", "LHON — COX1 variant"),
}


def map_fragment_to_reference(
    fragment: str,
    ref_seq: str,
    frag_start_hint: int = 0,
) -> Dict:
    """
    Find best alignment position of fragment in reference using
    sliding k-mer anchor approach (fast approximate mapping).
    """
    k = min(12, len(fragment) - 1)
    if k < 4:
        return {"mapped": False, "reason": "fragment too short"}

    # Build k-mer index of reference (first call only ideally, but kept simple)
    anchor = fragment[:k]
    best_pos, best_score = -1, -1

    # Search in a window around hint (±5000 bp)
    search_start = max(0, frag_start_hint - 5000)
    search_end   = min(len(ref_seq) - k, frag_start_hint + 5000)
    if search_end <= search_start:
        search_start, search_end = 0, len(ref_seq) - k

    for pos in range(search_start, search_end):
        if ref_seq[pos:pos+k] == anchor:
            # Count matching bases
            score = sum(
                1 for i, b in enumerate(fragment)
                if pos+i < len(ref_seq) and ref_seq[pos+i] == b
            )
            if score > best_score:
                best_score = score
                best_pos   = pos

    if best_pos == -1:
        # Fallback: just pick position with highest identity in windows
        step = max(1, (search_end - search_start) // 1000)
        for pos in range(search_start, search_end, step):
            score = sum(
                1 for i, b in enumerate(fragment[:50])
                if pos+i < len(ref_seq) and ref_seq[pos+i] == b
            )
            if score > best_score:
                best_score = score
                best_pos   = pos

    identity = best_score / max(1, len(fragment))
    gene_region = _find_gene_region(best_pos)

    # Variants at this locus
    variants = []
    disease_hits = []
    for i, b in enumerate(fragment):
        ref_pos = best_pos + i
        if ref_pos < len(ref_seq):
            ref_b = ref_seq[ref_pos]
            if b != ref_b and b != "N":
                variants.append({
                    "ref_pos": ref_pos,
                    "ref_base": ref_b,
                    "read_base": b,
                    "type": "SNP",
                })
                if ref_pos in DISEASE_MUTATIONS:
                    mut, disease = DISEASE_MUTATIONS[ref_pos]
                    disease_hits.append({
                        "ref_pos": ref_pos,
                        "mutation": mut,
                        "disease": disease,
                        "observed": f"{ref_b}→{b}",
                    })

    return {
        "mapped":       True,
        "ref_start":    best_pos,
        "ref_end":      best_pos + len(fragment),
        "identity":     round(identity, 4),
        "mapping_score": best_score,
        "gene_region":  gene_region,
        "variants":     variants,
        "disease_hits": disease_hits,
    }


def _find_gene_region(position: int) -> Dict:
    for gene in MTDNA_GENES:
        if gene["start"] <= position <= gene["end"]:
            return gene
    return {"gene": "intergenic", "type": "intergenic"}


def map_all_fragments(
    fragments: List[Dict],
    ref_seq: str,
    species_name: str = "unnamed",
) -> Dict:
    """Map all fragments from a simulated ancient DNA result."""
    mappings = []
    all_variants = []

    print(f"  [MAP] Mapping {len(fragments)} fragments for {species_name}...")
    for idx, frag in enumerate(fragments[:500]):  # cap at 500 for speed
        hint = frag.get("start", 0) % max(1, len(ref_seq))
        result = map_fragment_to_reference(frag["seq"], ref_seq, hint)
        result["fragment_id"]  = idx
        result["fragment_len"] = len(frag["seq"])
        mappings.append(result)
        all_variants.extend(result.get("variants", []))

    # Outlier detection
    outliers = _detect_variant_hotspots(all_variants)

    summary = {
        "species":          species_name,
        "total_fragments":  len(fragments),
        "mapped_fragments": sum(1 for m in mappings if m["mapped"]),
        "total_variants":   len(all_variants),
        "hotspots":         outliers,
        "disease_hits":     [h for m in mappings for h in m.get("disease_hits", [])],
        "mappings":         mappings,
    }

    out_path = os.path.join(MAP_DIR, f"{species_name}_mapping.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _detect_variant_hotspots(variants: List[Dict], window: int = 200) -> List[Dict]:
    if not variants:
        return []
    positions = np.array([v["ref_pos"] for v in variants], dtype=float)
    if len(positions) < 3:
        return []
    densities = np.array([
        np.sum(np.abs(positions - p) <= window/2) for p in positions
    ], dtype=float)
    mu, sig = densities.mean(), densities.std() + 1e-9
    hotspot_idx = np.where((densities - mu) / sig > 2.0)[0]
    hotspots = []
    seen = set()
    for i in hotspot_idx:
        pos = int(positions[i])
        region_key = pos // 500
        if region_key not in seen:
            seen.add(region_key)
            gene = _find_gene_region(pos)
            hotspots.append({
                "center_pos":   pos,
                "z_score":      round(float((densities[i]-mu)/sig), 3),
                "local_density": int(densities[i]),
                "gene_region":  gene,
            })
    return sorted(hotspots, key=lambda x: -x["z_score"])