"""
genome_mapper.py
================
Maps short fragmented DNA reads to a reference genome.
Fixed: dynamic path resolution from config.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import MAP_DIR, MTDNA_GENES, DISEASE_MUTATIONS


def map_fragment_to_reference(fragment: str, ref_seq: str,
                               frag_start_hint: int = 0) -> Dict:
    k = min(12, len(fragment) - 1)
    if k < 4:
        return {"mapped": False, "reason": "fragment too short"}

    anchor = fragment[:k]
    best_pos, best_score = -1, -1

    search_start = max(0, frag_start_hint - 5000)
    search_end   = min(len(ref_seq) - k, frag_start_hint + 5000)
    if search_end <= search_start:
        search_start, search_end = 0, len(ref_seq) - k

    for pos in range(search_start, search_end):
        if ref_seq[pos:pos+k] == anchor:
            score = sum(1 for i, b in enumerate(fragment)
                        if pos+i < len(ref_seq) and ref_seq[pos+i] == b)
            if score > best_score:
                best_score = score
                best_pos   = pos

    if best_pos == -1:
        step = max(1, (search_end - search_start) // 1000)
        for pos in range(search_start, search_end, step):
            score = sum(1 for i, b in enumerate(fragment[:50])
                        if pos+i < len(ref_seq) and ref_seq[pos+i] == b)
            if score > best_score:
                best_score = score
                best_pos   = pos

    identity = best_score / max(1, len(fragment))
    gene_region = _find_gene_region(best_pos)

    variants, disease_hits = [], []
    for i, b in enumerate(fragment):
        ref_pos = best_pos + i
        if ref_pos < len(ref_seq):
            ref_b = ref_seq[ref_pos]
            if b != ref_b and b != "N":
                variants.append({"ref_pos": ref_pos, "ref_base": ref_b,
                                 "read_base": b, "type": "SNP"})
                if ref_pos in DISEASE_MUTATIONS:
                    mut, disease = DISEASE_MUTATIONS[ref_pos]
                    disease_hits.append({"ref_pos": ref_pos, "mutation": mut,
                                         "disease": disease,
                                         "observed": f"{ref_b}→{b}"})

    return {
        "mapped": True, "ref_start": best_pos,
        "ref_end": best_pos + len(fragment),
        "identity": round(identity, 4),
        "mapping_score": best_score, "gene_region": gene_region,
        "variants": variants, "disease_hits": disease_hits,
    }


def _find_gene_region(position: int) -> Dict:
    for gene in MTDNA_GENES:
        if gene["start"] <= position <= gene["end"]:
            return gene
    return {"gene": "intergenic", "type": "intergenic"}


def map_all_fragments(fragments: List[Dict], ref_seq: str,
                      species_name: str = "unnamed") -> Dict:
    mappings, all_variants = [], []
    print(f"  [MAP] Mapping {len(fragments)} fragments for {species_name}…")

    for idx, frag in enumerate(fragments[:500]):
        hint = frag.get("start", 0) % max(1, len(ref_seq))
        result = map_fragment_to_reference(frag["seq"], ref_seq, hint)
        result["fragment_id"]  = idx
        result["fragment_len"] = len(frag["seq"])
        mappings.append(result)
        all_variants.extend(result.get("variants", []))

    outliers = _detect_variant_hotspots(all_variants)

    summary = {
        "species":          species_name,
        "total_fragments":  len(fragments),
        "mapped_fragments": sum(1 for m in mappings if m["mapped"]),
        "total_variants":   len(all_variants),
        "hotspots":         outliers,
        "disease_hits":     [h for m in mappings
                            for h in m.get("disease_hits", [])],
        "mappings":         mappings,
    }

    out_path = os.path.join(MAP_DIR, f"{species_name}_mapping.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _detect_variant_hotspots(variants: List[Dict],
                              window: int = 200) -> List[Dict]:
    if not variants:
        return []
    positions = np.array([v["ref_pos"] for v in variants], dtype=float)
    if len(positions) < 3:
        return []
    densities = np.array([
        np.sum(np.abs(positions - p) <= window / 2) for p in positions
    ], dtype=float)
    mu, sig = densities.mean(), densities.std() + 1e-9
    hotspot_idx = np.where((densities - mu) / sig > 2.0)[0]
    hotspots, seen = [], set()
    for i in hotspot_idx:
        pos = int(positions[i])
        rk  = pos // 500
        if rk not in seen:
            seen.add(rk)
            gene = _find_gene_region(pos)
            hotspots.append({
                "center_pos": pos,
                "z_score": round(float((densities[i] - mu) / sig), 3),
                "local_density": int(densities[i]),
                "gene_region": gene,
            })
    return sorted(hotspots, key=lambda x: -x["z_score"])