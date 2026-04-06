"""
benchmark.py
============
End-to-end benchmarking suite.
Runs all 5 metrics on all species and generates a report.
"""

import os
import sys
import json
from typing import Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import RESULTS_DIR, REF_MAP, PHYLO_DISTANCES
from evaluation.metrics import evaluate_reconstruction


def run_benchmark(
    reconstructions: Dict[str, Dict],
    sequences_raw:   Dict[str, str],
    simulated:       Dict[str, Dict],
    modern_seqs:     Dict[str, str],
) -> Dict:
    """
    Run full benchmark across all species.

    Returns dict with per-species metrics + aggregate summary.
    """
    print("\n" + "=" * 65)
    print("  📊 BENCHMARK — Evaluation Metrics")
    print("=" * 65)

    all_results = {}

    for name, recon_info in reconstructions.items():
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref_seq  = modern_seqs.get(ref_name, "")
        original = sequences_raw.get(name, "")

        # Get reconstructed sequence (may be truncated in JSON)
        recon_seq = recon_info.get("reconstructed_seq", "")
        # Strip truncation markers (Unicode ellipsis or ASCII dots)
        recon_seq = recon_seq.rstrip("\u2026")  # Unicode '…'
        if recon_seq.endswith("..."):
            recon_seq = recon_seq[:-3]
        # Remove any non-DNA characters
        recon_seq = "".join(c for c in recon_seq if c in "ACGTNacgtn")

        # Get confidence scores
        confs = recon_info.get("confidences", [0.5] * len(recon_seq))

        # Get expected phylogenetic distance
        exp_dist = 90.0  # default
        for (sp1, sp2), d in PHYLO_DISTANCES.items():
            if sp1 == name or sp2 == name:
                exp_dist = d
                break

        result = evaluate_reconstruction(
            reconstructed=recon_seq,
            reference=original,
            relative_seq=ref_seq,
            expected_distance=exp_dist,
            confidences=confs,
            species_name=name,
        )
        all_results[name] = result

        # Print summary
        acc  = result["accuracy"]["accuracy"]
        edit = result["edit_dist"]["normalised"]
        sim  = result["similarity"]["overall_identity"]
        phyl = result["phylo"]["consistency_score"]
        ece  = result["calibration"]["ece"]

        print(f"\n  {name}:")
        print(f"    Accuracy:          {acc:.4f}")
        print(f"    Edit Distance:     {edit:.4f} (normalised)")
        print(f"    Similarity:        {sim:.4f}")
        print(f"    Phylo Consistency: {phyl:.4f}")
        print(f"    Calibration (ECE): {ece:.4f}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    if all_results:
        avg_acc  = sum(r["accuracy"]["accuracy"]
                       for r in all_results.values()) / len(all_results)
        avg_sim  = sum(r["similarity"]["overall_identity"]
                       for r in all_results.values()) / len(all_results)
        avg_phyl = sum(r["phylo"]["consistency_score"]
                       for r in all_results.values()) / len(all_results)
        avg_ece  = sum(r["calibration"]["ece"]
                       for r in all_results.values()) / len(all_results)

        summary = {
            "n_species":         len(all_results),
            "avg_accuracy":      round(avg_acc, 4),
            "avg_similarity":    round(avg_sim, 4),
            "avg_phylo_consist": round(avg_phyl, 4),
            "avg_ece":           round(avg_ece, 4),
        }

        print(f"\n  ── Aggregate ──")
        print(f"    Avg Accuracy:          {avg_acc:.4f}")
        print(f"    Avg Similarity:        {avg_sim:.4f}")
        print(f"    Avg Phylo Consistency: {avg_phyl:.4f}")
        print(f"    Avg Calibration (ECE): {avg_ece:.4f}")
    else:
        summary = {}

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "per_species": all_results,
        "aggregate":   summary,
    }

    report_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  [REPORT] Saved → {report_path}")

    return report
