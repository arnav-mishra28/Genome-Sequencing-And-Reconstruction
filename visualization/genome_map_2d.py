"""
genome_map_2d.py
================
2D Genome / Chromosome Map using pure matplotlib.
Generates high-resolution multi-panel PNG images showing:
  1. Gene map (mtDNA ideogram with colored gene bands)
  2. Variant density histogram
  3. Hotspot & disease loci
  4. Sequence coverage heatmap
  5. Mutation type distribution
  6. Reconstruction comparison (before/after per species)
  7. Benchmark radar chart (5 metrics)
  8. Phylogenetic dendrogram
  9. Confidence calibration plot
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import VIZ_DIR, RESULTS_DIR
from typing import Dict, List, Any

GENE_COLORS = {
    "protein_coding": "#4CAF50",
    "rRNA":           "#2196F3",
    "tRNA":           "#FF9800",
    "control":        "#9C27B0",
    "intergenic":     "#607D8B",
}

BASE_COLORS = {"A": "#00CC44", "C": "#0066FF", "G": "#FF8800",
               "T": "#FF2222", "N": "#555555"}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_CLR  = "#c9d1d9"
GRID_CLR  = "#21262d"
ACCENT    = "#58a6ff"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Per-species chromosome ideogram
# ═══════════════════════════════════════════════════════════════════════════════
def create_chromosome_ideogram(
    gene_map:     List[Dict],
    variants:     List[Dict],
    hotspots:     List[Dict],
    disease_hits: List[Dict],
    title:        str = "mtDNA Map",
) -> plt.Figure:
    """5-panel chromosome ideogram."""
    CHROM_LEN = 16569

    fig, axes = plt.subplots(5, 1, figsize=(18, 14),
                             facecolor=DARK_BG,
                             gridspec_kw={"height_ratios": [1, 1.2, 1, 1.2, 0.8]})
    fig.suptitle(title, fontsize=16, color="white", fontweight="bold", y=0.98)
    panel_names = ["① Gene Map (mtDNA)", "② Variant Density",
                   "③ Hotspot & Disease Loci", "④ Sequence Coverage",
                   "⑤ Mutation Type Distribution"]

    for ax, pname in zip(axes, panel_names):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(pname, fontsize=10, color=ACCENT, loc="left", pad=4)
        ax.tick_params(colors=TEXT_CLR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)

    # ── ① Gene Map ──────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_xlim(0, CHROM_LEN)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.axhline(y=0.5, color="gray", linewidth=0.5, alpha=0.3)

    for gene in gene_map:
        color = GENE_COLORS.get(gene.get("type", "intergenic"), "#999999")
        width = gene["end"] - gene["start"]
        rect = mpatches.FancyBboxPatch(
            (gene["start"], 0.15), width, 0.7,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=0.5,
            alpha=0.85,
        )
        ax1.add_patch(rect)
        if width > 300:
            ax1.text((gene["start"] + gene["end"]) / 2, 0.5,
                     gene.get("gene", ""), fontsize=6, color="white",
                     ha="center", va="center", fontweight="bold")

    # Gene legend
    legend_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in GENE_COLORS.items()]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=7,
               facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
               ncol=5)

    # ── ② Variant Density ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(0, CHROM_LEN)
    if variants:
        positions = [v.get("ref_pos", v.get("aln_pos", 0)) for v in variants]
        bins = np.linspace(0, CHROM_LEN, 200)
        counts, edges = np.histogram(positions, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        # Color by density
        norm_counts = counts / (counts.max() + 1e-9)
        colors = cm.plasma(norm_counts)
        ax2.bar(centers, counts, width=CHROM_LEN / 200,
                color=colors, edgecolor="none", alpha=0.85)
        ax2.set_ylabel("Variants", fontsize=8, color=TEXT_CLR)

        # Highlight transitions vs transversions
        ti_pos = [v.get("ref_pos", 0) for v in variants
                  if v.get("type") == "transition"]
        tv_pos = [v.get("ref_pos", 0) for v in variants
                  if v.get("type") == "transversion"]
        if ti_pos:
            ax2.hist(ti_pos, bins=100, alpha=0.4, color="#4CAF50",
                     label=f"Transitions ({len(ti_pos)})")
        if tv_pos:
            ax2.hist(tv_pos, bins=100, alpha=0.4, color="#FF5722",
                     label=f"Transversions ({len(tv_pos)})")
        ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_CLR,
                   labelcolor=TEXT_CLR)
    ax2.set_xlabel("Position (bp)", fontsize=8, color=TEXT_CLR)

    # ── ③ Hotspots & Disease ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_xlim(0, CHROM_LEN)
    for hs in hotspots:
        pos = hs["center_pos"]
        zscore = hs["z_score"]
        ax3.bar(pos, zscore, width=300, color="red", alpha=0.6,
                edgecolor="white", linewidth=0.5)
        gene_name = hs.get("gene_region", {}).get("gene", "")
        ax3.annotate(f"z={zscore:.1f}\n{gene_name}",
                     xy=(pos, zscore), fontsize=6, color="red",
                     ha="center", va="bottom")

    for dh in disease_hits[:10]:
        pos = dh.get("ref_pos", 0)
        ax3.axvline(x=pos, color="yellow", linewidth=1.5, alpha=0.8,
                    linestyle="--")
        ax3.annotate(f"🧬 {dh.get('disease', '')[:20]}",
                     xy=(pos, ax3.get_ylim()[1] * 0.8),
                     fontsize=6, color="yellow", rotation=30,
                     ha="left")
    ax3.set_ylabel("Z-score", fontsize=8, color=TEXT_CLR)

    # ── ④ Coverage ────────────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.set_xlim(0, CHROM_LEN)
    if variants:
        coverage = np.zeros(CHROM_LEN)
        for v in variants:
            pos = v.get("ref_pos", 0)
            if 0 <= pos < CHROM_LEN:
                lo = max(0, pos - 50)
                hi = min(CHROM_LEN, pos + 50)
                coverage[lo:hi] += 1
        x_cov = np.arange(CHROM_LEN)
        ax4.fill_between(x_cov, coverage, alpha=0.4, color="#00c853")
        ax4.plot(x_cov, coverage, color="#00c853", linewidth=0.5, alpha=0.8)
        ax4.set_ylabel("Coverage", fontsize=8, color=TEXT_CLR)

    # ── ⑤ Mutation Types ─────────────────────────────────────────────────
    ax5 = axes[4]
    mut_types = {}
    for v in variants:
        t = v.get("type", "unknown")
        mut_types[t] = mut_types.get(t, 0) + 1
    if mut_types:
        labels = list(mut_types.keys())
        values = list(mut_types.values())
        bar_colors = ["#2196F3", "#FF5722", "#9C27B0", "#FF9800",
                      "#4CAF50", "#00BCD4"][:len(labels)]
        bars = ax5.barh(labels, values, color=bar_colors, edgecolor="white",
                        linewidth=0.5, alpha=0.85)
        ax5.set_xlabel("Count", fontsize=8, color=TEXT_CLR)
        # Value labels
        for bar, val in zip(bars, values):
            ax5.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     str(val), fontsize=7, color=TEXT_CLR, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Reconstruction comparison (all species before/after)
# ═══════════════════════════════════════════════════════════════════════════════
def create_reconstruction_comparison(
    species_results: Dict[str, Any],
) -> plt.Figure:
    """Heatmap-style before/after for all species."""
    species = list(species_results.keys())
    n = len(species)
    if n == 0:
        return plt.figure()

    fig, axes = plt.subplots(n, 2, figsize=(20, max(6, n * 1.4)),
                             facecolor=DARK_BG)
    if n == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle("Genome Reconstruction — Before vs After",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    base_to_val = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    cmap = LinearSegmentedColormap.from_list(
        "dna", ["#00CC44", "#0066FF", "#FF8800", "#FF2222", "#333333"], N=5)

    for row, sp in enumerate(species):
        res = species_results[sp]
        damaged = res.get("damaged_seq", "")[:400].upper()
        reconstructed = res.get("reconstructed_seq", "")[:400].upper()

        for col, (seq, label) in enumerate(
            [(damaged, "Damaged"), (reconstructed, "Reconstructed")]):
            ax = axes[row, col]
            ax.set_facecolor(PANEL_BG)

            if len(seq) > 0:
                vals = np.array([base_to_val.get(b, 4) for b in seq])
                # Reshape into rows for a heatmap
                cols_per_row = 100
                n_rows = max(1, len(vals) // cols_per_row)
                trim = n_rows * cols_per_row
                mat = vals[:trim].reshape(n_rows, cols_per_row)
                ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4,
                          interpolation="nearest")

            ax.set_title(f"{sp} — {label}", fontsize=8, color=ACCENT, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

            # Stats
            rel = res.get("reliability_score", 0)
            cov = res.get("coverage", 0)
            gaps = seq.count("N") if seq else 0
            ax.text(0.98, 0.05,
                    f"Gaps:{gaps}  Cov:{cov:.0%}  Rel:{rel:.2f}",
                    transform=ax.transAxes, fontsize=6, color="cyan",
                    ha="right", va="bottom",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))

    # Color legend
    patches = [mpatches.Patch(color=c, label=b)
               for b, c in BASE_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, facecolor="#1a1a1a", edgecolor="gray",
               labelcolor="white")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Benchmark radar chart (5 metrics)
# ═══════════════════════════════════════════════════════════════════════════════
def create_benchmark_radar(benchmark_report: dict) -> plt.Figure:
    """Create a radar chart comparing all 5 metrics across species."""
    per_species = benchmark_report.get("per_species", {})
    if not per_species:
        return plt.figure()

    metrics = ["Accuracy", "Similarity", "Phylo\nConsist.", "1-ECE",
               "1-EditDist"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True),
                           facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Benchmark Radar — 5 Metrics per Species",
                 fontsize=14, color="white", fontweight="bold", pad=20)

    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(per_species)))

    for idx, (sp, result) in enumerate(per_species.items()):
        values = [
            result.get("accuracy", {}).get("accuracy", 0),
            result.get("similarity", {}).get("overall_identity", 0),
            result.get("phylo", {}).get("consistency_score", 0),
            1.0 - result.get("calibration", {}).get("ece", 0),
            result.get("edit_dist", {}).get("similarity", 0),
        ]
        values += values[:1]
        color = color_cycle[idx]
        ax.plot(angles, values, "o-", linewidth=1.5, color=color,
                label=sp, markersize=4, alpha=0.8)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics,
                      fontsize=9, color=TEXT_CLR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=7, color="gray")
    ax.yaxis.grid(True, color=GRID_CLR, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_CLR, alpha=0.5)
    ax.spines["polar"].set_color(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, ncol=1)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Confidence calibration plot
# ═══════════════════════════════════════════════════════════════════════════════
def create_calibration_plot(benchmark_report: dict) -> plt.Figure:
    """Calibration reliability diagram for all species."""
    per_species = benchmark_report.get("per_species", {})
    if not per_species:
        return plt.figure()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)
    fig.suptitle("Confidence Calibration Analysis",
                 fontsize=14, color="white", fontweight="bold", y=0.97)

    # Left: Reliability diagram
    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    ax1.set_title("Reliability Diagram", fontsize=11, color=ACCENT)
    ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5,
             label="Perfect calibration")

    color_cycle = plt.cm.Set2(np.linspace(0, 1, len(per_species)))
    for idx, (sp, result) in enumerate(per_species.items()):
        cal = result.get("calibration", {})
        bins_info = cal.get("bins", [])
        if not bins_info:
            continue
        avg_confs = [b["avg_conf"] for b in bins_info if b["count"] > 0]
        avg_accs  = [b["avg_acc"]  for b in bins_info if b["count"] > 0]
        if avg_confs:
            ax1.plot(avg_confs, avg_accs, "o-", color=color_cycle[idx],
                     label=f"{sp} (ECE={cal.get('ece', 0):.3f})",
                     markersize=4, linewidth=1.2, alpha=0.8)

    ax1.set_xlabel("Mean Predicted Confidence", fontsize=9, color=TEXT_CLR)
    ax1.set_ylabel("Fraction of Correct", fontsize=9, color=TEXT_CLR)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=GRID_CLR,
               labelcolor=TEXT_CLR)
    ax1.tick_params(colors=TEXT_CLR, labelsize=7)
    ax1.grid(True, color=GRID_CLR, alpha=0.3)

    # Right: ECE bar chart
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_title("Expected Calibration Error (ECE)", fontsize=11,
                  color=ACCENT)

    species_names = list(per_species.keys())
    ece_vals = [per_species[sp].get("calibration", {}).get("ece", 0)
                for sp in species_names]
    bar_colors = cm.RdYlGn_r(np.array(ece_vals))

    bars = ax2.barh(species_names, ece_vals, color=bar_colors,
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ece_vals):
        ax2.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", fontsize=7, color=TEXT_CLR, va="center")

    ax2.set_xlabel("ECE (lower is better)", fontsize=9, color=TEXT_CLR)
    ax2.tick_params(colors=TEXT_CLR, labelsize=7)
    ax2.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Phylogenetic dendrogram
# ═══════════════════════════════════════════════════════════════════════════════
def create_phylogenetic_tree(
    species_names: List[str],
    embeddings:    np.ndarray,
) -> plt.Figure:
    """Dendrogram from GNN embeddings using scipy."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram

    dist_vec = pdist(embeddings, metric="cosine")
    Z = linkage(dist_vec, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Phylogenetic Tree (GNN Embedding Distances)",
                 fontsize=14, color="white", fontweight="bold")

    dend = dendrogram(Z, labels=species_names, ax=ax,
                      leaf_rotation=45, leaf_font_size=9,
                      color_threshold=0.5 * Z[-1, 2],
                      above_threshold_color=ACCENT)

    ax.set_ylabel("Ward Distance", fontsize=10, color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, alpha=0.3)

    # Color leaf labels
    for label in ax.get_xticklabels():
        label.set_color(TEXT_CLR)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Aggregate benchmark bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def create_benchmark_bars(benchmark_report: dict) -> plt.Figure:
    """Grouped bar chart for all 5 metrics per species."""
    per_species = benchmark_report.get("per_species", {})
    if not per_species:
        return plt.figure()

    species = list(per_species.keys())
    metric_names = ["Accuracy", "Similarity", "Phylo Consist.",
                    "1 - ECE", "1 - Edit Dist"]
    n_metrics = len(metric_names)
    n_species = len(species)

    data = np.zeros((n_species, n_metrics))
    for i, sp in enumerate(species):
        r = per_species[sp]
        data[i] = [
            r.get("accuracy", {}).get("accuracy", 0),
            r.get("similarity", {}).get("overall_identity", 0),
            r.get("phylo", {}).get("consistency_score", 0),
            1.0 - r.get("calibration", {}).get("ece", 0),
            r.get("edit_dist", {}).get("similarity", 0),
        ]

    fig, ax = plt.subplots(figsize=(18, 7), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Benchmark Results — All Species × All Metrics",
                 fontsize=14, color="white", fontweight="bold")

    x = np.arange(n_species)
    width = 0.15
    metric_colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#00BCD4"]

    for j in range(n_metrics):
        offset = (j - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width,
                      label=metric_names[j], color=metric_colors[j],
                      edgecolor="white", linewidth=0.3, alpha=0.85)
        # Value labels
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", fontsize=5, color=TEXT_CLR,
                        ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=30, ha="right",
                       fontsize=8, color=TEXT_CLR)
    ax.set_ylabel("Score", fontsize=10, color=TEXT_CLR)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, ncol=5, loc="upper right")
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.yaxis.grid(True, color=GRID_CLR, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)

    # Aggregate line
    agg = benchmark_report.get("aggregate", {})
    avg_acc = agg.get("avg_accuracy", 0)
    ax.axhline(y=avg_acc, color="yellow", linewidth=1, linestyle="--",
               alpha=0.6, label=f"Avg Accuracy: {avg_acc:.3f}")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Main: generate ALL visualizations
# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_2d_maps(pipeline_results: dict):
    """Generate all PNG visualizations from pipeline results."""
    from config.settings import MTDNA_GENES, MAP_DIR, SIM_DIR

    recon_data = pipeline_results.get("reconstructions", {})

    # ── Per-species chromosome ideograms ──────────────────────────────────
    for species, recon in recon_data.items():
        print(f"  [2D MAP] Generating chromosome map for {species} ...")

        map_path = os.path.join(MAP_DIR, f"{species}_mapping.json")
        variants, hotspots, disease_hits = [], [], []
        if os.path.exists(map_path):
            with open(map_path) as f:
                mdata = json.load(f)
            for m in mdata.get("mappings", []):
                variants.extend(m.get("variants", []))
            hotspots     = mdata.get("hotspots", [])
            disease_hits = mdata.get("disease_hits", [])

        fig = create_chromosome_ideogram(
            gene_map=MTDNA_GENES,
            variants=variants[:2000],
            hotspots=hotspots[:20],
            disease_hits=disease_hits[:20],
            title=f"mtDNA Genome Map — {species}",
        )
        out = os.path.join(VIZ_DIR, f"genome_map_2d_{species}.png")
        fig.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig)
        print(f"    → {out}")

    # ── Reconstruction comparison ─────────────────────────────────────────
    print("  [2D MAP] Generating reconstruction comparison ...")
    spec_data = {}
    for sp, rec in recon_data.items():
        sim_path = os.path.join(SIM_DIR, f"{sp}_simulated.json")
        damaged  = ""
        if os.path.exists(sim_path):
            with open(sim_path) as f:
                sim = json.load(f)
            damaged = sim.get("damaged_sequence", "")[:400]
        # Strip non-DNA from reconstructed seq
        recon_seq = rec.get("reconstructed_seq", "")
        recon_seq = "".join(c for c in recon_seq if c in "ACGTNacgtn")[:400]
        spec_data[sp] = {
            "damaged_seq":       damaged,
            "reconstructed_seq": recon_seq,
            "reliability_score": rec.get("reliability_score", 0),
            "coverage":          rec.get("coverage", 0),
        }
    if spec_data:
        fig2 = create_reconstruction_comparison(spec_data)
        out2 = os.path.join(VIZ_DIR, "reconstruction_comparison.png")
        fig2.savefig(out2, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig2)
        print(f"    → {out2}")

    # ── Benchmark radar ─────────────────────────────────────────────────────
    bench_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench = json.load(f)

        print("  [BENCHMARK] Generating radar chart ...")
        fig3 = create_benchmark_radar(bench)
        out3 = os.path.join(VIZ_DIR, "benchmark_radar.png")
        fig3.savefig(out3, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig3)
        print(f"    → {out3}")

        print("  [BENCHMARK] Generating bar chart ...")
        fig4 = create_benchmark_bars(bench)
        out4 = os.path.join(VIZ_DIR, "benchmark_bars.png")
        fig4.savefig(out4, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig4)
        print(f"    → {out4}")

        print("  [CALIBRATION] Generating calibration plot ...")
        fig5 = create_calibration_plot(bench)
        out5 = os.path.join(VIZ_DIR, "calibration_analysis.png")
        fig5.savefig(out5, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
        plt.close(fig5)
        print(f"    → {out5}")

    # ── Phylogenetic tree ───────────────────────────────────────────────────
    emb = pipeline_results.get("gnn_embeddings")
    if emb:
        print("  [PHYLO] Generating phylogenetic dendrogram ...")
        emb_arr  = np.array(emb)
        sp_names = list(recon_data.keys())
        # Embeddings may cover all species (ancient + modern)
        # Trim to match
        n_use = min(len(sp_names), len(emb_arr))
        if n_use >= 2:
            fig6 = create_phylogenetic_tree(sp_names[:n_use],
                                            emb_arr[:n_use])
            out6 = os.path.join(VIZ_DIR, "phylogenetic_tree.png")
            fig6.savefig(out6, dpi=150, facecolor=DARK_BG,
                         bbox_inches="tight")
            plt.close(fig6)
            print(f"    → {out6}")

    # ── Summary ─────────────────────────────────────────────────────────────
    png_files = sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".png"))
    if png_files:
        print(f"\n  📊 Generated {len(png_files)} visualization images:")
        for pf in png_files:
            print(f"    • {os.path.join(VIZ_DIR, pf)}")


if __name__ == "__main__":
    generate_all_2d_maps({})
