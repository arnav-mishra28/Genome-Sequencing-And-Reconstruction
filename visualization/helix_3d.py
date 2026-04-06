"""
helix_3d.py
===========
3D Double Helix Reconstruction Visualization using pure matplotlib.
Generates high-resolution PNG images showing:
  - Damaged DNA helix with gaps and damage coloring
  - Reconstructed helix with repair annotations
  - Side-by-side before/after comparison
Each base is color-coded (A=green, C=blue, G=orange, T=red, N=gray).
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import VIZ_DIR

BASE_COLORS = {"A": "#00CC44", "C": "#0066FF", "G": "#FF8800",
               "T": "#FF2222", "N": "#555555"}
COMPLEMENT  = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def helix_coords(n_bases, turns_per_base=0.1, rise_per_base=0.34,
                 radius=1.0):
    """Generate (x, y, z) for one strand of a helix."""
    t = np.linspace(0, 2 * np.pi * turns_per_base * n_bases, n_bases)
    z = np.linspace(0, rise_per_base * n_bases, n_bases)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x, y, z


def _base_color_list(seq):
    return [BASE_COLORS.get(b, "#555555") for b in seq.upper()]


def create_helix_figure(
    damaged_seq:       str,
    reconstructed_seq: str,
    repair_log:        list,
    species_name:      str = "Species",
    max_bases:         int = 200,
):
    """
    Create a matplotlib figure with two 3D subplots:
    Left  = damaged helix
    Right = reconstructed helix
    """
    dam = damaged_seq[:max_bases].upper()
    rec = reconstructed_seq[:max_bases].upper()
    n   = min(len(dam), len(rec))
    dam, rec = dam[:n], rec[:n]

    x1, y1, z = helix_coords(n)
    x2, y2, _ = -x1, -y1, z   # complementary strand

    fig = plt.figure(figsize=(20, 10), facecolor="#0a0a0a")
    fig.suptitle(f"🧬  3D DNA Double Helix — {species_name}",
                 fontsize=18, color="white", fontweight="bold", y=0.96)

    for idx, (seq, label) in enumerate([(dam, "Damaged"), (rec, "Reconstructed")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d",
                             facecolor="#0a0a0a")
        ax.set_title(label, fontsize=14, color="cyan", pad=10)

        colors = _base_color_list(seq)
        comp_colors = _base_color_list(
            "".join(COMPLEMENT.get(b, "N") for b in seq)
        )
        sizes = [40 if b != "N" else 12 for b in seq]

        # Strand 1
        ax.scatter(x1, y1, z, c=colors, s=sizes,
                   alpha=0.9, edgecolors="white", linewidths=0.3,
                   depthshade=True, zorder=5)
        ax.plot(x1, y1, z, color="gray", alpha=0.3, linewidth=1, zorder=1)

        # Strand 2 (complement)
        ax.scatter(x2, y2, z, c=comp_colors, s=[30] * n,
                   alpha=0.6, edgecolors="white", linewidths=0.2,
                   depthshade=True, zorder=4)
        ax.plot(x2, y2, z, color="gray", alpha=0.2, linewidth=1, zorder=0)

        # Hydrogen bond rungs (every 4th base)
        for i in range(0, n, 4):
            color = BASE_COLORS.get(seq[i], "#555555")
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z[i], z[i]],
                    color=color, alpha=0.3, linewidth=0.8, zorder=2)

        # Mark repair positions
        if label == "Reconstructed":
            repair_pos = {r["global_pos"] for r in repair_log
                          if r["global_pos"] < n}
            rp_list = sorted(repair_pos)[:50]
            if rp_list:
                rx = x1[rp_list]
                ry = y1[rp_list]
                rz = z[rp_list]
                ax.scatter(rx, ry, rz, c="yellow", s=80,
                           marker="*", alpha=0.8, zorder=10,
                           label="Repaired bases")

        # Style
        ax.set_xlabel("X", color="gray", fontsize=8)
        ax.set_ylabel("Y", color="gray", fontsize=8)
        ax.set_zlabel("Position (bp)", color="gray", fontsize=8)
        ax.tick_params(colors="gray", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333333")
        ax.yaxis.pane.set_edgecolor("#333333")
        ax.zaxis.pane.set_edgecolor("#333333")
        ax.view_init(elev=20, azim=45)

    # Legend
    patches = [mpatches.Patch(color=c, label=b)
               for b, c in BASE_COLORS.items()]
    patches.append(mpatches.Patch(color="yellow", label="Repaired ★"))
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=10, facecolor="#1a1a1a", edgecolor="gray",
               labelcolor="white", framealpha=0.9)

    # Stats annotation
    n_gaps_before = dam.count("N")
    n_gaps_after  = rec.count("N")
    stats_text = (f"Gaps: {n_gaps_before} → {n_gaps_after}  |  "
                  f"Repairs: {len(repair_log)}  |  "
                  f"Bases shown: {n}")
    fig.text(0.5, 0.02, stats_text, ha="center",
             fontsize=10, color="#aaaaaa",
             fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    return fig


def save_helix_html(species_data: dict):
    """
    Generate 3D helix PNG images for all species.
    species_data: {name: {"damaged": str, "reconstructed": str,
                          "repair_log": list}}
    """
    for name, data in species_data.items():
        print(f"  [3D HELIX] Generating for {name} ...")
        fig = create_helix_figure(
            damaged_seq=data.get("damaged", "ACGTNNN" * 30),
            reconstructed_seq=data.get("reconstructed", "ACGT" * 50),
            repair_log=data.get("repair_log", []),
            species_name=name,
        )
        out = os.path.join(VIZ_DIR, f"helix_3d_{name}.png")
        fig.savefig(out, dpi=150, facecolor="#0a0a0a",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"    → Saved: {out}")


if __name__ == "__main__":
    test_data = {
        "neanderthal_test": {
            "damaged": ("ACGTNNNNACGT" * 15 + "TTCGNNACGT" * 5),
            "reconstructed": "ACGTAACCACGT" * 20,
            "repair_log": [{"global_pos": i * 4 + 3, "original": "N",
                            "repaired": "A", "confidence": 0.9,
                            "action": "gap_fill"} for i in range(20)],
        }
    }
    save_helix_html(test_data)
    print("Done.")
