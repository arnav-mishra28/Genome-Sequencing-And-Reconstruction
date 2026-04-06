"""
live_genome_2d.py
=================
Real-time 2D genome visualization dashboard using matplotlib.
Four-panel live layout:
  1. Chromosome ideogram with damage markers appearing in real-time
  2. Damage density histogram (rolling, updating live)
  3. Mutation type distribution (live bar chart)
  4. Sequence detail strip (zoomed view of ~100 bases around latest damage)

Dark theme, color-coded damage events, live statistics ticker.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from typing import Optional, List, Dict

# ── Color constants ───────────────────────────────────────────────────────────
BASE_COLORS = {
    "A": "#00CC44", "C": "#0066FF", "G": "#FF8800",
    "T": "#FF2222", "N": "#555555",
}

DAMAGE_TYPE_COLORS = {
    "deamination_C_to_T": "#FF4444",
    "deamination_G_to_A": "#FF6644",
    "oxidative_G_to_T":   "#FFAA00",
    "random_substitution": "#AA44FF",
    "deletion":            "#FF00AA",
    "insertion":           "#00AAFF",
    "missing_segment":     "#888888",
}

# Short labels for display
DAMAGE_SHORT_LABELS = {
    "deamination_C_to_T": "C→T",
    "deamination_G_to_A": "G→A",
    "oxidative_G_to_T":   "G→T (ox)",
    "random_substitution": "Random",
    "deletion":            "Del",
    "insertion":           "Ins",
    "missing_segment":     "Gap",
}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_CLR  = "#c9d1d9"
GRID_CLR  = "#21262d"
ACCENT    = "#58a6ff"

# ── Gene map (human mtDNA NC_012920) ──────────────────────────────────────────
GENE_COLORS = {
    "protein_coding": "#4CAF50",
    "rRNA":           "#2196F3",
    "tRNA":           "#FF9800",
    "control":        "#9C27B0",
}

MTDNA_GENES = [
    {"gene": "D-loop",       "start": 1,     "end": 576,   "type": "control"},
    {"gene": "12S rRNA",     "start": 648,   "end": 1601,  "type": "rRNA"},
    {"gene": "16S rRNA",     "start": 1671,  "end": 3229,  "type": "rRNA"},
    {"gene": "ND1",          "start": 3307,  "end": 4262,  "type": "protein_coding"},
    {"gene": "ND2",          "start": 4470,  "end": 5511,  "type": "protein_coding"},
    {"gene": "COX1",         "start": 5904,  "end": 7445,  "type": "protein_coding"},
    {"gene": "COX2",         "start": 7586,  "end": 8269,  "type": "protein_coding"},
    {"gene": "ATP6",         "start": 8527,  "end": 9207,  "type": "protein_coding"},
    {"gene": "COX3",         "start": 9207,  "end": 9990,  "type": "protein_coding"},
    {"gene": "ND4",          "start": 10404, "end": 11935, "type": "protein_coding"},
    {"gene": "ND5",          "start": 11742, "end": 13565, "type": "protein_coding"},
    {"gene": "Cyt-b",        "start": 14747, "end": 15887, "type": "protein_coding"},
    {"gene": "D-loop2",      "start": 15888, "end": 16569, "type": "control"},
]


class LiveGenome2D:
    """
    Real-time 2D genome visualization with four updating panels.
    
    Usage:
        genome2d = LiveGenome2D(sequence="ACGTACGT...", seq_length=16569)
        genome2d.setup_axes()
        
        # From simulation callback:
        genome2d.add_damage_event(event_dict)
    """

    def __init__(
        self,
        sequence:   str,
        seq_length: int           = 16569,
        name:       str           = "specimen",
        fig:        Optional[plt.Figure] = None,
        gs:         Optional[gridspec.GridSpec] = None,
        gs_slots:   Optional[list] = None,
    ):
        self.sequence    = list(sequence.upper())
        self.original    = list(sequence.upper())
        self.seq_length  = seq_length
        self.name        = name
        self.fig         = fig
        self._gs         = gs
        self._gs_slots   = gs_slots
        self._owns_figure = fig is None
        
        # Damage tracking
        self.damage_events:    List[Dict] = []
        self.damage_positions: List[int]  = []
        self.type_counts:      Dict[str, int] = {}
        self.latest_pos:       int = 0
        
        # Density histogram data
        self.n_bins = 100
        self.density = np.zeros(self.n_bins)
        
        # Axes references
        self.ax_ideogram   = None
        self.ax_density    = None
        self.ax_types      = None
        self.ax_detail     = None
        
        # Plot elements for fast updating
        self._density_bars  = None
        self._type_bars     = None
        self._detail_image  = None
        self._damage_dots   = None
        self._stats_text    = None
        self._event_text    = None
    
    def setup_axes(self):
        """Create the four-panel layout."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
            self.fig.suptitle(
                f"🧬  Live 2D Genome Dashboard — {self.name}",
                fontsize=16, color="white", fontweight="bold", y=0.97,
            )
        
        if self._gs_slots is not None:
            # We're embedded in a larger figure — use provided grid slots
            inner = gridspec.GridSpecFromSubplotSpec(
                4, 1, subplot_spec=self._gs_slots,
                height_ratios=[1.0, 1.2, 1.0, 1.2],
                hspace=0.35,
            )
            self.ax_ideogram = self.fig.add_subplot(inner[0])
            self.ax_density  = self.fig.add_subplot(inner[1])
            self.ax_types    = self.fig.add_subplot(inner[2])
            self.ax_detail   = self.fig.add_subplot(inner[3])
        else:
            gs = self.fig.add_gridspec(
                4, 1, hspace=0.4,
                height_ratios=[1.0, 1.2, 1.0, 1.2],
            )
            self.ax_ideogram = self.fig.add_subplot(gs[0])
            self.ax_density  = self.fig.add_subplot(gs[1])
            self.ax_types    = self.fig.add_subplot(gs[2])
            self.ax_detail   = self.fig.add_subplot(gs[3])
        
        for ax, title in [
            (self.ax_ideogram, "① Chromosome Ideogram — Damage Overlay"),
            (self.ax_density,  "② Damage Density (Live Histogram)"),
            (self.ax_types,    "③ Mutation Type Distribution"),
            (self.ax_detail,   "④ Sequence Detail (around latest damage)"),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, fontsize=9, color=ACCENT, loc="left", pad=4)
            ax.tick_params(colors=TEXT_CLR, labelsize=6)
            for spine in ax.spines.values():
                spine.set_color(GRID_CLR)
        
        self._draw_ideogram()
        self._draw_density()
        self._draw_types()
        self._draw_detail()
        
        # Stats text at bottom
        self._stats_text = self.fig.text(
            0.5, 0.01,
            "Waiting for simulation...",
            ha="center", fontsize=9, color="#aaaaaa",
            fontfamily="monospace",
            bbox=dict(facecolor="#111111", alpha=0.8, edgecolor="#333333",
                      pad=5, boxstyle="round,pad=0.5"),
        )
        
        # Event log text
        self._event_text = self.fig.text(
            0.99, 0.01,
            "", ha="right", fontsize=7, color="#888888",
            fontfamily="monospace",
        )
    
    # ── Panel 1: Chromosome ideogram ──────────────────────────────────────────
    def _draw_ideogram(self):
        ax = self.ax_ideogram
        chrom_len = self.seq_length
        ax.set_xlim(0, chrom_len)
        ax.set_ylim(0, 1.5)
        ax.set_yticks([])
        
        # Backbone line
        ax.axhline(y=0.5, color="#333333", linewidth=1, alpha=0.5)
        
        # Gene bands
        for gene in MTDNA_GENES:
            color = GENE_COLORS.get(gene["type"], "#999999")
            width = gene["end"] - gene["start"]
            scale = chrom_len / 16569.0
            start = gene["start"] * scale
            w = width * scale
            rect = mpatches.FancyBboxPatch(
                (start, 0.2), w, 0.6,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor="white", linewidth=0.4,
                alpha=0.7,
            )
            ax.add_patch(rect)
            if w > chrom_len * 0.04:
                ax.text(start + w / 2, 0.5, gene["gene"],
                        fontsize=5, color="white", ha="center", va="center",
                        fontweight="bold")
        
        # Damage markers (empty initially)
        self._damage_scatter = ax.scatter(
            [], [], c=[], s=15, marker="v", alpha=0.8,
            edgecolors="none", zorder=10,
        )
        
        # Gene legend
        patches = [mpatches.Patch(color=c, label=t)
                   for t, c in GENE_COLORS.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=5,
                  facecolor=PANEL_BG, edgecolor=GRID_CLR,
                  labelcolor=TEXT_CLR, ncol=4)
    
    # ── Panel 2: Density histogram ────────────────────────────────────────────
    def _draw_density(self):
        ax = self.ax_density
        ax.set_xlim(0, self.seq_length)
        
        bin_edges = np.linspace(0, self.seq_length, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = self.seq_length / self.n_bins
        
        self._density_bars = ax.bar(
            bin_centers, self.density,
            width=bin_width * 0.9,
            color="#333333", edgecolor="none", alpha=0.85,
        )
        ax.set_ylabel("Count", fontsize=7, color=TEXT_CLR)
        ax.set_xlabel("Position (bp)", fontsize=7, color=TEXT_CLR)
    
    # ── Panel 3: Mutation type bars ───────────────────────────────────────────
    def _draw_types(self):
        ax = self.ax_types
        labels = list(DAMAGE_SHORT_LABELS.values())
        colors = list(DAMAGE_TYPE_COLORS.values())
        zeros  = [0] * len(labels)
        
        self._type_bar_labels = list(DAMAGE_SHORT_LABELS.keys())
        y_pos = np.arange(len(labels))
        
        self._type_bars = ax.barh(
            y_pos, zeros,
            color=colors, edgecolor="white", linewidth=0.3,
            alpha=0.85, height=0.7,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7, color=TEXT_CLR)
        ax.set_xlabel("Count", fontsize=7, color=TEXT_CLR)
        ax.set_xlim(0, 10)
        ax.invert_yaxis()
    
    # ── Panel 4: Sequence detail strip ────────────────────────────────────────
    def _draw_detail(self):
        ax = self.ax_detail
        # Show a heatmap of ~100 bases around a position
        window = 100
        base_to_val = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        cmap = LinearSegmentedColormap.from_list(
            "dna", ["#00CC44", "#0066FF", "#FF8800", "#FF2222", "#333333"],
            N=5,
        )
        
        center = min(len(self.sequence) // 2, self.seq_length // 2)
        start  = max(0, center - window // 2)
        end    = min(len(self.sequence), start + window)
        snippet = self.sequence[start:end]
        
        vals = np.array([base_to_val.get(b, 4) for b in snippet])
        if len(vals) < window:
            vals = np.pad(vals, (0, window - len(vals)), constant_values=4)
        
        # Two rows: top = current, bottom = original
        orig_snippet = self.original[start:end]
        orig_vals = np.array([base_to_val.get(b, 4) for b in orig_snippet])
        if len(orig_vals) < window:
            orig_vals = np.pad(orig_vals, (0, window - len(orig_vals)),
                               constant_values=4)
        
        mat = np.vstack([orig_vals.reshape(1, -1), vals.reshape(1, -1)])
        
        self._detail_image = ax.imshow(
            mat, aspect="auto", cmap=cmap, vmin=0, vmax=4,
            interpolation="nearest", extent=[start, end, -0.5, 1.5],
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Original", "Current"], fontsize=6, color=TEXT_CLR)
        ax.set_xlabel(f"Position (bp) — window around latest damage",
                      fontsize=7, color=TEXT_CLR)
        
        # Store for updates
        self._detail_cmap = cmap
        self._detail_window = window
    
    # ── Damage interface ──────────────────────────────────────────────────────
    def add_damage_event(self, event: Dict):
        """
        Add a damage event (from simulation) and update all panels.
        """
        self.damage_events.append(event)
        pos  = event.get("position", 0)
        etype = event.get("type", "unknown")
        
        self.latest_pos = pos
        
        # Update position tracking
        if etype == "missing_segment":
            end = event.get("end", pos + 50)
            for p in range(pos, min(end, len(self.sequence))):
                self.damage_positions.append(p)
                if p < len(self.sequence):
                    self.sequence[p] = "N"
        elif event.get("applied", True):
            self.damage_positions.append(pos)
            new_base = event.get("mutated", "N")
            if pos < len(self.sequence) and new_base:
                self.sequence[pos] = new_base
        
        # Update type counts
        self.type_counts[etype] = self.type_counts.get(etype, 0) + 1
        
        # Update density
        bin_idx = int(pos / max(self.seq_length, 1) * self.n_bins)
        bin_idx = min(bin_idx, self.n_bins - 1)
        self.density[bin_idx] += 1
    
    def update_panels(self):
        """Refresh all four panels with current data."""
        self._update_ideogram()
        self._update_density()
        self._update_types()
        self._update_detail()
        self._update_stats()
    
    def _update_ideogram(self):
        """Update damage markers on the ideogram."""
        if not self.damage_positions:
            return
        
        ax = self.ax_ideogram
        # Clear old scatter and redraw
        if self._damage_scatter is not None:
            self._damage_scatter.remove()
        
        # Scale positions
        scale = self.seq_length / max(len(self.sequence), 1)
        positions = [p * scale for p in self.damage_positions[-200:]]
        
        # Color by type
        colors = []
        for i in range(max(0, len(self.damage_events) - 200),
                       len(self.damage_events)):
            if i < len(self.damage_events):
                t = self.damage_events[i].get("type", "unknown")
                colors.append(DAMAGE_TYPE_COLORS.get(t, "#FFFFFF"))
        
        # Pad if needed
        while len(colors) < len(positions):
            colors.append("#FF4444")
        colors = colors[:len(positions)]
        
        y_vals = [1.1 + 0.05 * (i % 3) for i in range(len(positions))]
        
        self._damage_scatter = ax.scatter(
            positions, y_vals,
            c=colors, s=12, marker="v", alpha=0.7,
            edgecolors="none", zorder=10,
        )
    
    def _update_density(self):
        """Update density histogram bars."""
        if self._density_bars is None:
            return
        
        max_val = max(self.density.max(), 1)
        cmap = cm.plasma
        
        for i, (bar, val) in enumerate(zip(self._density_bars, self.density)):
            bar.set_height(val)
            bar.set_color(cmap(val / max_val))
        
        self.ax_density.set_ylim(0, max_val * 1.2 + 1)
    
    def _update_types(self):
        """Update mutation type bar chart."""
        if self._type_bars is None:
            return
        
        max_val = 1
        for i, key in enumerate(self._type_bar_labels):
            count = self.type_counts.get(key, 0)
            self._type_bars[i].set_width(count)
            max_val = max(max_val, count)
        
        self.ax_types.set_xlim(0, max_val * 1.3 + 1)
        
        # Add value labels
        # Clear old texts
        for txt in list(self.ax_types.texts):
            txt.remove()
        for i, key in enumerate(self._type_bar_labels):
            count = self.type_counts.get(key, 0)
            if count > 0:
                self.ax_types.text(
                    count + max_val * 0.02, i, str(count),
                    fontsize=6, color=TEXT_CLR, va="center",
                )
    
    def _update_detail(self):
        """Update the sequence detail strip around latest damage."""
        if self._detail_image is None:
            return
        
        window = self._detail_window
        center = self.latest_pos
        start  = max(0, center - window // 2)
        end    = min(len(self.sequence), start + window)
        
        base_to_val = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        
        snippet = self.sequence[start:end]
        vals = np.array([base_to_val.get(b, 4) for b in snippet])
        if len(vals) < window:
            vals = np.pad(vals, (0, window - len(vals)), constant_values=4)
        
        orig_snippet = self.original[start:end]
        orig_vals = np.array([base_to_val.get(b, 4) for b in orig_snippet])
        if len(orig_vals) < window:
            orig_vals = np.pad(orig_vals, (0, window - len(orig_vals)),
                               constant_values=4)
        
        mat = np.vstack([orig_vals.reshape(1, -1), vals.reshape(1, -1)])
        
        self._detail_image.set_data(mat)
        self._detail_image.set_extent([start, end, -0.5, 1.5])
        self.ax_detail.set_xlim(start, end)
        self.ax_detail.set_xlabel(
            f"Position (bp) — bases {start}–{end} (around latest damage)",
            fontsize=7, color=TEXT_CLR,
        )
    
    def _update_stats(self):
        """Update the bottom stats ticker."""
        if self._stats_text is None:
            return
        
        total = len(self.damage_events)
        seq_str = "".join(self.sequence)
        n_gaps = seq_str.count("N")
        
        # Identity
        matches = sum(1 for a, b in zip(self.original, self.sequence)
                      if a == b and a != "N")
        total_valid = sum(1 for c in self.original if c != "N")
        identity = matches / max(total_valid, 1)
        
        top_types = sorted(self.type_counts.items(),
                           key=lambda x: x[1], reverse=True)[:3]
        top_str = "  ".join(f"{DAMAGE_SHORT_LABELS.get(t, t)}:{c}"
                            for t, c in top_types)
        
        self._stats_text.set_text(
            f"Total damages: {total}  │  Gaps: {n_gaps}  │  "
            f"Identity: {identity:.1%}  │  Top: {top_str}"
        )
        
        # Last event text
        if self.damage_events and self._event_text is not None:
            last = self.damage_events[-1]
            self._event_text.set_text(
                f"Last: {last.get('type', '?')} @ pos {last.get('position', '?')}  "
                f"({last.get('original', '?')}→{last.get('mutated', '?')})"
            )
    
    # ── Standalone ────────────────────────────────────────────────────────────
    def start(self, interval: int = 100):
        """Start standalone 2D viewer (for testing)."""
        self.setup_axes()
        plt.show()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    seq = "".join(random.choices("ACGT", k=2000))
    viz = LiveGenome2D(seq, seq_length=len(seq), name="test")
    viz.setup_axes()
    
    # Fake some events
    for i in range(50):
        pos = random.randint(0, 1999)
        types = list(DAMAGE_TYPE_COLORS.keys())
        viz.add_damage_event({
            "type": random.choice(types),
            "position": pos,
            "original": seq[pos],
            "mutated": "N",
            "applied": True,
        })
    
    viz.update_panels()
    plt.show()
