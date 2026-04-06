"""
live_helix_3d.py
================
Real-time interactive 3D DNA double helix visualization using matplotlib.
Features:
  - Continuously rotating 3D helix via FuncAnimation
  - Base-colored nucleotides (A=green, C=blue, G=orange, T=red, N=gray)
  - Real-time damage animation with flash effects on affected bases
  - Hydrogen bond rungs between strands
  - Live stats overlay (mutation count, identity, gaps)
  - Keyboard controls: Space=pause, S=step, +/-=speed, R=reset
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from typing import Optional, List, Dict

# ── Color constants ───────────────────────────────────────────────────────────
BASE_COLORS = {
    "A": "#00CC44",
    "C": "#0066FF",
    "G": "#FF8800",
    "T": "#FF2222",
    "N": "#555555",
}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

DARK_BG    = "#0a0a0a"
PANEL_BG   = "#0d1117"
ACCENT     = "#58a6ff"
FLASH_CLR  = "#FFFF00"   # yellow flash for new damage


class LiveHelix3D:
    """
    Real-time 3D double helix that updates as damage events arrive.
    
    Usage:
        helix = LiveHelix3D(sequence="ACGTACGT...")
        helix.start()  # opens interactive matplotlib window
        
        # From simulation callback:
        helix.apply_damage(position=42, new_base="T")
    """

    def __init__(
        self,
        sequence:       str,
        max_bases:      int   = 200,
        turns_per_base: float = 0.1,
        rise_per_base:  float = 0.34,
        radius:         float = 1.0,
        rotation_speed: float = 0.5,   # degrees per frame
        fig:            Optional[plt.Figure] = None,
        ax:             Optional[plt.Axes]   = None,
        subplot_spec:   object = None,
    ):
        self.sequence       = list(sequence.upper()[:max_bases])
        self.original_seq   = list(sequence.upper()[:max_bases])
        self.n_bases        = len(self.sequence)
        self.max_bases      = max_bases
        self.turns_per_base = turns_per_base
        self.rise_per_base  = rise_per_base
        self.radius         = radius
        self.rotation_speed = rotation_speed
        
        # Animation state
        self.azimuth     = 45.0
        self.elevation   = 20.0
        self.frame_count = 0
        
        # Damage tracking
        self.damaged_positions: List[int] = []
        self.flash_positions:   List[int] = []   # positions to flash this frame
        self.flash_timer:       int       = 0
        
        # Stats
        self.total_damage  = 0
        self.current_phase = "Waiting..."
        
        # Compute helix geometry
        self._compute_coords()
        
        # Figure setup
        self.fig = fig
        self.ax  = ax
        self._owns_figure = fig is None
        self._subplot_spec = subplot_spec
        self._scatter1  = None
        self._scatter2  = None
        self._lines     = []
        self._stats_text = None
    
    def _compute_coords(self):
        """Pre-compute helix backbone coordinates."""
        n = self.n_bases
        t = np.linspace(0, 2 * np.pi * self.turns_per_base * n, n)
        self.z  = np.linspace(0, self.rise_per_base * n, n)
        self.x1 = self.radius * np.cos(t)
        self.y1 = self.radius * np.sin(t)
        self.x2 = -self.x1
        self.y2 = -self.y1
    
    def _base_colors(self, seq: List[str]) -> List[str]:
        """Get color list for a sequence, with flash override."""
        colors = []
        for i, base in enumerate(seq):
            if i in self.flash_positions and self.flash_timer > 0:
                colors.append(FLASH_CLR)
            elif i in self.damaged_positions:
                # Damaged bases get a slightly brighter tint
                base_clr = BASE_COLORS.get(base, "#555555")
                colors.append(base_clr)
            else:
                colors.append(BASE_COLORS.get(base, "#555555"))
        return colors
    
    def _base_sizes(self, seq: List[str]) -> List[float]:
        """Get marker sizes — gaps are smaller, flashing bases are bigger."""
        sizes = []
        for i, base in enumerate(seq):
            if i in self.flash_positions and self.flash_timer > 0:
                sizes.append(120)  # big flash
            elif base == "N":
                sizes.append(12)
            else:
                sizes.append(40)
        return sizes
    
    def setup_axes(self):
        """Initialize the 3D axes and draw initial helix."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 10), facecolor=DARK_BG)
            self.fig.suptitle("🧬  Live 3D DNA Double Helix",
                              fontsize=16, color="white", fontweight="bold",
                              y=0.96)
        
        if self.ax is None:
            if self._subplot_spec is not None:
                self.ax = self.fig.add_subplot(self._subplot_spec,
                                                projection="3d",
                                                facecolor=DARK_BG)
            else:
                self.ax = self.fig.add_subplot(111, projection="3d",
                                                facecolor=DARK_BG)
        
        ax = self.ax
        ax.set_facecolor(DARK_BG)
        ax.set_title("3D DNA Helix — Live Damage", fontsize=12,
                      color=ACCENT, pad=8)
        
        # Style panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#222222")
        ax.yaxis.pane.set_edgecolor("#222222")
        ax.zaxis.pane.set_edgecolor("#222222")
        ax.tick_params(colors="#666666", labelsize=5)
        ax.set_xlabel("", fontsize=0)
        ax.set_ylabel("", fontsize=0)
        ax.set_zlabel("bp", color="#666666", fontsize=7)
        
        # Draw initial state
        self._draw_helix()
        
        # Stats text
        self._stats_text = ax.text2D(
            0.02, 0.02, "", transform=ax.transAxes,
            fontsize=8, color="#aaaaaa", fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(facecolor="#111111", alpha=0.8, edgecolor="#333333",
                      pad=5, boxstyle="round,pad=0.5"),
        )
        
        return ax
    
    def _draw_helix(self):
        """Full redraw of the helix (called on setup and reset)."""
        ax = self.ax
        
        colors1 = self._base_colors(self.sequence)
        comp = [COMPLEMENT.get(b, "N") for b in self.sequence]
        colors2 = self._base_colors(comp)
        sizes1  = self._base_sizes(self.sequence)
        
        # Strand 1
        self._scatter1 = ax.scatter(
            self.x1, self.y1, self.z,
            c=colors1, s=sizes1,
            alpha=0.9, edgecolors="white", linewidths=0.3,
            depthshade=True, zorder=5,
        )
        ax.plot(self.x1, self.y1, self.z,
                color="gray", alpha=0.25, linewidth=0.8, zorder=1)
        
        # Strand 2 (complement)
        self._scatter2 = ax.scatter(
            self.x2, self.y2, self.z,
            c=colors2, s=[28] * self.n_bases,
            alpha=0.6, edgecolors="white", linewidths=0.2,
            depthshade=True, zorder=4,
        )
        ax.plot(self.x2, self.y2, self.z,
                color="gray", alpha=0.15, linewidth=0.8, zorder=0)
        
        # Hydrogen bonds (every 3rd base)
        for i in range(0, self.n_bases, 3):
            clr = BASE_COLORS.get(self.sequence[i], "#555555")
            line, = ax.plot(
                [self.x1[i], self.x2[i]],
                [self.y1[i], self.y2[i]],
                [self.z[i],  self.z[i]],
                color=clr, alpha=0.2, linewidth=0.6, zorder=2,
            )
            self._lines.append(line)
        
        ax.view_init(elev=self.elevation, azim=self.azimuth)
    
    # ── Damage interface ──────────────────────────────────────────────────────
    def apply_damage(self, position: int, new_base: str,
                     damage_type: str = ""):
        """
        Apply a single damage event to the helix.
        Called from the simulation callback.
        """
        if position < 0 or position >= self.n_bases:
            return
        
        self.sequence[position] = new_base.upper()
        if position not in self.damaged_positions:
            self.damaged_positions.append(position)
        
        # Trigger flash
        self.flash_positions = [position]
        self.flash_timer = 8  # frames to flash
        self.total_damage += 1
        self.current_phase = damage_type or "damage"
    
    def apply_gap(self, start: int, end: int):
        """Apply a large gap (missing segment)."""
        for i in range(max(0, start), min(end, self.n_bases)):
            self.sequence[i] = "N"
            if i not in self.damaged_positions:
                self.damaged_positions.append(i)
        
        self.flash_positions = list(range(max(0, start),
                                          min(end, self.n_bases)))
        self.flash_timer = 12
        self.total_damage += 1
        self.current_phase = "missing_segment"
    
    # ── Animation update ──────────────────────────────────────────────────────
    def update_frame(self, frame: int):
        """Called by FuncAnimation each frame — rotates and updates colors."""
        self.frame_count = frame
        
        # Rotate
        self.azimuth = (self.azimuth + self.rotation_speed) % 360
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        
        # Update colors (handles flash decay)
        if self.flash_timer > 0:
            self.flash_timer -= 1
            if self.flash_timer == 0:
                self.flash_positions.clear()
        
        colors1 = self._base_colors(self.sequence)
        comp = [COMPLEMENT.get(b, "N") for b in self.sequence]
        colors2 = self._base_colors(comp)
        sizes1  = self._base_sizes(self.sequence)
        
        # Update scatter colors and sizes
        if self._scatter1 is not None:
            self._scatter1.set_facecolors(colors1)
            self._scatter1.set_sizes(sizes1)
        if self._scatter2 is not None:
            self._scatter2.set_facecolors(colors2)
        
        # Update stats
        if self._stats_text is not None:
            seq_str = "".join(self.sequence)
            n_gaps = seq_str.count("N")
            matches = sum(1 for a, b in zip(self.original_seq, self.sequence)
                          if a == b and a != "N")
            total = sum(1 for c in self.original_seq if c != "N")
            identity = matches / max(total, 1)
            
            self._stats_text.set_text(
                f"Damages: {self.total_damage:,}  │  "
                f"Gaps: {n_gaps:,}  │  "
                f"Identity: {identity:.1%}  │  "
                f"Phase: {self.current_phase}"
            )
        
        return [self._scatter1, self._scatter2, self._stats_text]
    
    # ── Standalone launch ─────────────────────────────────────────────────────
    def start(self, interval: int = 50):
        """
        Start the interactive helix viewer (standalone mode).
        Opens a matplotlib window with continuous rotation.
        """
        self.setup_axes()
        
        # Legend
        patches = [mpatches.Patch(color=c, label=b)
                   for b, c in BASE_COLORS.items()]
        patches.append(mpatches.Patch(color=FLASH_CLR, label="Flash ★"))
        self.fig.legend(handles=patches, loc="lower center", ncol=6,
                        fontsize=9, facecolor="#1a1a1a", edgecolor="#444444",
                        labelcolor="white", framealpha=0.9)
        
        self.anim = FuncAnimation(
            self.fig, self.update_frame,
            interval=interval, blit=False, cache_frame_data=False,
        )
        plt.show()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    seq = "".join(random.choices("ACGT", k=200))
    helix = LiveHelix3D(seq, max_bases=200)
    
    # Simulate some damage before starting
    for i in range(0, 200, 15):
        helix.apply_damage(i, "N", "test_damage")
    
    helix.start()
