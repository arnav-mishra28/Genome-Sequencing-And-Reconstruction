"""
live_viewer.py
==============
Unified orchestrator that ties the LiveDamageSimulator to the live 3D helix
and 2D genome dashboard visualizations.

Features:
  - Combined matplotlib figure with 3D helix + 2D panels
  - FuncAnimation loop drives both rotation and damage simulation
  - Keyboard controls:
      Space  — pause / resume auto-simulation
      S      — single step (when paused)
      +/=    — speed up
      -      — slow down
      R      — reset simulation
      Q      — quit
  - Auto mode: continuous simulation at configurable speed
  - Manual mode: step-through with keyboard only

All pure Python — matplotlib interactive backend (TkAgg).
"""

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from visualization.live_helix_3d import LiveHelix3D
from visualization.live_genome_2d import LiveGenome2D


# ── Color / style ─────────────────────────────────────────────────────────────
DARK_BG  = "#0a0a0a"
ACCENT   = "#58a6ff"
BTN_CLR  = "#21262d"
BTN_HOVER = "#30363d"


class LiveViewer:
    """
    Unified real-time viewer combining 3D helix + 2D genome dashboard
    with simulation engine.
    
    Usage:
        from simulation.live_simulation import LiveDamageSimulator
        
        sim = LiveDamageSimulator("ACGTACGT...", name="neanderthal")
        viewer = LiveViewer(sim)
        viewer.launch()   # Opens interactive matplotlib window
    """

    def __init__(
        self,
        simulator,
        show_3d:    bool  = True,
        max_bases:  int   = 200,
        auto_mode:  bool  = True,
        speed:      float = 5.0,
    ):
        self.sim       = simulator
        self.show_3d   = show_3d
        self.max_bases = max_bases
        self.auto_mode = auto_mode
        self.speed     = speed
        
        self.sim.speed = speed
        if not auto_mode:
            self.sim.paused = True
        
        # Components
        self.helix_3d:  Optional[LiveHelix3D]  = None
        self.genome_2d: Optional[LiveGenome2D] = None
        
        # Figure
        self.fig  = None
        self.anim = None
        
        # Frame counter
        self._frame       = 0
        self._last_step   = 0
        self._steps_per_frame = 1
        
        # Control state
        self._header_text  = None
        self._control_text = None
        self._phase_text   = None
    
    def _setup_figure(self):
        """Create the combined figure layout."""
        # Use interactive backend
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass  # Fall back to whatever is available
        
        if self.show_3d:
            # Layout: left half = 3D helix, right half = 2D panels
            self.fig = plt.figure(
                figsize=(24, 13),
                facecolor=DARK_BG,
                num="🧬 Live DNA Damage Simulation",
            )
            
            # Main grid: 2 columns
            gs_main = self.fig.add_gridspec(
                1, 2, width_ratios=[1, 1.2],
                wspace=0.08,
                left=0.03, right=0.97, top=0.92, bottom=0.08,
            )
            
            # Left: 3D helix
            self.helix_3d = LiveHelix3D(
                sequence=self.sim.original_seq,
                max_bases=self.max_bases,
                fig=self.fig,
                subplot_spec=gs_main[0],
            )
            self.helix_3d.setup_axes()
            
            # Right: 2D panels
            self.genome_2d = LiveGenome2D(
                sequence=self.sim.original_seq,
                seq_length=len(self.sim.original_seq),
                name=self.sim.name,
                fig=self.fig,
                gs_slots=gs_main[1],
            )
            self.genome_2d.setup_axes()
        else:
            # 2D only
            self.fig = plt.figure(
                figsize=(18, 12),
                facecolor=DARK_BG,
                num="🧬 Live DNA Damage Simulation",
            )
            
            self.genome_2d = LiveGenome2D(
                sequence=self.sim.original_seq,
                seq_length=len(self.sim.original_seq),
                name=self.sim.name,
                fig=self.fig,
            )
            self.genome_2d.setup_axes()
        
        # ── Header ────────────────────────────────────────────────────────────
        self.fig.suptitle(
            f"🧬  LIVE ANCIENT DNA DAMAGE SIMULATION  —  {self.sim.name}",
            fontsize=16, color="white", fontweight="bold", y=0.98,
        )
        
        # ── Controls text ─────────────────────────────────────────────────────
        self._control_text = self.fig.text(
            0.5, 0.04,
            "[SPACE] Pause/Resume  │  [S] Step  │  "
            "[+] Speed Up  │  [-] Slow Down  │  [R] Reset  │  [Q] Quit",
            ha="center", fontsize=8, color="#666666",
            fontfamily="monospace",
        )
        
        # ── Phase indicator ───────────────────────────────────────────────────
        self._phase_text = self.fig.text(
            0.02, 0.98,
            "⏳ Initializing...",
            fontsize=10, color=ACCENT,
            fontfamily="monospace",
            verticalalignment="top",
        )
        
        # ── Speed indicator ───────────────────────────────────────────────────
        self._speed_text = self.fig.text(
            0.98, 0.98,
            f"Speed: {self.speed:.1f} evt/s",
            fontsize=9, color="#888888",
            fontfamily="monospace",
            ha="right", va="top",
        )
        
        # ── Connect keyboard events ───────────────────────────────────────────
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
    
    # ── Keyboard handler ──────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.key == " ":
            self.sim.toggle_pause()
            state = "▶ RUNNING" if not self.sim.paused else "⏸ PAUSED"
            self._phase_text.set_text(state)
            self._phase_text.set_color("#00ff88" if not self.sim.paused
                                       else "#ff8800")
        
        elif event.key == "s":
            # Single step
            if self.sim.paused:
                self._do_simulation_step()
        
        elif event.key in ("+", "="):
            self.sim.speed_up()
            self.speed = self.sim.speed
            self._speed_text.set_text(f"Speed: {self.speed:.1f} evt/s")
        
        elif event.key == "-":
            self.sim.slow_down()
            self.speed = self.sim.speed
            self._speed_text.set_text(f"Speed: {self.speed:.1f} evt/s")
        
        elif event.key == "r":
            self._reset()
        
        elif event.key == "q":
            plt.close(self.fig)
    
    # ── Simulation step ───────────────────────────────────────────────────────
    def _do_simulation_step(self):
        """Execute one simulation step and feed results to visualizations."""
        if self.sim.finished:
            self._phase_text.set_text("✅ SIMULATION COMPLETE")
            self._phase_text.set_color("#00ff88")
            return
        
        event = self.sim.manual_step()
        if event is None:
            return
        
        pos   = event.get("position", 0)
        etype = event.get("type", "")
        
        # Feed to 3D helix
        if self.helix_3d is not None:
            if etype == "missing_segment":
                end = event.get("end", pos + 50)
                self.helix_3d.apply_gap(pos, end)
            else:
                new_base = event.get("mutated", "N")
                if new_base:
                    self.helix_3d.apply_damage(pos, new_base, etype)
        
        # Feed to 2D dashboard
        if self.genome_2d is not None:
            self.genome_2d.add_damage_event(event)
        
        # Update phase text
        phase = event.get("phase", etype)
        step  = event.get("step", 0)
        total = self.sim.stats["total_planned"]
        desc  = event.get("description", "")
        
        self._phase_text.set_text(
            f"▶ Step {step + 1}/{total}  │  {phase}  │  {desc[:60]}"
        )
    
    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        """Reset simulation and visualizations."""
        self.sim.reset()
        
        if self.helix_3d is not None:
            self.helix_3d.sequence = list(self.sim.original_seq[:self.max_bases])
            self.helix_3d.damaged_positions.clear()
            self.helix_3d.flash_positions.clear()
            self.helix_3d.total_damage = 0
        
        if self.genome_2d is not None:
            self.genome_2d.sequence = list(self.sim.original_seq)
            self.genome_2d.damage_events.clear()
            self.genome_2d.damage_positions.clear()
            self.genome_2d.type_counts.clear()
            self.genome_2d.density = np.zeros(self.genome_2d.n_bins)
        
        self._phase_text.set_text("🔄 RESET — Press Space to start")
        self._phase_text.set_color("#ffaa00")
        self.sim.paused = True
    
    # ── Animation frame ───────────────────────────────────────────────────────
    def _update_frame(self, frame):
        """Called by FuncAnimation each frame."""
        self._frame = frame
        
        # Run simulation steps if not paused
        if not self.sim.paused and not self.sim.finished:
            # Calculate how many steps to run this frame
            # Target: self.speed events per second, at ~20 fps animation
            steps_this_frame = max(1, int(self.speed / 20.0))
            for _ in range(steps_this_frame):
                if not self.sim.finished:
                    self._do_simulation_step()
        
        # Update 3D helix rotation and colors
        if self.helix_3d is not None:
            self.helix_3d.update_frame(frame)
        
        # Update 2D panels
        if self.genome_2d is not None:
            self.genome_2d.update_panels()
        
        # Speed text
        if self._speed_text is not None:
            pause_str = " [PAUSED]" if self.sim.paused else ""
            fin_str   = " [DONE]" if self.sim.finished else ""
            self._speed_text.set_text(
                f"Speed: {self.speed:.1f} evt/s{pause_str}{fin_str}"
            )
    
    # ── Launch ────────────────────────────────────────────────────────────────
    def launch(self):
        """
        Open the interactive matplotlib window and start the animation loop.
        This is blocking — it runs until the user closes the window.
        """
        self._setup_figure()
        
        # FuncAnimation drives everything
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=50,      # ~20 fps
            blit=False,
            cache_frame_data=False,
        )
        
        # Maximize the window
        try:
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
        except Exception:
            pass
        
        plt.show()
    
    # ── Save snapshot ─────────────────────────────────────────────────────────
    def save_snapshot(self, path: str = None):
        """Save the current figure state as a PNG."""
        if path is None:
            from config.settings import VIZ_DIR
            path = os.path.join(VIZ_DIR,
                                f"live_snapshot_{self.sim.name}.png")
        
        if self.fig is not None:
            self.fig.savefig(path, dpi=150, facecolor=DARK_BG,
                             bbox_inches="tight")
            print(f"  📸 Snapshot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience launcher functions
# ═══════════════════════════════════════════════════════════════════════════════

def launch_live_simulation(
    species_name: str      = "neanderthal_mtDNA",
    sequence:     str      = None,
    speed:        float    = 5.0,
    manual:       bool     = False,
    show_3d:      bool     = True,
    max_bases:    int      = 200,
    max_events:   int      = 500,
    seed:         int      = 42,
):
    """
    Main entry point for the live simulation viewer.
    
    Args:
        species_name: Name of species to simulate (loads from disk)
        sequence:     Custom sequence string (overrides species loading)
        speed:        Events per second in auto mode
        manual:       If True, start paused in manual step mode
        show_3d:      If True, show 3D helix (disable for slow machines)
        max_bases:    Max bases to show in 3D view
        max_events:   Max damage events to simulate
        seed:         Random seed
    """
    from simulation.live_simulation import (
        LiveDamageSimulator,
        create_simulator_from_species,
    )
    
    print("\n" + "=" * 65)
    print("  🧬  LIVE ANCIENT DNA DAMAGE SIMULATION")
    print("=" * 65)
    
    if sequence:
        sim = LiveDamageSimulator(
            sequence=sequence,
            name=species_name,
            seed=seed,
            max_events=max_events,
        )
        print(f"  Sequence: custom ({len(sequence)} bp)")
    else:
        sim = create_simulator_from_species(
            species_name=species_name,
            max_bases=max(max_bases, 2000),
            max_events=max_events,
            seed=seed,
        )
    
    print(f"  Species:    {species_name}")
    print(f"  Seq length: {len(sim.original_seq)} bp")
    print(f"  3D bases:   {min(max_bases, len(sim.original_seq))}")
    print(f"  Events:     {len(sim._damage_queue)} planned")
    print(f"  Speed:      {speed} events/sec")
    print(f"  Mode:       {'Manual (step-through)' if manual else 'Auto'}")
    print(f"  3D helix:   {'Yes' if show_3d else 'No'}")
    print()
    print("  Controls:")
    print("    [Space]  Pause / Resume")
    print("    [S]      Single step (when paused)")
    print("    [+/=]    Speed up")
    print("    [-]      Slow down")
    print("    [R]      Reset simulation")
    print("    [Q]      Quit")
    print("=" * 65)
    
    viewer = LiveViewer(
        simulator=sim,
        show_3d=show_3d,
        max_bases=max_bases,
        auto_mode=not manual,
        speed=speed,
    )
    viewer.launch()
    
    # After window closes, print summary
    stats = sim.stats
    print("\n" + "=" * 65)
    print("  📊  SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  Total steps:  {stats['total_applied']}")
    print(f"  Gaps:         {stats['gaps']}")
    print(f"  Identity:     {stats['identity']:.2%}")
    print(f"  Type breakdown:")
    for t, c in sorted(stats['type_counts'].items(),
                        key=lambda x: x[1], reverse=True):
        print(f"    • {t}: {c}")
    print("=" * 65)
    
    return sim


# ── Standalone ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    launch_live_simulation(
        species_name="neanderthal_mtDNA",
        speed=10.0,
        manual=False,
        show_3d=True,
        max_bases=200,
        max_events=200,
    )
