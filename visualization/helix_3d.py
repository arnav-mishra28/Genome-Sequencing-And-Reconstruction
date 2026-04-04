"""
helix_3d.py
===========
3D Double Helix Reconstruction Simulation using Plotly.
Shows:
  Phase 1 — Damaged helix (gaps, colored by damage type)
  Phase 2 — Sequencing scan animation (beam moving along helix)
  Phase 3 — Step-by-step reconstruction (bases filling in)
  Phase 4 — Final clean helix
Each base is color-coded (A=green, C=blue, G=orange, T=red, N=gray).
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR  = r"D:\Genome Sequencing And Reconstruction"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR   = os.path.join(BASE_DIR, "results", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

BASE_COLORS = {"A": "#00CC44", "C": "#0066FF", "G": "#FF8800", "T": "#FF2222", "N": "#888888"}
BASE_COLORS_LIGHT = {"A": "#88FFaa", "C": "#88aaFF", "G": "#FFcc88", "T": "#FF9988", "N": "#CCCCCC"}


def helix_coords(n_bases: int, turns_per_base: float = 0.1,
                 rise_per_base: float = 0.34, radius: float = 1.0):
    """Generate (x, y, z) for one strand of a helix."""
    t = np.linspace(0, 2 * np.pi * turns_per_base * n_bases, n_bases)
    z = np.linspace(0, rise_per_base * n_bases, n_bases)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return x, y, z


def create_helix_animation(
    sequence: str,
    reconstructed: str,
    repair_log: list,
    species_name: str = "Species",
    max_bases: int = 200,
) -> go.Figure:
    """
    Create a Plotly figure with animation frames showing:
    F0: Damaged helix
    F1–Fn: Progressive reconstruction
    Fn+1: Final clean helix
    """
    seq     = sequence[:max_bases].upper()
    rec_seq = reconstructed[:max_bases].upper()
    n       = len(seq)

    # Coordinates
    x1, y1, z = helix_coords(n)          # Strand 1
    x2, y2, _  = helix_coords(n, turns_per_base=0.1, radius=1.0)
    # Strand 2 is offset by π
    x2 = -x1; y2 = -y1

    def make_strand_trace(seq_str, label, opacity=1.0, phase="damaged"):
        colors = [BASE_COLORS.get(b, "#888888") if b != "N"
                  else "#888888" for b in seq_str]
        sizes  = [6 if b != "N" else 3 for b in seq_str]
        return go.Scatter3d(
            x=x1, y=y1, z=z,
            mode="markers+lines",
            marker=dict(size=sizes, color=colors, opacity=opacity,
                        line=dict(width=0.5, color="white")),
            line=dict(color="rgba(200,200,200,0.4)", width=2),
            name=f"{label} Strand 1",
            text=[f"Pos {i}: {b}" for i, b in enumerate(seq_str)],
            hovertemplate="%{text}<extra></extra>",
        )

    def make_complement_trace(seq_str, opacity=1.0):
        comp = {"A":"T","T":"A","C":"G","G":"C","N":"N"}
        comp_seq = [comp.get(b,"N") for b in seq_str]
        colors = [BASE_COLORS.get(b,"#888888") for b in comp_seq]
        return go.Scatter3d(
            x=x2, y=y2, z=z,
            mode="markers+lines",
            marker=dict(size=5, color=colors, opacity=opacity),
            line=dict(color="rgba(200,200,200,0.3)", width=2),
            name="Complement Strand",
            text=[f"Pos {i}: {b} (complement)" for i,b in enumerate(comp_seq)],
            hovertemplate="%{text}<extra></extra>",
        )

    def make_rungs(seq_str, opacity=0.6):
        """Hydrogen bond rungs connecting base pairs."""
        rung_traces = []
        for i in range(0, n, 5):   # draw every 5th rung for clarity
            b = seq_str[i] if i < len(seq_str) else "N"
            color = BASE_COLORS.get(b, "#888888")
            rung_traces.append(go.Scatter3d(
                x=[x1[i], x2[i]], y=[y1[i], y2[i]], z=[z[i], z[i]],
                mode="lines",
                line=dict(color=color, width=2),
                opacity=opacity,
                showlegend=False,
                hoverinfo="none",
            ))
        return rung_traces

    # ── Build Frames ────────────────────────────────────────────────────────────
    frames = []
    repair_positions = {r["global_pos"] for r in repair_log}

    # Frame 0: fully damaged
    frame0_data = [
        make_strand_trace(seq, "Damaged"),
        make_complement_trace(seq),
    ]
    frame0_data += make_rungs(seq)
    frames.append(go.Frame(data=frame0_data, name="Damaged"))

    # Intermediate frames: progressive reconstruction
    n_steps = min(10, len(repair_log) + 1)
    chunk   = max(1, len(repair_log) // n_steps)
    current_seq = list(seq)

    for step in range(n_steps):
        repairs_so_far = repair_log[:step * chunk]
        for r in repairs_so_far:
            pos = r["global_pos"]
            if pos < len(current_seq):
                current_seq[pos] = r["repaired"]
        cs = "".join(current_seq)
        fd = [
            make_strand_trace(cs, f"Step {step+1}", phase="reconstructing"),
            make_complement_trace(cs, opacity=0.7),
        ]
        fd += make_rungs(cs, opacity=0.5)
        frames.append(go.Frame(data=fd, name=f"Step {step+1}"))

    # Final frame: fully reconstructed
    frame_final = [
        make_strand_trace(rec_seq, "Reconstructed", opacity=1.0),
        make_complement_trace(rec_seq, opacity=0.9),
    ]
    frame_final += make_rungs(rec_seq, opacity=0.8)
    frames.append(go.Frame(data=frame_final, name="Reconstructed"))

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=f"<b>3D DNA Double Helix — {species_name}</b><br>"
                     f"<sub>Green=A | Blue=C | Orange=G | Red=T | Gray=N(missing)</sub>",
                font=dict(size=16, color="white"),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(title="X", showgrid=False, zeroline=False,
                           backgroundcolor="black", color="white"),
                yaxis=dict(title="Y", showgrid=False, zeroline=False,
                           backgroundcolor="black", color="white"),
                zaxis=dict(title="Position (bp)", showgrid=True, zeroline=False,
                           backgroundcolor="#111111", color="white"),
                bgcolor="black",
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.0)),
            ),
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="white"),
            legend=dict(bgcolor="#1a1a1a", bordercolor="gray"),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="▶ Play",  method="animate",
                         args=[None, {"frame": {"duration": 800, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 400}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
                ],
                direction="left", pad={"r": 10, "t": 70},
                showactive=False, x=0.1, xanchor="right", y=0, yanchor="top",
                bgcolor="#333333", bordercolor="#666666", font=dict(color="white"),
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate",
                            args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                             "mode": "immediate"}],
                            label=f.name)
                       for f in frames],
                x=0.1, len=0.9, xanchor="left", y=0,
                currentvalue=dict(prefix="Phase: ", visible=True,
                                  font=dict(color="white", size=12)),
                bgcolor="#333333", bordercolor="#666666", font=dict(color="white"),
            )],
        )
    )
    return fig


def save_helix_html(species_data: dict):
    """
    Generate 3D helix HTML for all species.
    species_data: {name: {"damaged": str, "reconstructed": str, "repair_log": list}}
    """
    for name, data in species_data.items():
        print(f"  [3D HELIX] Generating for {name} ...")
        fig = create_helix_animation(
            sequence=data.get("damaged", "ACGTNNN" * 30),
            reconstructed=data.get("reconstructed", "ACGT" * 50),
            repair_log=data.get("repair_log", []),
            species_name=name,
        )
        out = os.path.join(VIZ_DIR, f"helix_3d_{name}.html")
        fig.write_html(out, full_html=True, include_plotlyjs="cdn")
        print(f"    → Saved: {out}")


if __name__ == "__main__":
    # Quick test with synthetic data
    test_data = {
        "neanderthal_test": {
            "damaged": ("ACGTNNNNACGT" * 15 + "TTCGNNACGT" * 5),
            "reconstructed": "ACGTAACCACGT" * 20,
            "repair_log": [{"global_pos": i*4+3, "original": "N",
                             "repaired": "A", "confidence": 0.9,
                             "action": "gap_fill"} for i in range(20)],
        }
    }
    save_helix_html(test_data)


