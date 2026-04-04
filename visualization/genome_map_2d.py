## 📄 `visualization/genome_map_2d.py`

"""
genome_map_2d.py
================
2D Genome / Chromosome Map using Plotly (interactive HTML).
Shows:
  - Chromosome ideogram with gene bands
  - Variant positions (colored by type)
  - Mutation hotspots
  - Disease-associated loci
  - Before/After reconstruction comparison
  - Phylogenetic dendrogram
  - Fragment coverage track
All panels update based on reconstruction iteration data.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

BASE_DIR  = r"D:\Genome Sequencing And Reconstruction"
VIZ_DIR   = os.path.join(BASE_DIR, "results", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

GENE_COLORS = {
    "protein_coding": "#4CAF50",
    "rRNA":           "#2196F3",
    "tRNA":           "#FF9800",
    "control":        "#9C27B0",
    "intergenic":     "#607D8B",
}


def create_chromosome_ideogram(
    gene_map: List[Dict],
    variants: List[Dict],
    hotspots: List[Dict],
    disease_hits: List[Dict],
    title: str = "mtDNA Map",
) -> go.Figure:
    """Complete chromosome visualization."""
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            "① Gene Map (mtDNA)",
            "② Variant Density",
            "③ Hotspot & Disease Loci",
            "④ Sequence Coverage",
            "⑤ Mutation Type Distribution",
        ],
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
        vertical_spacing=0.08,
    )

    CHROM_LEN = 16569   # human mtDNA length

    # ── ① Gene Map ──────────────────────────────────────────────────────────────
    for gene in gene_map:
        color = GENE_COLORS.get(gene["type"], "#999999")
        fig.add_shape(
            type="rect",
            x0=gene["start"], x1=gene["end"], y0=0.1, y1=0.9,
            fillcolor=color, opacity=0.8,
            line=dict(color="white", width=0.5),
            row=1, col=1,
        )
        mid = (gene["start"] + gene["end"]) / 2
        if gene["end"] - gene["start"] > 200:
            fig.add_annotation(
                x=mid, y=0.5, text=f"<b>{gene['gene']}</b>",
                showarrow=False, font=dict(size=8, color="white"),
                row=1, col=1,
            )

    # Chromosome backbone
    fig.add_shape(type="rect", x0=0, x1=CHROM_LEN, y0=0, y1=1,
                  fillcolor="rgba(0,0,0,0)", line=dict(color="gray", width=1),
                  row=1, col=1)

    # ── ② Variant Density ───────────────────────────────────────────────────────
    if variants:
        positions = [v.get("ref_pos", v.get("aln_pos", 0)) for v in variants]
        # Kernel density estimation
        bins = np.linspace(0, CHROM_LEN, 200)
        density = np.zeros(len(bins)-1)
        for p in positions:
            idx = min(int(p / CHROM_LEN * (len(bins)-1)), len(bins)-2)
            density[idx] += 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        transitions  = [v for v in variants if v.get("type") == "transition"]
        transversions= [v for v in variants if v.get("type") == "transversion"]

        fig.add_trace(go.Bar(
            x=bin_centers, y=density,
            marker_color="rgba(0, 150, 255, 0.7)",
            name="All Variants",
        ), row=2, col=1)

        if transversions:
            tv_pos = [v.get("ref_pos", 0) for v in transversions]
            tv_bins = np.zeros(len(bins)-1)
            for p in tv_pos:
                idx = min(int(p / CHROM_LEN * (len(bins)-1)), len(bins)-2)
                tv_bins[idx] += 1
            fig.add_trace(go.Bar(
                x=bin_centers, y=tv_bins,
                marker_color="rgba(255, 100, 0, 0.7)",
                name="Transversions",
            ), row=2, col=1)

    # ── ③ Hotspots & Disease ────────────────────────────────────────────────────
    for hs in hotspots:
        pos = hs["center_pos"]
        fig.add_shape(
            type="rect",
            x0=pos-100, x1=pos+100, y0=0, y1=hs["z_score"],
            fillcolor="rgba(255,50,50,0.4)", line=dict(color="red", width=1),
            row=3, col=1,
        )
        fig.add_annotation(
            x=pos, y=hs["z_score"] + 0.2,
            text=f"⚠ z={hs['z_score']:.1f}<br>{hs['gene_region'].get('gene','')}",
            showarrow=True, arrowhead=2, font=dict(size=8, color="red"),
            row=3, col=1,
        )

    for dh in disease_hits[:10]:
        pos = dh.get("ref_pos", 0)
        fig.add_shape(
            type="line",
            x0=pos, x1=pos, y0=0, y1=3,
            line=dict(color="yellow", width=2, dash="dot"),
            row=3, col=1,
        )
        fig.add_annotation(
            x=pos, y=3.2,
            text=f"🧬 {dh['disease'][:25]}",
            showarrow=False, font=dict(size=7, color="yellow"),
            textangle=-45,
            row=3, col=1,
        )

    # ── ④ Coverage ──────────────────────────────────────────────────────────────
    if variants:
        coverage = np.zeros(CHROM_LEN)
        for v in variants:
            pos = v.get("ref_pos", 0)
            if 0 <= pos < CHROM_LEN:
                coverage[max(0, pos-50):pos+50] += 1
        x_cov = np.arange(CHROM_LEN)
        fig.add_trace(go.Scatter(
            x=x_cov, y=coverage,
            fill="tozeroy",
            fillcolor="rgba(0,200,100,0.3)",
            line=dict(color="rgba(0,200,100,0.8)", width=1),
            name="Read Coverage",
        ), row=4, col=1)

    # ── ⑤ Mutation Type Pie — as bar ────────────────────────────────────────────
    mut_types = {}
    for v in variants:
        t = v.get("type", "unknown")
        mut_types[t] = mut_types.get(t, 0) + 1
    if mut_types:
        fig.add_trace(go.Bar(
            x=list(mut_types.keys()),
            y=list(mut_types.values()),
            marker_color=["#2196F3", "#FF5722", "#9C27B0", "#FF9800"],
            name="Mutation Types",
        ), row=5, col=1)

    # ── Layout ──────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18, color="white")),
        height=1100,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="monospace"),
        showlegend=True,
        legend=dict(bgcolor="#21262d", bordercolor="#30363d",
                    font=dict(color="white")),
        margin=dict(l=60, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#c9d1d9")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d", zeroline=False, color="#c9d1d9")

    # Legend for gene types
    for gtype, color in GENE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=gtype,
            showlegend=True,
        ), row=1, col=1)

    return fig


def create_reconstruction_comparison(
    species_results: Dict[str, Any],
) -> go.Figure:
    """Side-by-side before/after reconstruction for all species."""
    species = list(species_results.keys())
    n = len(species)

    fig = make_subplots(
        rows=2, cols=n,
        subplot_titles=[f"{s}<br>(Before)" for s in species] +
                       [f"{s}<br>(After)"  for s in species],
        vertical_spacing=0.1,
    )

    for col, sp in enumerate(species, 1):
        res = species_results[sp]
        damaged      = res.get("damaged_seq", "")[:300]
        reconstructed= res.get("reconstructed_seq", "")[:300]

        def seq_to_colored_trace(seq, row):
            base_int = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
            colors_map = {0: "green", 1: "blue", 2: "orange", 3: "red", 4: "gray"}
            y = [base_int.get(b, 4) for b in seq]
            x = list(range(len(seq)))
            c = [colors_map[v] for v in y]
            return go.Bar(
                x=x, y=[1]*len(x),
                marker_color=c,
                name=sp,
                showlegend=(col == 1),
                hovertext=[f"Pos {i}: {b}" for i, b in enumerate(seq)],
                hovertemplate="%{hovertext}<extra></extra>",
            )

        fig.add_trace(seq_to_colored_trace(damaged, row=1),       row=1, col=col)
        fig.add_trace(seq_to_colored_trace(reconstructed, row=2), row=2, col=col)

        # Reliability score annotation
        rel = res.get("reliability_score", 0)
        cov = res.get("coverage", 0)
        fig.add_annotation(
            text=f"Reliability: {rel:.2%}<br>Coverage: {cov:.2%}",
            x=150, y=1.2, xref=f"x{col if col>1 else ''}", yref=f"y{n+col if col>1 else n+1}",
            showarrow=False, font=dict(size=9, color="cyan"),
            bgcolor="rgba(0,0,0,0.5)",
        )

    fig.update_layout(
        title=dict(text="<b>Genome Reconstruction — Before vs After</b>",
                   x=0.5, font=dict(size=16, color="white")),
        height=500,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        showlegend=False,
        bargap=0, bargroupgap=0,
    )
    return fig


def create_phylogenetic_tree(
    species_names: List[str],
    embeddings: np.ndarray,
) -> go.Figure:
    """Visualize phylogenetic relationships using GNN embedding distances."""
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram

    dist_matrix = squareform(pdist(embeddings, metric="cosine"))
    Z = linkage(squareform(dist_matrix), method="ward")
    d = dendrogram(Z, labels=species_names, no_plot=True)

    fig = go.Figure()

    # Draw dendrogram lines
    icoord = np.array(d["icoord"])
    dcoord = np.array(d["dcoord"])
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=dcoord[i], y=icoord[i],
            mode="lines",
            line=dict(color="#58a6ff", width=2),
            showlegend=False,
            hoverinfo="none",
        ))

    # Species labels
    for label, x, y in zip(d["ivl"],
                            [0] * len(d["ivl"]),
                            d["leaves_color_list"] if "leaves_color_list" in d
                            else range(len(d["ivl"]))):
        pass

    fig.update_layout(
        title=dict(text="<b>Phylogenetic Tree (GNN Embedding Distances)</b>",
                   x=0.5, font=dict(size=14, color="white")),
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        xaxis=dict(title="Distance", color="#c9d1d9"),
        yaxis=dict(title="Species", showticklabels=True, color="#c9d1d9"),
        height=500,
    )
    return fig


def generate_all_2d_maps(pipeline_results: dict):
    """Generate all 2D HTML visualizations."""
    from pipeline.genome_mapper import MTDNA_GENES

    recon_data    = pipeline_results.get("reconstructions", {})
    alignment_data= pipeline_results.get("alignments", {})
    mapping_data  = pipeline_results.get("mappings", {})

    # Load full mapping files
    MAP_DIR = os.path.join(BASE_DIR, "data", "mappings")

    for species, recon in recon_data.items():
        print(f"  [2D MAP] Generating chromosome map for {species} ...")

        # Load full mapping
        map_path = os.path.join(MAP_DIR, f"{species}_mapping.json")
        variants, hotspots, disease_hits = [], [], []
        if os.path.exists(map_path):
            with open(map_path) as f:
                mdata = json.load(f)
            for m in mdata.get("mappings", []):
                variants.extend(m.get("variants", []))
            hotspots    = mdata.get("hotspots", [])
            disease_hits= mdata.get("disease_hits", [])

        fig = create_chromosome_ideogram(
            gene_map=MTDNA_GENES,
            variants=variants[:2000],
            hotspots=hotspots[:20],
            disease_hits=disease_hits[:20],
            title=f"mtDNA Genome Map — {species}",
        )
        out = os.path.join(VIZ_DIR, f"genome_map_2d_{species}.html")
        fig.write_html(out, full_html=True, include_plotlyjs="cdn")
        print(f"    → {out}")

    # Reconstruction comparison
    print("  [2D MAP] Generating reconstruction comparison ...")
    # Prepare data dict
    spec_data = {}
    for sp, rec in recon_data.items():
        sim_path = os.path.join(BASE_DIR, "data", "simulated", f"{sp}_simulated.json")
        damaged = ""
        if os.path.exists(sim_path):
            with open(sim_path) as f:
                sim = json.load(f)
            damaged = sim.get("damaged_sequence", "")[:300]
        spec_data[sp] = {
            "damaged_seq":       damaged,
            "reconstructed_seq": rec.get("reconstructed_seq", ""),
            "reliability_score": rec.get("reliability_score", 0),
            "coverage":          rec.get("coverage", 0),
        }

    if spec_data:
        fig2 = create_reconstruction_comparison(spec_data)
        out2 = os.path.join(VIZ_DIR, "reconstruction_comparison.html")
        fig2.write_html(out2, full_html=True, include_plotlyjs="cdn")
        print(f"    → {out2}")

    # Phylogenetic tree
    emb = pipeline_results.get("gnn_embeddings")
    if emb:
        emb_arr = np.array(emb)
        sp_names = list(recon_data.keys())
        if len(sp_names) == len(emb_arr):
            fig3 = create_phylogenetic_tree(sp_names, emb_arr)
            out3 = os.path.join(VIZ_DIR, "phylogenetic_tree.html")
            fig3.write_html(out3, full_html=True, include_plotlyjs="cdn")
            print(f"    → {out3}")

