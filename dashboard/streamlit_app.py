"""
Streamlit Dashboard — Genome Sequencing & Reconstruction System
================================================================
Full evaluation metrics dashboard with:
  📊 Metrics Overview  📈 Training Curves  🧬 Sequence Viewer
  🌡️ Confidence Heatmap  🌳 Phylogenetic Tree  📊 Calibration Plot
  🔬 Per-Species Breakdown  📋 Benchmark Table  🔄 Live Training Monitor

Launch:
  streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import json
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

try:
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Install required packages: pip install streamlit plotly")
    sys.exit(1)

from config.settings import (
    RESULTS_DIR, MODEL_DIR, VIZ_DIR, BASE_DIR,
    NCBI_SEQUENCES, MODERN_SPECIES, REF_MAP, PHYLO_DISTANCES,
    MTDNA_GENES, DISEASE_MUTATIONS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Page Config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🧬 Genome Reconstruction Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
        color: #c9d1d9;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .metric-card h3 {
        color: #58a6ff;
        margin: 0;
        font-size: 14px;
    }
    .metric-card h1 {
        color: #f0f6fc;
        margin: 5px 0;
        font-size: 28px;
    }
    .species-tag {
        display: inline-block;
        background: #1f6feb33;
        color: #58a6ff;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        margin: 2px;
        border: 1px solid #1f6feb55;
    }
    .gene-protein { color: #3fb950; }
    .gene-rrna { color: #f0883e; }
    .gene-trna { color: #a371f7; }
    .gene-control { color: #58a6ff; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
        color: #8b949e;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb33;
        color: #58a6ff;
        border-color: #1f6feb;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_benchmark_report():
    path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=30)
def load_pipeline_log():
    path = os.path.join(RESULTS_DIR, "pipeline_log.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=30)
def load_training_history(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=30)
def load_reconstructions():
    path = os.path.join(RESULTS_DIR, "reconstructions.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=60)
    st.title("🧬 Genome Dashboard")
    st.markdown("---")

    page = st.radio(
        "📋 Navigation",
        ["📊 Overview", "📈 Training", "🧬 Sequences", "🌡️ Confidence",
         "🌳 Phylogenetics", "📊 Calibration", "🔬 Species Detail",
         "📋 Benchmark", "☁️ Upload & Reconstruct"],
        index=0,
    )

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.markdown(f"_Last refresh: {time.strftime('%H:%M:%S')}_")
        time.sleep(0.1)
        st.rerun()

    st.markdown("---")
    st.caption("Genome Sequencing & Reconstruction v3.0")
    st.caption(f"Results: `{RESULTS_DIR}`")


# ═══════════════════════════════════════════════════════════════════════════════
#  📊 OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #f0f6fc; margin: 0;">🧬 Genome Reconstruction Dashboard</h1>
        <p style="color: #8b949e; margin: 5px 0 0 0;">
            Advanced Ancient DNA Sequencing & Reconstruction — Performance Metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    report = load_benchmark_report()
    pipeline = load_pipeline_log()

    if report and "aggregate" in report:
        agg = report["aggregate"]

        cols = st.columns(5)
        metrics = [
            ("Accuracy", agg.get("avg_accuracy", 0), "🎯"),
            ("Similarity", agg.get("avg_similarity", 0), "📊"),
            ("Phylo Score", agg.get("avg_phylo_consist", 0), "🌳"),
            ("Calibration", agg.get("avg_ece", 0), "🌡️"),
            ("Species", agg.get("n_species", 0), "🧬"),
        ]

        for col, (name, val, icon) in zip(cols, metrics):
            with col:
                if isinstance(val, float):
                    st.metric(f"{icon} {name}", f"{val:.4f}")
                else:
                    st.metric(f"{icon} {name}", str(val))

        st.markdown("---")

        # Per-species summary table
        if "per_species" in report:
            st.subheader("📋 Per-Species Results")
            species_data = []
            for sp_name, sp_data in report["per_species"].items():
                species_data.append({
                    "Species": sp_name,
                    "Accuracy": sp_data["accuracy"]["accuracy"],
                    "Edit Dist": sp_data["edit_dist"]["normalised"],
                    "Similarity": sp_data["similarity"]["overall_identity"],
                    "Phylo Score": sp_data["phylo"]["consistency_score"],
                    "ECE": sp_data["calibration"]["ece"],
                    "Gaps": sp_data["similarity"].get("n_gaps_remaining", 0),
                })

            import pandas as pd
            df = pd.DataFrame(species_data)
            st.dataframe(
                df.style.background_gradient(cmap="RdYlGn", subset=["Accuracy", "Similarity", "Phylo Score"])
                    .background_gradient(cmap="RdYlGn_r", subset=["Edit Dist", "ECE"]),
                use_container_width=True,
            )

        # Pipeline timing
        if pipeline:
            elapsed = pipeline.get("elapsed_sec", 0)
            st.info(f"⏱️ Pipeline completed in **{elapsed:.1f}s** ({elapsed/60:.1f} min)")

    else:
        st.warning("No benchmark results found. Run `python main.py` first to generate results.")
        st.code("python main.py", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
#  📈 TRAINING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training":
    st.header("📈 Training Curves")

    training_models = {
        "DNABERT-2": "dnabert2",
        "Denoising AE": "ae",
        "Phylogenetic GNN": "gnn",
        "Fusion (T+GNN)": "fusion",
        "Evoformer": "evoformer",
    }

    tabs = st.tabs(list(training_models.keys()))

    for tab, (model_name, file_prefix) in zip(tabs, training_models.items()):
        with tab:
            history = load_training_history(file_prefix)

            if history:
                fig = make_subplots(rows=1, cols=1)

                epochs = [h["epoch"] for h in history]
                losses = [h.get("loss", h.get("recon_loss", 0)) for h in history]

                fig.add_trace(go.Scatter(
                    x=epochs, y=losses,
                    mode="lines+markers",
                    name="Training Loss",
                    line=dict(color="#58a6ff", width=2),
                    marker=dict(size=6),
                ))

                if "val_loss" in history[0]:
                    val_losses = [h["val_loss"] for h in history]
                    fig.add_trace(go.Scatter(
                        x=epochs, y=val_losses,
                        mode="lines+markers",
                        name="Validation Loss",
                        line=dict(color="#f0883e", width=2, dash="dash"),
                        marker=dict(size=6),
                    ))

                if "mlm_loss" in history[0]:
                    mlm_losses = [h["mlm_loss"] for h in history]
                    fig.add_trace(go.Scatter(
                        x=epochs, y=mlm_losses,
                        mode="lines",
                        name="MLM Loss",
                        line=dict(color="#3fb950", width=1.5),
                    ))

                if "conf_loss" in history[0]:
                    conf_losses = [h["conf_loss"] for h in history]
                    fig.add_trace(go.Scatter(
                        x=epochs, y=conf_losses,
                        mode="lines",
                        name="Confidence Loss",
                        line=dict(color="#a371f7", width=1.5),
                    ))

                if "evo_loss" in history[0]:
                    evo_losses = [h.get("evo_loss", 0) for h in history]
                    fig.add_trace(go.Scatter(
                        x=epochs, y=evo_losses,
                        mode="lines",
                        name="Evolution Loss",
                        line=dict(color="#f778ba", width=1.5),
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    title=f"{model_name} — Training Progress",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=450,
                    font=dict(color="#c9d1d9"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show final stats
                last = history[-1]
                cols = st.columns(4)
                for i, (k, v) in enumerate(last.items()):
                    if k != "epoch" and isinstance(v, (int, float)):
                        cols[i % 4].metric(k.replace("_", " ").title(), f"{v:.4f}")
            else:
                st.info(f"No training history found for {model_name}. Run training first.")


# ═══════════════════════════════════════════════════════════════════════════════
#  🧬 SEQUENCES PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Sequences":
    st.header("🧬 Sequence Viewer")

    recons = load_reconstructions()

    if recons:
        species_list = list(recons.keys())
        selected = st.selectbox("Select Species", species_list)

        if selected and selected in recons:
            info = recons[selected]
            seq = info.get("reconstructed_seq", "")
            confs = info.get("confidences", [])

            col1, col2, col3 = st.columns(3)
            col1.metric("Sequence Length", f"{len(seq):,} bp")
            col2.metric("Gaps Remaining", info.get("gaps_remaining", "N/A"))
            col3.metric("Reliability", f"{info.get('reliability_score', 0):.4f}")

            st.markdown("---")
            st.subheader("Sequence (first 2000 bp)")

            # Color-coded sequence display
            display_len = min(len(seq), 2000)
            color_map = {
                "A": "#00CC44", "C": "#0066FF",
                "G": "#FF8800", "T": "#FF2222", "N": "#555555",
            }

            html_seq = ""
            for i in range(display_len):
                base = seq[i]
                color = color_map.get(base, "#888")
                html_seq += f'<span style="color:{color}; font-family:monospace; font-size:11px;">{base}</span>'
                if (i + 1) % 80 == 0:
                    html_seq += "<br>"

            st.markdown(
                f'<div style="background:#0d1117; padding:15px; border-radius:8px; '
                f'border:1px solid #30363d; max-height:400px; overflow-y:auto;">'
                f'{html_seq}</div>',
                unsafe_allow_html=True,
            )

            # Base composition
            st.subheader("Base Composition")
            comp = {b: seq.count(b) for b in "ACGTN"}
            fig = go.Figure(data=[go.Pie(
                labels=list(comp.keys()),
                values=list(comp.values()),
                marker_colors=["#00CC44", "#0066FF", "#FF8800", "#FF2222", "#555555"],
                hole=0.4,
            )])
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No reconstructions found. Run the pipeline first.")


# ═══════════════════════════════════════════════════════════════════════════════
#  🌡️ CONFIDENCE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌡️ Confidence":
    st.header("🌡️ Confidence Heatmap")

    recons = load_reconstructions()
    if recons:
        species_list = list(recons.keys())
        selected = st.selectbox("Select Species", species_list)

        if selected and selected in recons:
            confs = recons[selected].get("confidences", [])
            if confs:
                n = min(len(confs), 2000)
                confs_arr = np.array(confs[:n])

                # Heatmap
                rows = n // 100 + (1 if n % 100 else 0)
                heatmap_data = np.full((rows, 100), np.nan)
                for i, c in enumerate(confs_arr):
                    heatmap_data[i // 100, i % 100] = c

                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    colorscale=[
                        [0, "#ff3333"],
                        [0.5, "#ffcc00"],
                        [0.85, "#00ff88"],
                        [1, "#00ff88"],
                    ],
                    zmin=0, zmax=1,
                    colorbar=dict(title="Confidence"),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    title=f"Per-Base Confidence — {selected}",
                    xaxis_title="Position (mod 100)",
                    yaxis_title="Row (×100 bp)",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Distribution
                fig2 = go.Figure(data=[go.Histogram(
                    x=confs_arr,
                    nbinsx=50,
                    marker_color="#58a6ff",
                    opacity=0.8,
                )])
                fig2.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    title="Confidence Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Stats
                cols = st.columns(4)
                cols[0].metric("Mean", f"{np.mean(confs_arr):.4f}")
                cols[1].metric("Median", f"{np.median(confs_arr):.4f}")
                cols[2].metric("Min", f"{np.min(confs_arr):.4f}")
                cols[3].metric("High Conf (>0.85)", f"{(confs_arr > 0.85).sum():,}")
    else:
        st.warning("No data available.")


# ═══════════════════════════════════════════════════════════════════════════════
#  🌳 PHYLOGENETICS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Phylogenetics":
    st.header("🌳 Phylogenetic Relationships")

    report = load_benchmark_report()

    # Build species relationship graph
    species_all = list(NCBI_SEQUENCES.keys())

    # Phylo distances chart
    st.subheader("Evolutionary Distances (Myr)")
    dist_data = []
    for (sp1, sp2), d in PHYLO_DISTANCES.items():
        dist_data.append({"Species 1": sp1, "Species 2": sp2, "Distance (Myr)": d})

    if dist_data:
        import pandas as pd
        df = pd.DataFrame(dist_data)
        st.dataframe(df, use_container_width=True)

    # Phylo consistency plot
    if report and "per_species" in report:
        st.subheader("Phylogenetic Consistency Scores")
        sp_names = []
        consistency = []
        for sp, data in report["per_species"].items():
            sp_names.append(sp)
            consistency.append(data["phylo"]["consistency_score"])

        fig = go.Figure(data=[go.Bar(
            x=sp_names, y=consistency,
            marker_color=["#3fb950" if c > 0.8 else "#ffcc00" if c > 0.5 else "#ff3333"
                          for c in consistency],
        )])
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            title="Phylo Consistency per Species",
            xaxis_title="Species",
            yaxis_title="Consistency Score",
            yaxis_range=[0, 1.05],
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Gene map
    st.subheader("🗺️ mtDNA Gene Map")
    gene_data = []
    for gene in MTDNA_GENES:
        gene_data.append({
            "Gene": gene["gene"],
            "Start": gene["start"],
            "End": gene["end"],
            "Length": gene["end"] - gene["start"],
            "Type": gene["type"],
        })
    import pandas as pd
    df_genes = pd.DataFrame(gene_data)
    st.dataframe(df_genes, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  📊 CALIBRATION PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Calibration":
    st.header("📊 Confidence Calibration")

    report = load_benchmark_report()

    if report and "per_species" in report:
        species_list = list(report["per_species"].keys())
        selected = st.selectbox("Select Species", species_list)

        if selected:
            cal_data = report["per_species"][selected]["calibration"]

            st.metric("Expected Calibration Error (ECE)", f"{cal_data['ece']:.4f}")

            bins = cal_data.get("bins", [])
            if bins:
                bin_labels = [b["bin"] for b in bins]
                avg_confs = [b["avg_conf"] for b in bins]
                avg_accs = [b["avg_acc"] for b in bins]
                counts = [b["count"] for b in bins]

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Reliability diagram
                fig.add_trace(go.Bar(
                    x=bin_labels, y=avg_accs,
                    name="Avg Accuracy",
                    marker_color="#3fb950",
                    opacity=0.7,
                ))
                fig.add_trace(go.Scatter(
                    x=bin_labels, y=avg_confs,
                    name="Avg Confidence",
                    mode="lines+markers",
                    line=dict(color="#58a6ff", width=2),
                ))
                # Perfect calibration line
                fig.add_trace(go.Scatter(
                    x=bin_labels, y=[i/10 + 0.05 for i in range(10)],
                    name="Perfect Calibration",
                    mode="lines",
                    line=dict(color="#8b949e", dash="dash", width=1),
                ))

                fig.add_trace(go.Bar(
                    x=bin_labels, y=counts,
                    name="Bin Count",
                    marker_color="#a371f7",
                    opacity=0.3,
                ), secondary_y=True)

                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    title=f"Reliability Diagram — {selected}",
                    yaxis_title="Accuracy / Confidence",
                    height=500,
                    barmode="overlay",
                )
                fig.update_yaxes(title_text="Bin Count", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                cols = st.columns(3)
                cols[0].metric("Mean Confidence", f"{cal_data['mean_confidence']:.4f}")
                cols[1].metric("Mean Accuracy", f"{cal_data['mean_accuracy']:.4f}")
                cols[2].metric("Samples", f"{cal_data['n_samples']:,}")
    else:
        st.warning("No calibration data available.")


# ═══════════════════════════════════════════════════════════════════════════════
#  🔬 SPECIES DETAIL PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Species Detail":
    st.header("🔬 Species Detail View")

    report = load_benchmark_report()

    if report and "per_species" in report:
        species_list = list(report["per_species"].keys())
        selected = st.selectbox("Select Species", species_list)

        if selected:
            data = report["per_species"][selected]

            # Metrics cards
            cols = st.columns(5)
            cols[0].metric("🎯 Accuracy", f"{data['accuracy']['accuracy']:.4f}")
            cols[1].metric("📏 Edit Distance", f"{data['edit_dist']['normalised']:.4f}")
            cols[2].metric("📊 Similarity", f"{data['similarity']['overall_identity']:.4f}")
            cols[3].metric("🌳 Phylo Score", f"{data['phylo']['consistency_score']:.4f}")
            cols[4].metric("🌡️ ECE", f"{data['calibration']['ece']:.4f}")

            st.markdown("---")

            # Per-base accuracy
            if "per_base_accuracy" in data["similarity"]:
                st.subheader("Per-Base Accuracy")
                pba = data["similarity"]["per_base_accuracy"]
                fig = go.Figure(data=[go.Bar(
                    x=list(pba.keys()),
                    y=[v if v is not None else 0 for v in pba.values()],
                    marker_color=["#00CC44", "#0066FF", "#FF8800", "#FF2222"],
                )])
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0d1117",
                    paper_bgcolor="#0d1117",
                    title="Accuracy by Base Type",
                    yaxis_range=[0, 1.05],
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Window identities
            if "window_identities" in data["similarity"]:
                windows = data["similarity"]["window_identities"]
                if windows:
                    st.subheader("Sliding Window Identity")
                    w_starts = [w["start"] for w in windows]
                    w_ids = [w["identity"] for w in windows]

                    fig = go.Figure(data=[go.Scatter(
                        x=w_starts, y=w_ids,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#58a6ff", width=1.5),
                        fillcolor="rgba(88, 166, 255, 0.15)",
                    )])
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0d1117",
                        paper_bgcolor="#0d1117",
                        title="Identity Along Sequence",
                        xaxis_title="Position",
                        yaxis_title="Identity",
                        yaxis_range=[0, 1.05],
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Phylo details
            st.subheader("Phylogenetic Details")
            phylo = data["phylo"]
            cols = st.columns(3)
            cols[0].metric("Cosine Similarity", f"{phylo['cosine_similarity']:.4f}")
            cols[1].metric("Expected Similarity", f"{phylo['expected_similarity']:.4f}")
            cols[2].metric("Deviation", f"{phylo['deviation']:.4f}")
    else:
        st.warning("No species data available.")


# ═══════════════════════════════════════════════════════════════════════════════
#  📋 BENCHMARK PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Benchmark":
    st.header("📋 Full Benchmark Report")

    report = load_benchmark_report()

    if report:
        st.subheader("Aggregate Metrics")
        agg = report.get("aggregate", {})
        st.json(agg)

        st.subheader("Per-Species Raw Data")
        st.json(report.get("per_species", {}))

        # Radar chart
        if "per_species" in report:
            st.subheader("Metrics Radar Chart")
            species = list(report["per_species"].keys())[:6]

            fig = go.Figure()
            categories = ["Accuracy", "Similarity", "Phylo", "1-ECE", "1-EditDist"]

            for sp in species:
                d = report["per_species"][sp]
                vals = [
                    d["accuracy"]["accuracy"],
                    d["similarity"]["overall_identity"],
                    d["phylo"]["consistency_score"],
                    1 - d["calibration"]["ece"],
                    1 - d["edit_dist"]["normalised"],
                ]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    name=sp,
                    fill="toself",
                    opacity=0.6,
                ))

            fig.update_layout(
                template="plotly_dark",
                polar=dict(
                    bgcolor="#0d1117",
                    radialaxis=dict(visible=True, range=[0, 1]),
                ),
                paper_bgcolor="#0d1117",
                title="Species Performance Comparison",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No benchmark data available.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ☁️ UPLOAD & RECONSTRUCT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "☁️ Upload & Reconstruct":
    st.header("☁️ Upload DNA & Reconstruct")
    st.markdown("""
    Upload a DNA FASTA file or paste a sequence to get:
    - 🧬 **Reconstructed genome** with gap-filling
    - 📊 **Confidence scores** per position
    - 🌳 **Evolutionary comparison** to modern relatives
    """)

    input_method = st.radio("Input Method", ["Paste Sequence", "Upload FASTA"])

    dna_seq = ""

    if input_method == "Paste Sequence":
        dna_seq = st.text_area(
            "Paste DNA sequence (A/C/G/T/N)",
            height=200,
            placeholder="ACGTNNNNACGT...",
        )
    else:
        uploaded = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "fna"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            lines = content.strip().split("\n")
            dna_seq = "".join(l.strip() for l in lines if not l.startswith(">"))

    species_hint = st.selectbox(
        "Species Hint (for evolutionary comparison)",
        ["neanderthal_mtDNA", "mammoth_mtDNA", "cave_bear_mtDNA",
         "thylacine_mtDNA", "passenger_pigeon", "dodo_partial"],
    )

    if st.button("🔬 Reconstruct", type="primary"):
        if dna_seq:
            dna_seq = "".join(c.upper() for c in dna_seq if c.upper() in "ACGTN")

            if len(dna_seq) < 10:
                st.error("Sequence too short (min 10 bp)")
            else:
                with st.spinner("Running reconstruction..."):
                    try:
                        from visualization.reconstruction_engine import (
                            create_reconstruction_engine,
                        )
                        engine = create_reconstruction_engine(
                            species_name=species_hint,
                            damaged_seq=dna_seq,
                            max_bases=min(len(dna_seq), 5000),
                        )

                        # Run reconstruction
                        for event in engine.reconstruct():
                            pass  # consume all events

                        result_seq = engine.get_current_sequence()
                        result_conf = engine.get_confidence_array()
                        stats = engine.stats

                        st.success(f"✅ Reconstruction complete! "
                                   f"Filled {stats['gaps_filled']}/{stats['total_gaps']} gaps")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Gaps Filled", f"{stats['gaps_filled']:,}")
                        col2.metric("Remaining", f"{stats['gaps_remaining']:,}")
                        col3.metric("Confidence", f"{stats['mean_confidence']:.4f}")

                        # Download buttons
                        st.download_button(
                            "📥 Download Reconstructed FASTA",
                            f">{species_hint}_reconstructed\n{result_seq}\n",
                            file_name=f"{species_hint}_reconstructed.fasta",
                            mime="text/plain",
                        )

                        result_json = json.dumps({
                            "species": species_hint,
                            "reconstructed_sequence": result_seq[:10000],
                            "confidence_scores": result_conf[:10000],
                            "stats": stats,
                        }, indent=2)
                        st.download_button(
                            "📥 Download Results JSON",
                            result_json,
                            file_name=f"{species_hint}_results.json",
                            mime="application/json",
                        )

                    except Exception as e:
                        st.error(f"Reconstruction failed: {e}")
                        st.info("Run `python main.py` first to train models.")
        else:
            st.warning("Please provide a DNA sequence.")
