"""
main.py
=======
Entry point — Genome Sequencing & Reconstruction
D:\MY WORK\Genome Sequencing And Reconstruction\main.py
"""

import os
import sys
import json
import argparse

# ── Resolve base directory from THIS file's location ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Ensure ALL subdirectories exist ───────────────────────────────────────────
REQUIRED_DIRS = [
    "data",
    os.path.join("data", "sequences"),
    os.path.join("data", "simulated"),
    os.path.join("data", "alignments"),
    os.path.join("data", "mappings"),
    "models",
    os.path.join("models", "checkpoints"),
    "preprocessing",
    "pipeline",
    "visualization",
    "results",
    os.path.join("results", "visualizations"),
]
for d in REQUIRED_DIRS:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# ── Create __init__.py for every package folder ────────────────────────────────
PACKAGES = ["data", "preprocessing", "models", "pipeline", "visualization"]
for pkg in PACKAGES:
    init_path = os.path.join(BASE_DIR, pkg, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# auto-generated\n")

# ── Insert base dir at front of sys.path ──────────────────────────────────────
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Verify critical modules can be found before importing ─────────────────────
def verify_modules():
    missing = []
    checks = [
        os.path.join(BASE_DIR, "pipeline", "full_pipeline.py"),
        os.path.join(BASE_DIR, "data", "fetch_sequences.py"),
        os.path.join(BASE_DIR, "data", "simulate_ancient_dna.py"),
        os.path.join(BASE_DIR, "preprocessing", "encoding.py"),
        os.path.join(BASE_DIR, "preprocessing", "alignment.py"),
        os.path.join(BASE_DIR, "models", "lstm_predictor.py"),
        os.path.join(BASE_DIR, "models", "dnabert_transformer.py"),
        os.path.join(BASE_DIR, "models", "denoising_autoencoder.py"),
        os.path.join(BASE_DIR, "models", "gnn_phylogenetic.py"),
        os.path.join(BASE_DIR, "pipeline", "genome_mapper.py"),
        os.path.join(BASE_DIR, "visualization", "helix_3d.py"),
        os.path.join(BASE_DIR, "visualization", "genome_map_2d.py"),
        os.path.join(BASE_DIR, "visualization", "realtime_dashboard.py"),
    ]
    for path in checks:
        if not os.path.exists(path):
            missing.append(path)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Genome Sequencing & Reconstruction")
    parser.add_argument("--skip-download",  action="store_true",
                        help="Skip NCBI download (use cached sequences)")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Launch dashboard only (pipeline already ran)")
    parser.add_argument("--no-dashboard",   action="store_true",
                        help="Run pipeline + visualizations, skip dashboard")
    parser.add_argument("--lstm-epochs",    type=int, default=5)
    parser.add_argument("--bert-epochs",    type=int, default=4)
    parser.add_argument("--ae-epochs",      type=int, default=6)
    parser.add_argument("--gnn-epochs",     type=int, default=30)
    parser.add_argument("--batch-size",     type=int, default=16)
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     🧬  GENOME SEQUENCING & RECONSTRUCTION SYSTEM  🧬        ║
║         Ancient DNA  →  Modern Reconstruction                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    print(f"  Base directory : {BASE_DIR}")
    print(f"  Python path    : {sys.path[0]}\n")

    # ── Verify all source files exist ─────────────────────────────────────────
    missing = verify_modules()
    if missing:
        print("❌  Missing source files — cannot continue:\n")
        for m in missing:
            print(f"    • {m}")
        print("\nPlease make sure all project files are saved in the correct folders.")
        sys.exit(1)
    else:
        print("✅  All source files found.\n")

    # ── Dashboard-only mode ────────────────────────────────────────────────────
    if args.dashboard_only:
        from visualization.realtime_dashboard import run_dashboard
        run_dashboard()
        return

    # ── Full pipeline ──────────────────────────────────────────────────────────
    from pipeline.full_pipeline import run_pipeline

    results = run_pipeline(
        skip_download=args.skip_download,
        lstm_epochs=args.lstm_epochs,
        bert_epochs=args.bert_epochs,
        ae_epochs=args.ae_epochs,
        gnn_epochs=args.gnn_epochs,
        batch_size=args.batch_size,
    )

    # ── 3D Helix Visualizations ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating 3D Helix Visualizations")
    print("=" * 65)
    from visualization.helix_3d import save_helix_html

    recon_data = results.get("reconstructions", {})
    sim_dir    = os.path.join(BASE_DIR, "data", "simulated")
    helix_input = {}

    for species, rec in recon_data.items():
        sim_path = os.path.join(sim_dir, f"{species}_simulated.json")
        damaged, repair_log = "", []
        if os.path.exists(sim_path):
            with open(sim_path) as f:
                sim = json.load(f)
            damaged    = sim.get("damaged_sequence", "")
            repair_log = rec.get("ae_repairs", [])
        helix_input[species] = {
            "damaged":       damaged[:300] if damaged else "ACGTNNN" * 30,
            "reconstructed": rec.get("reconstructed_seq", "ACGT" * 50),
            "repair_log":    repair_log,
        }

    save_helix_html(helix_input)

    # ── 2D Genome Maps ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Generating 2D Genome Maps")
    print("=" * 65)
    from visualization.genome_map_2d import generate_all_2d_maps
    import glob

    aln_dir    = os.path.join(BASE_DIR, "data", "alignments")
    alignments = {}
    for f in glob.glob(os.path.join(aln_dir, "*.json")):
        name = os.path.basename(f).replace("_alignment.json", "")
        with open(f) as fh:
            alignments[name] = json.load(fh)
    results["alignments"] = alignments

    generate_all_2d_maps(results)

    # ── Summary ────────────────────────────────────────────────────────────────
    viz_dir    = os.path.join(BASE_DIR, "results", "visualizations")
    html_files = sorted(f for f in os.listdir(viz_dir) if f.endswith(".html"))

    print("\n" + "=" * 65)
    print("  📊 Generated Visualizations:")
    for hf in html_files:
        print(f"    • {os.path.join(viz_dir, hf)}")
    print("=" * 65)

    # ── Dashboard ──────────────────────────────────────────────────────────────
    if not args.no_dashboard:
        from visualization.realtime_dashboard import run_dashboard
        print("\n  🌐 Launching Real-Time Dashboard...")
        run_dashboard()

if __name__ == "__main__":
    main()