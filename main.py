"""
main.py
=======
Entry point — Advanced Genome Sequencing & Reconstruction System v3.0

Subcommands:
  train           — Run training (all phases or specific phase)
  reconstruct     — Reconstruct from input FASTA file
  reconstruct-live — Launch real-time 3D reconstruction viewer
  evaluate        — Run benchmark evaluation
  serve           — Start FastAPI server
  dashboard       — Launch Dash real-time dashboard
"""

import os
import sys
import json
import argparse

# ── Resolve base directory from THIS file's location ──────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Ensure all subdirectories exist ───────────────────────────────────────────
REQUIRED_DIRS = [
    "config",
    "data",
    os.path.join("data", "sequences"),
    os.path.join("data", "simulated"),
    os.path.join("data", "alignments"),
    os.path.join("data", "mappings"),
    "models",
    os.path.join("models", "checkpoints"),
    "preprocessing",
    "pipeline",
    "training",
    "evaluation",
    "api",
    "visualization",
    "simulation",
    "results",
    os.path.join("results", "visualizations"),
]
for d in REQUIRED_DIRS:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

# ── Create __init__.py for every package folder ────────────────────────────────
PACKAGES = ["config", "data", "preprocessing", "models", "pipeline",
            "visualization", "training", "evaluation", "api", "simulation"]
for pkg in PACKAGES:
    init_path = os.path.join(BASE_DIR, pkg, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# auto-generated\n")

# ── Insert base dir at front of sys.path ──────────────────────────────────────
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║     🧬  GENOME SEQUENCING & RECONSTRUCTION SYSTEM v3.0  🧬     ║
║                                                                  ║
║  Powered by: DNABERT-2 · Fusion(T+GNN) · ESM · AlphaFold · GNN  ║
║  New:        Confidence Scoring · Multi-Species · Live 3D Viewer ║
║  Data:       NCBI · Ensembl · UCSC Genome Browser                ║
║  Training:   5-Phase Curriculum Learning                         ║
║  Live Sim:   Real-time 3D/2D Damage Visualization                ║
║  Live Recon: Real-time 3D DNA Reconstruction (PyOpenGL)          ║
║  API:        FastAPI + Swagger                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def verify_modules():
    """Check that all critical source files exist."""
    missing = []
    checks = [
        os.path.join(BASE_DIR, "config", "settings.py"),
        os.path.join(BASE_DIR, "pipeline", "full_pipeline.py"),
        os.path.join(BASE_DIR, "data", "fetch_sequences.py"),
        os.path.join(BASE_DIR, "data", "simulate_ancient_dna.py"),
        os.path.join(BASE_DIR, "data", "dataset_builder.py"),
        os.path.join(BASE_DIR, "preprocessing", "encoding.py"),
        os.path.join(BASE_DIR, "preprocessing", "alignment.py"),
        os.path.join(BASE_DIR, "preprocessing", "corruption.py"),
        os.path.join(BASE_DIR, "models", "lstm_predictor.py"),
        os.path.join(BASE_DIR, "models", "dnabert2_transformer.py"),
        os.path.join(BASE_DIR, "models", "esm_structure_encoder.py"),
        os.path.join(BASE_DIR, "models", "alphafold_attention.py"),
        os.path.join(BASE_DIR, "models", "denoising_autoencoder.py"),
        os.path.join(BASE_DIR, "models", "gnn_phylogenetic.py"),
        os.path.join(BASE_DIR, "models", "ensemble_reconstructor.py"),
        os.path.join(BASE_DIR, "models", "fusion_model.py"),
        os.path.join(BASE_DIR, "models", "confidence_scorer.py"),
        os.path.join(BASE_DIR, "training", "trainer.py"),
        os.path.join(BASE_DIR, "training", "phase1_pretrain.py"),
        os.path.join(BASE_DIR, "training", "phase2_corruption.py"),
        os.path.join(BASE_DIR, "training", "phase3_evolution_aware.py"),
        os.path.join(BASE_DIR, "training", "phase4_finetune.py"),
        os.path.join(BASE_DIR, "training", "phase5_fusion.py"),
        os.path.join(BASE_DIR, "evaluation", "metrics.py"),
        os.path.join(BASE_DIR, "evaluation", "benchmark.py"),
        os.path.join(BASE_DIR, "api", "app.py"),
        os.path.join(BASE_DIR, "api", "routes.py"),
        os.path.join(BASE_DIR, "api", "schemas.py"),
        os.path.join(BASE_DIR, "pipeline", "genome_mapper.py"),
        os.path.join(BASE_DIR, "visualization", "helix_3d.py"),
        os.path.join(BASE_DIR, "visualization", "genome_map_2d.py"),
        os.path.join(BASE_DIR, "visualization", "live_helix_3d.py"),
        os.path.join(BASE_DIR, "visualization", "live_genome_2d.py"),
        os.path.join(BASE_DIR, "visualization", "live_viewer.py"),
        os.path.join(BASE_DIR, "visualization", "reconstruction_viewer.py"),
        os.path.join(BASE_DIR, "visualization", "reconstruction_engine.py"),
        os.path.join(BASE_DIR, "simulation", "live_simulation.py"),
    ]
    for path in checks:
        if not os.path.exists(path):
            missing.append(path)
    return missing


def cmd_train(args):
    """Run training pipeline."""
    from pipeline.full_pipeline import run_pipeline

    results = run_pipeline(
        skip_download=args.skip_download,
        phase1_epochs=args.phase1_epochs,
        bert_epochs=args.bert_epochs,
        ae_epochs=args.ae_epochs,
        gnn_epochs=args.gnn_epochs,
        lstm_epochs=args.lstm_epochs,
        batch_size=args.batch_size,
    )

    # Generate visualizations if not skipping
    if not args.no_viz:
        _generate_visualizations(results)


def cmd_serve(args):
    """Start FastAPI server."""
    from api.app import run_api
    print("  🌐 Starting FastAPI server …")
    run_api()


def cmd_dashboard(args):
    """Launch Dash dashboard."""
    from visualization.realtime_dashboard import run_dashboard
    run_dashboard()


def cmd_simulate(args):
    """Launch live real-time DNA damage simulation with 3D/2D visualization."""
    from visualization.live_viewer import launch_live_simulation

    launch_live_simulation(
        species_name=args.species,
        sequence=args.sequence,
        speed=args.speed,
        manual=args.manual,
        show_3d=not args.no_3d,
        max_bases=args.max_bases,
        max_events=args.max_events,
        seed=args.seed,
    )


def cmd_reconstruct_live(args):
    """Launch real-time 3D DNA reconstruction viewer."""
    from visualization.reconstruction_viewer import launch_reconstruction_viewer

    launch_reconstruction_viewer(
        species_name=args.species,
        damaged_seq=args.sequence,
        max_bases=args.max_bases,
        speed=args.speed,
        width=args.width,
        height=args.height,
    )



def cmd_evaluate(args):
    """Run benchmark evaluation on existing results."""
    results_path = os.path.join(RESULTS_DIR, "reconstructions.json")
    if not os.path.exists(results_path):
        print("❌ No reconstructions.json found. Run training first.")
        sys.exit(1)

    with open(results_path) as f:
        reconstructions = json.load(f)

    # Load sequences
    from config.settings import SEQ_DIR, MODERN_SPECIES
    from data.fetch_sequences import load_fasta

    meta_path = os.path.join(SEQ_DIR, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    sequences_raw = {}
    for name, info in metadata.items():
        if os.path.exists(info["path"]):
            recs = load_fasta(info["path"])
            sequences_raw[name] = next(iter(recs.values()))[:8000]

    modern_seqs = {k: v for k, v in sequences_raw.items()
                   if k in MODERN_SPECIES}

    from evaluation.benchmark import run_benchmark
    run_benchmark(reconstructions, sequences_raw, {}, modern_seqs)


def _generate_visualizations(results):
    """Generate 3D helix and 2D genome map visualizations."""
    print("\n" + "=" * 65)
    print("  Generating Visualizations")
    print("=" * 65)

    try:
        from visualization.helix_3d import save_helix_html
        import glob

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
    except Exception as e:
        print(f"  [VIZ WARN] Helix visualization: {e}")

    try:
        from visualization.genome_map_2d import generate_all_2d_maps
        import glob

        aln_dir    = os.path.join(BASE_DIR, "data", "alignments")
        alignments = {}
        for f_path in glob.glob(os.path.join(aln_dir, "*.json")):
            name = os.path.basename(f_path).replace("_alignment.json", "")
            with open(f_path) as fh:
                alignments[name] = json.load(fh)
        results["alignments"] = alignments

        generate_all_2d_maps(results)
    except Exception as e:
        print(f"  [VIZ WARN] 2D maps: {e}")

    # Summary
    viz_dir = os.path.join(BASE_DIR, "results", "visualizations")
    if os.path.exists(viz_dir):
        png_files = sorted(f for f in os.listdir(viz_dir)
                            if f.endswith(".png"))
        if png_files:
            print(f"\n  📊 Generated {len(png_files)} Visualization Images:")
            for pf in png_files:
                print(f"    • {os.path.join(viz_dir, pf)}")
            print(f"\n  📂 Open folder: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="🧬 Genome Sequencing & Reconstruction System v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Run 4-phase training pipeline")
    p_train.add_argument("--skip-download", action="store_true",
                         help="Skip NCBI download")
    p_train.add_argument("--no-viz", action="store_true",
                         help="Skip visualization generation")
    p_train.add_argument("--no-dashboard", action="store_true",
                         help="Skip dashboard launch")
    p_train.add_argument("--phase1-epochs", type=int, default=None)
    p_train.add_argument("--bert-epochs", type=int, default=None)
    p_train.add_argument("--ae-epochs", type=int, default=None)
    p_train.add_argument("--gnn-epochs", type=int, default=None)
    p_train.add_argument("--lstm-epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)

    # ── serve ─────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start FastAPI server")

    # ── dashboard ─────────────────────────────────────────────────────────────
    p_dash = sub.add_parser("dashboard", help="Launch Dash dashboard")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Run benchmark evaluation")

    # ── simulate ─────────────────────────────────────────────────────────────
    p_sim = sub.add_parser(
        "simulate",
        help="Launch live real-time DNA damage simulation with 3D/2D visualization",
    )
    p_sim.add_argument(
        "--species", type=str, default="neanderthal_mtDNA",
        help="Species to simulate damage on (default: neanderthal_mtDNA)",
    )
    p_sim.add_argument(
        "--sequence", type=str, default=None,
        help="Custom DNA sequence string (overrides --species loading)",
    )
    p_sim.add_argument(
        "--speed", type=float, default=5.0,
        help="Events per second in auto mode (default: 5.0)",
    )
    p_sim.add_argument(
        "--manual", action="store_true",
        help="Start paused in manual step-through mode",
    )
    p_sim.add_argument(
        "--no-3d", action="store_true",
        help="Skip 3D helix visualization (faster on slower machines)",
    )
    p_sim.add_argument(
        "--max-bases", type=int, default=200,
        help="Max bases to show in 3D helix view (default: 200)",
    )
    p_sim.add_argument(
        "--max-events", type=int, default=500,
        help="Max damage events to simulate (default: 500)",
    )
    p_sim.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # ── reconstruct-live ─────────────────────────────────────────────────────
    p_rlive = sub.add_parser(
        "reconstruct-live",
        help="Launch real-time 3D DNA reconstruction viewer",
    )
    p_rlive.add_argument(
        "--species", type=str, default="neanderthal_mtDNA",
        help="Species to reconstruct (default: neanderthal_mtDNA)",
    )
    p_rlive.add_argument(
        "--sequence", type=str, default=None,
        help="Custom damaged DNA sequence (overrides --species)",
    )
    p_rlive.add_argument(
        "--max-bases", type=int, default=300,
        help="Max bases to show in 3D helix (default: 300)",
    )
    p_rlive.add_argument(
        "--speed", type=float, default=10.0,
        help="Reconstruction events per second (default: 10.0)",
    )
    p_rlive.add_argument(
        "--width", type=int, default=1600,
        help="Window width (default: 1600)",
    )
    p_rlive.add_argument(
        "--height", type=int, default=900,
        help="Window height (default: 900)",
    )

    args = parser.parse_args()

    print(BANNER)
    print(f"  Base directory : {BASE_DIR}")
    print(f"  Python path    : {sys.path[0]}\n")

    # Verify modules
    missing = verify_modules()
    if missing:
        print("❌  Missing source files:\n")
        for m in missing:
            print(f"    • {m}")
        print("\nPlease make sure all project files are saved.")
        sys.exit(1)
    else:
        print("✅  All source files found.\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "train":
        cmd_train(args)
        if not args.no_dashboard:
            try:
                cmd_dashboard(args)
            except Exception:
                pass
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "evaluate":
        from config.settings import RESULTS_DIR
        cmd_evaluate(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "reconstruct-live":
        cmd_reconstruct_live(args)
    else:
        # Default: run full pipeline (backward compatible)
        # Create a namespace with default args
        class DefaultArgs:
            skip_download  = False
            no_viz         = False
            no_dashboard   = True
            phase1_epochs  = None
            bert_epochs    = 4
            ae_epochs      = 6
            gnn_epochs     = 30
            lstm_epochs    = 5
            batch_size     = 16
        cmd_train(DefaultArgs())


if __name__ == "__main__":
    main()