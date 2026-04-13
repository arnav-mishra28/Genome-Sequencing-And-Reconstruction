"""
main.py
=======
Entry point — Advanced Genome Sequencing & Reconstruction System v3.0

JUST RUN:  python main.py
Everything runs automatically — data download, training, reconstruction,
evaluation, visualizations, and live 3D reconstruction viewer.

Subcommands (optional):
  train            — Run training pipeline only
  reconstruct-live — Launch real-time 3D reconstruction viewer only
  evaluate         — Run benchmark evaluation on existing results
  simulate         — Launch live DNA damage simulation viewer
  serve            — Start FastAPI server
  dashboard        — Launch Dash real-time dashboard
"""

import os
import sys
import json
import time
import argparse

# ── Ensure project root is on sys.path ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import (
    RESULTS_DIR, SEQ_DIR, MODEL_DIR, VIZ_DIR, SIM_DIR,
    MODERN_SPECIES, REF_MAP, DEVICE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Banner
# ═══════════════════════════════════════════════════════════════════════════════
BANNER = """
================================================================
     GENOME SEQUENCING & RECONSTRUCTION SYSTEM v3.0

  Powered by: DNABERT-2 + Fusion(T+GNN) + ESM + AlphaFold + GNN
  New:        Confidence Scoring + Multi-Species + Live 3D Viewer
  Data:       NCBI + Ensembl + UCSC Genome Browser
  Training:   5-Phase Curriculum Learning
  Live Sim:   Real-time 3D/2D Damage Visualization
  Live Recon: Real-time 3D DNA Reconstruction (PyOpenGL)
  API:        FastAPI + Swagger
================================================================
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Module verification
# ═══════════════════════════════════════════════════════════════════════════════
def verify_modules():
    """Check that all required source files exist."""
    missing = []
    checks = [
        os.path.join(BASE_DIR, "config", "settings.py"),
        os.path.join(BASE_DIR, "data", "fetch_sequences.py"),
        os.path.join(BASE_DIR, "data", "simulate_ancient_dna.py"),
        os.path.join(BASE_DIR, "data", "dataset_builder.py"),
        os.path.join(BASE_DIR, "preprocessing", "encoding.py"),
        os.path.join(BASE_DIR, "preprocessing", "alignment.py"),
        os.path.join(BASE_DIR, "models", "dnabert2_transformer.py"),
        os.path.join(BASE_DIR, "models", "denoising_autoencoder.py"),
        os.path.join(BASE_DIR, "models", "lstm_predictor.py"),
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
        os.path.join(BASE_DIR, "pipeline", "full_pipeline.py"),
        os.path.join(BASE_DIR, "pipeline", "genome_mapper.py"),
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


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Download Real Data (NCBI + Ensembl + UCSC)
# ═══════════════════════════════════════════════════════════════════════════════
def step_download_data():
    """Download genome sequences from NCBI, Ensembl, and UCSC."""
    print("\n" + "=" * 65)
    print("  STEP 1 / Download Real Genome Sequences")
    print("  Sources: NCBI Entrez + Ensembl REST + UCSC Genome Browser")
    print("=" * 65)

    from data.fetch_sequences import fetch_all
    metadata = fetch_all()

    print(f"\n  Downloaded {len(metadata)} sequences total.")
    for name, info in metadata.items():
        print(f"    {name}: {info['length']:,} bp ({info['source']})")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Train Everything (5-phase curriculum)
# ═══════════════════════════════════════════════════════════════════════════════
def step_train(skip_download=False, **epoch_overrides):
    """Run the full 5-phase training + reconstruction pipeline."""
    from pipeline.full_pipeline import run_pipeline

    results = run_pipeline(
        skip_download=skip_download,
        **epoch_overrides,
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Evaluate & Generate Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
def step_evaluate_and_visualize(results):
    """Generate all visualizations and print final report."""
    print("\n" + "=" * 65)
    print("  STEP 3 / Generating Visualizations")
    print("=" * 65)

    # 3D Helix HTML
    try:
        from visualization.helix_3d import save_helix_html
        import glob
        recon_data = results.get("reconstructions", {})
        sim_dir = os.path.join(BASE_DIR, "data", "simulated")
        helix_input = {}
        for species, rec in recon_data.items():
            sim_path = os.path.join(sim_dir, f"{species}_simulated.json")
            damaged, repair_log = "", []
            if os.path.exists(sim_path):
                with open(sim_path) as f:
                    sim = json.load(f)
                damaged = sim.get("damaged_sequence", "")
                repair_log = rec.get("ae_repairs", [])
            helix_input[species] = {
                "damaged": damaged[:300] if damaged else "ACGTNNN" * 30,
                "reconstructed": rec.get("reconstructed_seq", "ACGT" * 50),
                "repair_log": repair_log,
            }
        save_helix_html(helix_input)
    except Exception as e:
        print(f"  [VIZ] Helix visualization: {e}")

    # 2D Genome Maps
    try:
        from visualization.genome_map_2d import generate_all_2d_maps
        import glob
        aln_dir = os.path.join(BASE_DIR, "data", "alignments")
        alignments = {}
        for f_path in glob.glob(os.path.join(aln_dir, "*.json")):
            name = os.path.basename(f_path).replace("_alignment.json", "")
            with open(f_path) as fh:
                alignments[name] = json.load(fh)
        results["alignments"] = alignments
        generate_all_2d_maps(results)
    except Exception as e:
        print(f"  [VIZ] 2D maps: {e}")

    # Show what was generated
    if os.path.exists(VIZ_DIR):
        png_files = sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".png"))
        if png_files:
            print(f"\n  Generated {len(png_files)} visualization images:")
            for pf in png_files:
                print(f"    {os.path.join(VIZ_DIR, pf)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Launch Live 3D Reconstruction Viewer
# ═══════════════════════════════════════════════════════════════════════════════
def step_live_reconstruction():
    """Launch the real-time 3D DNA reconstruction viewer."""
    print("\n" + "=" * 65)
    print("  STEP 4 / Launching Live 3D DNA Reconstruction Viewer")
    print("=" * 65)

    from visualization.reconstruction_viewer import launch_reconstruction_viewer

    launch_reconstruction_viewer(
        species_name="neanderthal_mtDNA",
        speed=15.0,
        max_bases=300,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL AUTO-RUN (python main.py — no arguments needed)
# ═══════════════════════════════════════════════════════════════════════════════
def run_everything():
    """
    THE ONE-COMMAND PIPELINE.

    Run with just: python main.py

    This executes everything end-to-end:
      1. Download real genome data (NCBI + Ensembl + UCSC)
      2. Simulate ancient DNA damage (deamination, oxidation, gaps)
      3. Encode sequences (k-mer + BPE tokenization)
      4. Train LSTM predictor
      5. Phase 1: Pre-train DNABERT-2 Transformer (MLM)
      6. Phase 2: Corruption training (Denoising Autoencoder)
      7. Phase 3: Evolution-aware training (Phylogenetic GNN)
      8. Phase 4: Fine-tune on ancient DNA
      9. Phase 5: Transformer + GNN Fusion (cross-attention + confidence)
     10. Multi-species ensemble reconstruction (all ancient species)
     11. Full evaluation (5 metrics: accuracy, edit distance,
         similarity, phylo consistency, confidence calibration)
     12. Generate visualizations (3D helix, 2D genome maps)
     13. Launch live 3D reconstruction viewer
    """
    total_start = time.time()

    print("\n" + "=" * 65)
    print("  RUNNING FULL PIPELINE — NO INPUTS NEEDED")
    print("  Everything runs automatically.")
    print("=" * 65)

    # ── Phase A: Download + Train + Reconstruct + Evaluate ────────────────────
    results = step_train(skip_download=False)

    # ── Phase B: Generate Visualizations ──────────────────────────────────────
    step_evaluate_and_visualize(results)

    # ── Final Report ──────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Device:         {DEVICE}")
    print(f"  Results:        {RESULTS_DIR}")
    print(f"  Visualizations: {VIZ_DIR}")
    print(f"  Checkpoints:    {MODEL_DIR}")

    # Show benchmark summary
    report_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        agg = report.get("aggregate", {})
        if agg:
            print(f"\n  -- Benchmark Summary --")
            print(f"  Species evaluated: {agg.get('n_species', 0)}")
            print(f"  Avg Accuracy:      {agg.get('avg_accuracy', 0):.4f}")
            print(f"  Avg Similarity:    {agg.get('avg_similarity', 0):.4f}")
            print(f"  Avg Phylo Consist: {agg.get('avg_phylo_consist', 0):.4f}")
            print(f"  Avg Calibration:   {agg.get('avg_ece', 0):.4f} (ECE)")

    print("\n" + "=" * 65)
    print("  Launching Live 3D Reconstruction Viewer...")
    print("  (Close the viewer window to exit)")
    print("=" * 65)

    # ── Phase C: Launch Live Viewer ───────────────────────────────────────────
    step_live_reconstruction()


# ═══════════════════════════════════════════════════════════════════════════════
#  Subcommand handlers
# ═══════════════════════════════════════════════════════════════════════════════
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

    # Generate visualizations
    if not args.no_viz:
        step_evaluate_and_visualize(results)

    # Optionally launch dashboard
    if not args.no_dashboard:
        try:
            from visualization.realtime_dashboard import run_dashboard
            run_dashboard()
        except Exception:
            pass


def cmd_serve(args):
    """Start FastAPI server."""
    from api.app import run_api
    print("  Starting FastAPI server...")
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
        print("No reconstructions.json found. Run training first.")
        sys.exit(1)

    with open(results_path) as f:
        reconstructions = json.load(f)

    from config.settings import MODERN_SPECIES
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


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI argument parser
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Genome Sequencing & Reconstruction System v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Just run 'python main.py' with no arguments to execute the full pipeline.
Everything runs automatically: download, train, reconstruct, evaluate,
visualize, and launch the live 3D reconstruction viewer.
""",
    )
    sub = parser.add_subparsers(dest="command")

    # -- train --
    p_train = sub.add_parser("train", help="Run full training pipeline")
    p_train.add_argument("--skip-download", action="store_true",
                         help="Skip NCBI/Ensembl download (use cached)")
    p_train.add_argument("--no-viz", action="store_true",
                         help="Skip visualization generation")
    p_train.add_argument("--no-dashboard", action="store_true", default=True,
                         help="Don't auto-launch Dash dashboard")
    p_train.add_argument("--phase1-epochs", type=int, default=None)
    p_train.add_argument("--bert-epochs", type=int, default=None,
                         help="DNABERT-2 pre-training epochs")
    p_train.add_argument("--ae-epochs", type=int, default=None,
                         help="Denoising autoencoder epochs")
    p_train.add_argument("--gnn-epochs", type=int, default=None,
                         help="Phylogenetic GNN epochs")
    p_train.add_argument("--lstm-epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)

    # -- serve --
    sub.add_parser("serve", help="Start FastAPI server")

    # -- dashboard --
    sub.add_parser("dashboard", help="Launch Dash dashboard")

    # -- evaluate --
    sub.add_parser("evaluate", help="Run benchmark evaluation")

    # -- simulate --
    p_sim = sub.add_parser("simulate",
                           help="Launch live DNA damage simulation viewer")
    p_sim.add_argument("--species", type=str, default="neanderthal_mtDNA")
    p_sim.add_argument("--sequence", type=str, default=None)
    p_sim.add_argument("--speed", type=float, default=10.0)
    p_sim.add_argument("--manual", action="store_true")
    p_sim.add_argument("--no-3d", action="store_true")
    p_sim.add_argument("--max-bases", type=int, default=200)
    p_sim.add_argument("--max-events", type=int, default=500)
    p_sim.add_argument("--seed", type=int, default=42)

    # -- reconstruct-live --
    p_rlive = sub.add_parser("reconstruct-live",
                             help="Launch real-time 3D DNA reconstruction viewer")
    p_rlive.add_argument("--species", type=str, default="neanderthal_mtDNA")
    p_rlive.add_argument("--sequence", type=str, default=None)
    p_rlive.add_argument("--max-bases", type=int, default=300)
    p_rlive.add_argument("--speed", type=float, default=15.0)
    p_rlive.add_argument("--width", type=int, default=1600)
    p_rlive.add_argument("--height", type=int, default=900)

    args = parser.parse_args()

    print(BANNER)
    print(f"  Base directory : {BASE_DIR}")
    print(f"  Device         : {DEVICE}")
    print(f"  Python path    : {sys.path[0]}\n")

    # Verify modules
    missing = verify_modules()
    if missing:
        print("Missing source files:\n")
        for m in missing:
            print(f"    {m}")
        print("\nPlease make sure all project files are saved.")
        sys.exit(1)
    else:
        print("  All source files verified.\n")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "train":
        cmd_train(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "reconstruct-live":
        cmd_reconstruct_live(args)
    else:
        # ── DEFAULT: Run everything with just 'python main.py' ────────────
        run_everything()


if __name__ == "__main__":
    main()