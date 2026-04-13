"""
full_pipeline.py
================
Orchestrates the entire genome reconstruction pipeline with:
  - 5-phase training curriculum
  - Multi-species ensemble reconstruction
  - Full evaluation metrics
  - Confidence scoring
"""

import os
import sys
import json
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    BASE_DIR, RESULTS_DIR, MODERN_SPECIES, REF_MAP,
    PHASE1_EPOCHS, PHASE2_EPOCHS, PHASE3_EPOCHS, PHASE4_EPOCHS,
    PHASE5_EPOCHS, LSTM_EPOCHS, AE_EPOCHS, GNN_EPOCHS, BATCH_SIZE,
)


def run_pipeline(
    max_seq_length: int  = 8000,
    phase1_epochs:  int  = None,
    phase2_epochs:  int  = None,
    phase3_epochs:  int  = None,
    phase4_epochs:  int  = None,
    phase5_epochs:  int  = None,
    lstm_epochs:    int  = None,
    ae_epochs:      int  = None,
    gnn_epochs:     int  = None,
    bert_epochs:    int  = None,
    batch_size:     int  = None,
    skip_download:  bool = False,
):
    """Execute the full 5-phase training + reconstruction pipeline."""
    phase1_epochs = phase1_epochs or (bert_epochs or PHASE1_EPOCHS)
    phase2_epochs = phase2_epochs or PHASE2_EPOCHS
    phase3_epochs = phase3_epochs or PHASE3_EPOCHS
    phase4_epochs = phase4_epochs or PHASE4_EPOCHS
    phase5_epochs = phase5_epochs or PHASE5_EPOCHS
    lstm_epochs   = lstm_epochs   or LSTM_EPOCHS
    ae_epochs     = ae_epochs     or AE_EPOCHS
    gnn_epochs    = gnn_epochs    or GNN_EPOCHS
    batch_size    = batch_size    or BATCH_SIZE

    pipeline_log = {"steps": [], "start_time": time.time()}

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Fetch sequences
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 1 — Fetching genome sequences (NCBI / Ensembl / UCSC)")
    print("=" * 65)
    from data.fetch_sequences import fetch_all, load_fasta
    from config.settings import SEQ_DIR

    if not skip_download:
        metadata = fetch_all()
    else:
        meta_path = os.path.join(SEQ_DIR, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)

    pipeline_log["steps"].append({
        "step": 1, "name": "fetch",
        "species": list(metadata.keys()),
    })

    # Load sequences into memory
    sequences_raw = {}
    for name, info in metadata.items():
        if not os.path.exists(info["path"]):
            continue
        recs = load_fasta(info["path"])
        seq  = next(iter(recs.values()))[:max_seq_length]
        sequences_raw[name] = seq
    print(f"\n  Loaded {len(sequences_raw)} sequences.")

    ANCIENT     = {k: v for k, v in sequences_raw.items()
                   if k not in MODERN_SPECIES}
    MODERN_SEQS = {k: v for k, v in sequences_raw.items()
                   if k in MODERN_SPECIES}

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Simulate ancient damage
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 2 — Simulating ancient DNA damage")
    print("=" * 65)
    from data.simulate_ancient_dna import simulate_ancient_damage

    simulated = {}
    for name, seq in ANCIENT.items():
        result = simulate_ancient_damage(seq, name,
                                          seed=hash(name) % (2**31))
        simulated[name] = result
        s = result["mutation_summary"]
        print(f"  {name}: {sum(s.values())} mutations — {s}")
    pipeline_log["steps"].append({
        "step": 2, "name": "simulate", "count": len(simulated),
    })

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: Alignment
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 3 — Aligning ancient sequences to modern references")
    print("=" * 65)
    from preprocessing.alignment import align_to_reference, find_outlier_variants

    alignments = {}
    for name, sim_result in simulated.items():
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref_seq  = MODERN_SEQS.get(ref_name, "ACGT" * 100)
        aln = align_to_reference(
            query=sim_result["damaged_sequence"],
            reference=ref_seq, name=name,
        )
        outliers = find_outlier_variants(aln["variants"])
        aln["outlier_variants"] = outliers
        alignments[name] = aln
        print(f"  {name}: identity={aln['identity']:.3f}, "
              f"variants={aln['n_variants']}, outliers={len(outliers)}")
    pipeline_log["steps"].append({"step": 3, "name": "align"})

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: Encode sequences
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 4 — Encoding sequences (k-mer + BPE)")
    print("=" * 65)
    from preprocessing.encoding import build_kmer_vocab

    all_seqs_list = list(sequences_raw.values())
    vocab = build_kmer_vocab(all_seqs_list, k=6)
    print(f"  k-mer vocabulary size: {len(vocab)}")

    vocab_path = os.path.join(RESULTS_DIR, "kmer_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(dict(list(vocab.items())[:1000]), f)
    pipeline_log["steps"].append({
        "step": 4, "name": "encode", "vocab_size": len(vocab),
    })

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: LSTM Training
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 5a — Training BiLSTM Sequence Predictor")
    print("=" * 65)
    from models.lstm_predictor import train_lstm
    lstm_model = train_lstm(sequences=all_seqs_list, epochs=lstm_epochs,
                            batch_size=16)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: Pre-train Transformer
    # ══════════════════════════════════════════════════════════════════════════
    from training.phase1_pretrain import run_phase1
    bert_model = run_phase1(sequences_raw, vocab,
                            epochs=phase1_epochs, batch_size=batch_size)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: Corruption Training
    # ══════════════════════════════════════════════════════════════════════════
    from training.phase2_corruption import run_phase2
    ae_model = run_phase2(sequences_raw, simulated,
                          ae_epochs=ae_epochs, batch_size=batch_size)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3: Evolution-Aware Training
    # ══════════════════════════════════════════════════════════════════════════
    from training.phase3_evolution_aware import run_phase3
    gnn_model, gnn_embeddings = run_phase3(sequences_raw,
                                            gnn_epochs=gnn_epochs)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 4: Fine-Tuning on Ancient DNA
    # ══════════════════════════════════════════════════════════════════════════
    from training.phase4_finetune import run_phase4
    bert_model = run_phase4(bert_model, simulated, MODERN_SEQS, vocab,
                            epochs=phase4_epochs, batch_size=batch_size)

    pipeline_log["steps"].append({"step": 5, "name": "4_phase_training"})

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 5: Transformer + GNN Fusion Training
    # ══════════════════════════════════════════════════════════════════════════
    all_species_names = sorted(sequences_raw.keys())
    fusion_model = None
    try:
        from training.phase5_fusion import run_phase5
        fusion_model = run_phase5(
            sequences_raw=sequences_raw,
            simulated=simulated,
            vocab=vocab,
            species_names=all_species_names,
            epochs=phase5_epochs,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"  [PHASE5] Fusion training error (continuing): {e}")
    pipeline_log["steps"].append({"step": "5b", "name": "fusion_training"})

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: Multi-Species Ensemble Reconstruction
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 6 — Multi-Species Ensemble Reconstruction")
    print("  (Transformer + GNN Fusion + AE + BERT + LSTM)")
    print("=" * 65)
    from models.ensemble_reconstructor import multi_species_ensemble_reconstruct

    reconstructions = multi_species_ensemble_reconstruct(
        simulated=simulated,
        sequences_raw=sequences_raw,
        bert_model=bert_model,
        ae_model=ae_model,
        lstm_model=lstm_model,
        vocab=vocab,
        fusion_model=fusion_model,
        species_names=all_species_names,
    )

    recon_path = os.path.join(RESULTS_DIR, "reconstructions.json")
    with open(recon_path, "w") as f:
        json.dump(reconstructions, f, indent=2, default=str)
    pipeline_log["steps"].append({"step": 6, "name": "multi_species_reconstruct"})

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 7: Evaluation Metrics
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 7 — Running Evaluation Metrics")
    print("=" * 65)
    from evaluation.benchmark import run_benchmark

    benchmark = run_benchmark(
        reconstructions=reconstructions,
        sequences_raw=sequences_raw,
        simulated=simulated,
        modern_seqs=MODERN_SEQS,
    )
    pipeline_log["steps"].append({"step": 7, "name": "benchmark"})

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 8: Fragment Mapping
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  STEP 8 — Mapping fragments to reference genome")
    print("=" * 65)
    from pipeline.genome_mapper import map_all_fragments

    mappings = {}
    for name, sim_result in simulated.items():
        frags    = sim_result.get("fragments", [])
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref      = MODERN_SEQS.get(ref_name, "ACGT" * 4000)
        m = map_all_fragments(frags, ref, species_name=name)
        mappings[name] = {
            "total":        m["total_fragments"],
            "mapped":       m["mapped_fragments"],
            "variants":     m["total_variants"],
            "hotspots":     len(m["hotspots"]),
            "disease_hits": len(m["disease_hits"]),
        }
        print(f"  {name}: {m['mapped_fragments']}/{m['total_fragments']} "
              f"mapped, {m['total_variants']} variants, "
              f"{len(m['hotspots'])} hotspots, "
              f"{len(m['disease_hits'])} disease loci")
    pipeline_log["steps"].append({"step": 8, "name": "mapping"})

    # ── Finalize ──────────────────────────────────────────────────────────────
    pipeline_log["end_time"]    = time.time()
    pipeline_log["elapsed_sec"] = round(
        pipeline_log["end_time"] - pipeline_log["start_time"], 1,
    )
    log_path = os.path.join(RESULTS_DIR, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  ✅  Pipeline complete in {pipeline_log['elapsed_sec']:.1f}s")
    print(f"  Results → {RESULTS_DIR}")
    print("=" * 65)

    return {
        "reconstructions": reconstructions,
        "alignments":      alignments,
        "mappings":        mappings,
        "benchmark":       benchmark,
        "gnn_embeddings":  gnn_embeddings.cpu().detach().numpy().tolist() if gnn_embeddings is not None else [],
        "vocab_size":      len(vocab),
    }


if __name__ == "__main__":
    run_pipeline()