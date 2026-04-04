"""
full_pipeline.py
================
Orchestrates the entire genome reconstruction pipeline.
"""

import os
import sys
import json
import time
import numpy as np

# ── Path setup — resolve relative to this file ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for sub in [
    "data",
    os.path.join("data", "sequences"),
    os.path.join("data", "simulated"),
    os.path.join("data", "alignments"),
    os.path.join("data", "mappings"),
    os.path.join("models", "checkpoints"),
    "results",
]:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── rest of full_pipeline.py continues unchanged from here ────────────────────


def run_pipeline(
    max_seq_length:   int  = 8000,
    lstm_epochs:      int  = 5,
    bert_epochs:      int  = 4,
    ae_epochs:        int  = 6,
    gnn_epochs:       int  = 30,
    batch_size:       int  = 16,
    skip_download:    bool = False,
):
    pipeline_log = {"steps": [], "start_time": time.time()}

    # ── STEP 1: Fetch sequences ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 1 — Fetching genome sequences from NCBI")
    print("="*65)
    from data.fetch_sequences import fetch_all, load_fasta, DATA_DIR
    if not skip_download:
        metadata = fetch_all()
    else:
        meta_path = os.path.join(DATA_DIR, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)
    pipeline_log["steps"].append({"step": 1, "name": "fetch", "species": list(metadata.keys())})

    # ── Load sequences into memory ─────────────────────────────────────────────
    sequences_raw = {}
    for name, info in metadata.items():
        recs = load_fasta(info["path"])
        seq  = next(iter(recs.values()))[:max_seq_length]
        sequences_raw[name] = seq
    print(f"\n  Loaded {len(sequences_raw)} sequences.")

    # Identify modern references
    MODERN = {"human_mtDNA", "elephant_mtDNA", "gray_wolf_mtDNA", "rock_pigeon_mtDNA"}
    ANCIENT = {k: v for k, v in sequences_raw.items() if k not in MODERN}
    MODERN_SEQS = {k: v for k, v in sequences_raw.items() if k in MODERN}

    # ── STEP 2: Simulate ancient damage ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 2 — Simulating ancient DNA damage")
    print("="*65)
    from data.simulate_ancient_dna import simulate_ancient_damage
    simulated = {}
    for name, seq in ANCIENT.items():
        result = simulate_ancient_damage(seq, name, seed=hash(name) % (2**31))
        simulated[name] = result
        s = result["mutation_summary"]
        print(f"  {name}: {sum(s.values())} mutations — {s}")
    pipeline_log["steps"].append({"step": 2, "name": "simulate", "count": len(simulated)})

    # ── STEP 3: Alignment ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 3 — Aligning ancient sequences to modern references")
    print("="*65)
    from preprocessing.alignment import align_to_reference, find_outlier_variants
    alignments = {}
    REF_MAP = {
        "neanderthal_mtDNA": "human_mtDNA",
        "mammoth_mtDNA":     "elephant_mtDNA",
        "woolly_rhino":      "elephant_mtDNA",
        "cave_bear_mtDNA":   "gray_wolf_mtDNA",
        "thylacine_mtDNA":   "gray_wolf_mtDNA",
        "passenger_pigeon":  "rock_pigeon_mtDNA",
        "dodo_partial":      "rock_pigeon_mtDNA",
        "saber_tooth_cat":   "gray_wolf_mtDNA",
    }
    for name, sim_result in simulated.items():
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref_seq  = MODERN_SEQS.get(ref_name, sequences_raw.get(ref_name, "ACGT"*100))
        aln = align_to_reference(
            query=sim_result["damaged_sequence"],
            reference=ref_seq,
            name=name,
        )
        outliers = find_outlier_variants(aln["variants"])
        aln["outlier_variants"] = outliers
        alignments[name] = aln
        print(f"  {name}: identity={aln['identity']:.3f}, "
              f"variants={aln['n_variants']}, outliers={len(outliers)}")
    pipeline_log["steps"].append({"step": 3, "name": "align", "results": {
        k: {"identity": v["identity"], "n_variants": v["n_variants"]}
        for k,v in alignments.items()
    }})

    # ── STEP 4: Encode sequences ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 4 — Encoding sequences")
    print("="*65)
    from preprocessing.encoding import build_kmer_vocab, integer_encode

    all_seqs_list = list(sequences_raw.values())
    vocab = build_kmer_vocab(all_seqs_list, k=6)
    print(f"  k-mer vocabulary size: {len(vocab)}")
    pipeline_log["steps"].append({"step": 4, "name": "encode", "vocab_size": len(vocab)})

    # Save vocab
    vocab_path = os.path.join(RESULTS_DIR, "kmer_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(dict(list(vocab.items())[:1000]), f)  # save first 1000

    # ── STEP 5: Train models ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 5 — Training Models")
    print("="*65)

    print("\n  [5a] Bidirectional LSTM Sequence Predictor")
    from models.lstm_predictor import train_lstm, BiLSTMPredictor
    lstm_model = train_lstm(
        sequences=all_seqs_list,
        epochs=lstm_epochs,
        batch_size=16,        # ← FIXED: was using passed batch_size, now hardcoded safe value
    )

    # 5b — DNABERT
    print("\n  [5b] DNABERT Transformer (Masked Language Model)")
    from models.dnabert_transformer import train_dnabert, DNABertModel
    bert_model = train_dnabert(
        sequences=all_seqs_list,
        vocab=vocab,
        epochs=bert_epochs,
        batch_size=batch_size,
    )

    # 5c — Denoising Autoencoder
    print("\n  [5c] Convolutional Denoising Autoencoder")
    from models.denoising_autoencoder import train_autoencoder, DenoisingAutoencoder
    clean_seqs = list(sequences_raw.values())
    noisy_seqs = [
        simulated.get(k, {"damaged_sequence": v})["damaged_sequence"]
        if k in simulated else v
        for k, v in sequences_raw.items()
    ]
# Ensure equal length lists
    min_len    = min(len(clean_seqs), len(noisy_seqs))
    clean_seqs = clean_seqs[:min_len]
    noisy_seqs = noisy_seqs[:min_len]

    ae_model = train_autoencoder(
    clean_seqs=clean_seqs,
    noisy_seqs=noisy_seqs,
    epochs=ae_epochs,
    batch_size=16,        # ← FIXED: safe batch size
)

    # 5d — Phylogenetic GNN
    print("\n  [5d] Phylogenetic Graph Neural Network")
    from models.gnn_phylogenetic import train_phylo_gnn
    gnn_model, gnn_embeddings = train_phylo_gnn(
        species_names=list(sequences_raw.keys()),
        sequences=sequences_raw,
        epochs=gnn_epochs,
    )

    pipeline_log["steps"].append({"step": 5, "name": "train_models"})

    # ── STEP 6: Reconstruct sequences ──────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 6 — Reconstructing sequences")
    print("="*65)
    import torch
    from models.lstm_predictor import predict_sequence
    from models.dnabert_transformer import fill_masked_sequence
    from models.denoising_autoencoder import denoise_sequence, train_autoencoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstructions = {}

    for name, sim_result in simulated.items():
        print(f"\n  Reconstructing: {name}")
        damaged = sim_result["damaged_sequence"]

        # Step A: AE denoising
        ae_recon, ae_confs, ae_repairs = denoise_sequence(ae_model, damaged, device)

        # Step B: BERT gap filling
        bert_recon, bert_confs = fill_masked_sequence(
            bert_model, ae_recon, vocab, device=device
        )

        # Step C: LSTM extension
        lstm_recon = predict_sequence(lstm_model, bert_recon[:1000], steps=3, device=device)

        # Combined confidence (average)
        final_seq = bert_recon
        n_gaps_before = damaged.count("N")
        n_gaps_after  = final_seq.count("N")
        coverage      = 1.0 - (n_gaps_after / max(1, len(final_seq)))

        # Per-base reliability (average BERT and AE confidences)
        min_len = min(len(ae_confs), len(bert_confs))
        combined_conf = [(ae_confs[i] + bert_confs[i]) / 2.0
                         for i in range(min_len)]
        reliability = np.mean(combined_conf) if combined_conf else 0.5

        reconstructions[name] = {
            "original_length":    len(sequences_raw.get(name, damaged)),
            "damaged_length":     len(damaged),
            "reconstructed_seq":  final_seq[:500] + "...",  # truncate for JSON
            "full_length":        len(final_seq),
            "gaps_before":        n_gaps_before,
            "gaps_after":         n_gaps_after,
            "coverage":           round(float(coverage), 4),
            "reliability_score":  round(float(reliability), 4),
            "mean_confidence":    round(float(np.mean(combined_conf)) if combined_conf else 0.5, 4),
            "ae_repairs":         ae_repairs[:20],   # first 20 for log
            "mutation_log":       sim_result["mutation_log"][:20],
            "mutation_summary":   sim_result["mutation_summary"],
        }

        print(f"    Coverage: {coverage:.2%}  Reliability: {reliability:.4f}  "
              f"Gaps: {n_gaps_before} → {n_gaps_after}")

    # Save reconstructions
    recon_path = os.path.join(RESULTS_DIR, "reconstructions.json")
    with open(recon_path, "w") as f:
        json.dump(reconstructions, f, indent=2)
    pipeline_log["steps"].append({"step": 6, "name": "reconstruct",
                                   "results": {k: {"coverage": v["coverage"],
                                                    "reliability": v["reliability_score"]}
                                               for k, v in reconstructions.items()}})

    # ── STEP 7: Fragment mapping ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 7 — Mapping fragments to reference genome")
    print("="*65)
    from pipeline.genome_mapper import map_all_fragments
    ref_seq = MODERN_SEQS.get("human_mtDNA", "ACGT" * 4000)
    mappings = {}
    for name, sim_result in simulated.items():
        frags = sim_result.get("fragments", [])
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref = MODERN_SEQS.get(ref_name, ref_seq)
        m = map_all_fragments(frags, ref, species_name=name)
        mappings[name] = {
            "total": m["total_fragments"],
            "mapped": m["mapped_fragments"],
            "variants": m["total_variants"],
            "hotspots": len(m["hotspots"]),
            "disease_hits": len(m["disease_hits"]),
        }
        print(f"  {name}: {m['mapped_fragments']}/{m['total_fragments']} mapped, "
              f"{m['total_variants']} variants, "
              f"{len(m['hotspots'])} hotspots, "
              f"{len(m['disease_hits'])} disease-associated loci")
    pipeline_log["steps"].append({"step": 7, "name": "mapping", "results": mappings})

    # ── Finalize pipeline log ───────────────────────────────────────────────────
    pipeline_log["end_time"]     = time.time()
    pipeline_log["elapsed_sec"]  = round(pipeline_log["end_time"] - pipeline_log["start_time"], 1)
    log_path = os.path.join(RESULTS_DIR, "pipeline_log.json")
    with open(log_path, "w") as f:
        json.dump(pipeline_log, f, indent=2)

    print("\n" + "="*65)
    print(f"  ✅  Pipeline complete in {pipeline_log['elapsed_sec']:.1f}s")
    print(f"  Results → {RESULTS_DIR}")
    print("="*65)

    return {
        "reconstructions": reconstructions,
        "alignments":      alignments,
        "mappings":        mappings,
        "gnn_embeddings":  gnn_embeddings.cpu().numpy().tolist(),
        "vocab_size":      len(vocab),
    }


if __name__ == "__main__":
    run_pipeline()