"""
phase6_evoformer.py
===================
Phase 6: Evoformer Training — AlphaFold-Inspired Multi-Species Training.

Training Stages:
  Stage 1: Pre-train on complete genomes (standard MLM)
  Stage 2: Corrupt sequences and train reconstruction
  Stage 3: Multi-species combined training with species embeddings
  Stage 4: Evolution-aware fine-tuning with phylogenetic constraints

Combined Loss:
  L = MLM + Reconstruction + Evolution + Bio_Constraint + Confidence_Calibration
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, MAX_SEQ_LEN, BATCH_SIZE,
    NUM_WORKERS, GRADIENT_CLIP,
)


def run_phase6(
    sequences_raw:  Dict[str, str],
    simulated:      Dict[str, Dict],
    vocab:          Dict[str, int],
    species_names:  List[str],
    evolution_seqs: Dict[str, str] = None,
    epochs:         int   = 3,
    batch_size:     int   = None,
    lr:             float = 1e-4,
    scale_profile:  str   = "small",
) -> "EvoformerGenomeModel":
    """
    Train the Evoformer genome model with multi-stage curriculum.

    Args:
        sequences_raw:  all species sequences (modern + ancient)
        simulated:      ancient damage simulation results
        vocab:          k-mer vocabulary
        species_names:  ordered list of species
        evolution_seqs: additional evolution species sequences
        epochs:         training epochs per stage
        batch_size:     batch size
        lr:             learning rate
        scale_profile:  model size ("small", "medium", "large", "xl")

    Returns:
        Trained EvoformerGenomeModel
    """
    from models.evoformer_model import (
        EvoformerGenomeModel, get_model, EvolutionLoss,
        SCALE_PROFILES,
    )
    from models.gnn_phylogenetic import BiologicalConstraintLoss
    from data.dataset_builder import PretrainDataset
    from data.multi_species_loader import (
        compute_species_features,
        build_phylo_adjacency,
        EVOLUTION_DISTANCES,
    )

    batch_size = batch_size or BATCH_SIZE
    device = DEVICE

    print("\n" + "=" * 65)
    print("  PHASE 6 — Evoformer (AlphaFold-Inspired) Training")
    print("=" * 65)
    print(f"  Device: {device} | AMP: {USE_AMP}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Scale profile: {scale_profile}")

    # ── Merge all sequences ───────────────────────────────────────────────────
    all_sequences = dict(sequences_raw)
    if evolution_seqs:
        all_sequences.update(evolution_seqs)

    # Use either provided species_names or auto-detect
    all_species = sorted(all_sequences.keys())
    print(f"  Total species: {len(all_species)}")

    # ── Build phylogenetic features ───────────────────────────────────────────
    features, feat_species = compute_species_features(all_sequences, k=4)
    species_feats = torch.tensor(features, dtype=torch.float).to(device)

    # Merge distance maps
    from config.settings import PHYLO_DISTANCES
    all_distances = dict(PHYLO_DISTANCES)
    all_distances.update(EVOLUTION_DISTANCES)

    adj = build_phylo_adjacency(all_species, all_distances)
    adjacency = torch.tensor(adj, dtype=torch.float).to(device)

    print(f"  Species features: {species_feats.shape}")
    print(f"  Adjacency: {adjacency.shape}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    all_seqs_list = []
    seq_species_indices = []

    for sp_name in all_species:
        # Add clean sequence
        if sp_name in all_sequences:
            seq = all_sequences[sp_name]
            if seq:
                all_seqs_list.append(seq)
                seq_species_indices.append(all_species.index(sp_name))

        # Add damaged sequence if available
        if sp_name in simulated:
            dam_seq = simulated[sp_name].get("damaged_sequence", "")
            if dam_seq:
                all_seqs_list.append(dam_seq)
                seq_species_indices.append(all_species.index(sp_name))

    if not all_seqs_list:
        print("  [PHASE6] No sequences available — skipping")
        return None

    print(f"  Dataset: {len(all_seqs_list)} sequences")

    dataset = PretrainDataset(all_seqs_list, vocab, max_len=MAX_SEQ_LEN)
    val_size = max(1, int(0.1 * len(dataset)))
    trn_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [trn_size, val_size]
    )

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
              pin_memory=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, **kw
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, **kw
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    config = SCALE_PROFILES.get(scale_profile, SCALE_PROFILES["small"])
    config.vocab_size = len(vocab)
    config.n_species = len(all_species)
    config.species_feat_dim = species_feats.shape[1]
    config.max_len = MAX_SEQ_LEN

    model = EvoformerGenomeModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Config: embed={config.embed_dim}, heads={config.n_heads}, "
          f"layers={config.n_seq_blocks}, evo_blocks={config.n_evo_blocks}, "
          f"recycles={config.n_recycles}")

    # ── Optimizer & Losses ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs, pct_start=0.1,
    )

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    conf_criterion = nn.BCELoss()
    bio_loss_fn = BiologicalConstraintLoss()
    evo_loss_fn = EvolutionLoss()

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # GC contents for bio loss
    gc_contents = []
    for sp in all_species:
        seq = all_sequences.get(sp, "")
        gc = (seq.upper().count("G") + seq.upper().count("C")) / max(1, len(seq))
        gc_contents.append(gc)
    gc_tensor = torch.tensor(gc_contents, dtype=torch.float).to(device)

    history = []
    best_val = float("inf")

    # ══════════════════════════════════════════════════════════════════════════
    #  Training Loop (all stages in one — curriculum by epoch progression)
    # ══════════════════════════════════════════════════════════════════════════
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_mlm = 0
        total_evo = 0
        total_conf = 0
        n_batches = 0

        # Adjust recycling: use fewer recycles early, more later
        n_recycles = min(config.n_recycles, 1 + epoch // 2)

        for tokens, labels, att in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            att = att.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Forward with Evoformer
            outputs = model(
                tokens, att,
                species_feats=species_feats,
                adjacency=adjacency,
                n_recycles=n_recycles,
            )

            # ── MLM loss ──────────────────────────────────────────────────────
            mlm_logits = outputs["mlm_logits"]
            mlm_loss = mlm_criterion(
                mlm_logits.reshape(-1, len(vocab)),
                labels.reshape(-1),
            )

            # ── Confidence calibration loss ───────────────────────────────────
            with torch.no_grad():
                preds = mlm_logits.argmax(dim=-1)
                valid_mask = (labels != -100)
                correct = (preds == labels).float() * valid_mask.float()

            per_base_conf = outputs["per_base_conf"]
            if valid_mask.any():
                conf_loss = conf_criterion(
                    per_base_conf[valid_mask],
                    correct[valid_mask],
                )
            else:
                conf_loss = torch.tensor(0.0, device=device)

            # ── Evolution loss ────────────────────────────────────────────────
            recon_logits = outputs.get("recon_logits")
            mutation_prob = outputs.get("mutation_prob")
            evo_loss = torch.tensor(0.0, device=device)
            if recon_logits is not None and mutation_prob is not None:
                # Use dummy phylo distances for now (batch-level)
                phylo_dist = torch.tensor(
                    [50.0] * tokens.size(0), device=device
                )
                evo_loss = evo_loss_fn(
                    recon_logits, mutation_prob, phylo_dist, gc_tensor[:1].expand(tokens.size(0)),
                )

            # ── Biological constraint loss ────────────────────────────────────
            bio_loss = bio_loss_fn(gc_tensor)

            # ── Joint loss ────────────────────────────────────────────────────
            loss = (
                1.0   * mlm_loss +
                0.5   * conf_loss +
                0.1   * evo_loss +
                0.05  * bio_loss
            )

            # Backward
            if USE_AMP and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

            sched.step()

            total_loss += loss.item()
            total_mlm += mlm_loss.item()
            total_evo += evo_loss.item()
            total_conf += conf_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_mlm = total_mlm / max(1, n_batches)
        avg_evo = total_evo / max(1, n_batches)
        avg_conf = total_conf / max(1, n_batches)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for tokens, labels, att in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                att = att.to(device)
                outputs = model(
                    tokens, att,
                    species_feats=species_feats,
                    adjacency=adjacency,
                    n_recycles=1,  # single pass for speed
                )
                vl = mlm_criterion(
                    outputs["mlm_logits"].reshape(-1, len(vocab)),
                    labels.reshape(-1),
                )
                val_loss += vl.item()
                vn += 1
        avg_val = val_loss / max(1, vn)

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "vocab_size": config.vocab_size,
                    "embed_dim": config.embed_dim,
                    "pair_dim": config.pair_dim,
                    "n_heads": config.n_heads,
                    "n_evo_blocks": config.n_evo_blocks,
                    "n_seq_blocks": config.n_seq_blocks,
                    "ffn_dim": config.ffn_dim,
                    "max_len": config.max_len,
                    "n_species": config.n_species,
                    "n_recycles": config.n_recycles,
                    "species_feat_dim": config.species_feat_dim,
                },
                "species": all_species,
            }, os.path.join(MODEL_DIR, "evoformer_best.pt"))

        temp = float(outputs.get("temperature", torch.tensor(1.5)).detach().cpu())
        print(
            f"  [EVO] Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_loss:.4f} | mlm={avg_mlm:.4f} | "
            f"evo={avg_evo:.4f} | conf={avg_conf:.4f} | "
            f"val={avg_val:.4f} | recycles={n_recycles} | "
            f"temp={temp:.3f}"
        )

        history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "mlm_loss": round(avg_mlm, 6),
            "evo_loss": round(avg_evo, 6),
            "conf_loss": round(avg_conf, 6),
            "val_loss": round(avg_val, 6),
            "n_recycles": n_recycles,
            "temperature": round(temp, 4),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt = os.path.join(MODEL_DIR, "evoformer.pt")
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "pair_dim": config.pair_dim,
            "n_heads": config.n_heads,
            "n_evo_blocks": config.n_evo_blocks,
            "n_seq_blocks": config.n_seq_blocks,
            "ffn_dim": config.ffn_dim,
            "max_len": config.max_len,
            "n_species": config.n_species,
            "n_recycles": config.n_recycles,
            "species_feat_dim": config.species_feat_dim,
        },
        "species": all_species,
    }, ckpt)

    with open(os.path.join(MODEL_DIR, "evoformer_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [EVO] Saved → {ckpt}")

    return model
