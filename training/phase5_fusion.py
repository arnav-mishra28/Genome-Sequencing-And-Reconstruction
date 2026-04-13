"""
phase5_fusion.py
================
Phase 5: Joint training of the Transformer + GNN Fusion model.

Loss = MLM + Reconstruction + Confidence_Calibration + GNN_Bio_Constraint

This phase runs after Phase 4 (fine-tuning) and trains the unified
fusion model that combines sequence-level and evolutionary-level reasoning.
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
    MODEL_DIR, DEVICE, USE_AMP, EMBED_DIM,
    MAX_SEQ_LEN, BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP,
    PHASE5_EPOCHS,
)


def run_phase5(
    sequences_raw:  Dict[str, str],
    simulated:      Dict[str, Dict],
    vocab:          Dict[str, int],
    species_names:  List[str],
    epochs:         int   = None,
    batch_size:     int   = None,
    lr:             float = 1e-4,
) -> "TransformerGNNFusion":
    """
    Train the Transformer+GNN Fusion model with joint objectives.

    Args:
        sequences_raw: all species sequences (modern + ancient)
        simulated:     ancient damage simulation results
        vocab:         k-mer vocabulary
        species_names: ordered list of species
        epochs:        training epochs
        batch_size:    batch size
        lr:            learning rate

    Returns:
        Trained TransformerGNNFusion model
    """
    from models.fusion_model import (
        TransformerGNNFusion,
        build_fusion_phylo_graph,
    )
    from models.gnn_phylogenetic import BiologicalConstraintLoss
    from data.dataset_builder import PretrainDataset

    epochs = epochs or PHASE5_EPOCHS
    batch_size = batch_size or BATCH_SIZE
    device = DEVICE

    print("\n" + "=" * 65)
    print("  PHASE 5 — Transformer + GNN Fusion Training")
    print("=" * 65)
    print(f"  Device: {device} | AMP: {USE_AMP}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # ── Build phylogenetic graph ──────────────────────────────────────────────
    species_feats, adjacency = build_fusion_phylo_graph(
        species_names, sequences_raw,
    )
    species_feats = species_feats.to(device)
    adjacency = adjacency.to(device)
    print(f"  Species: {len(species_names)} | "
          f"Features: {species_feats.shape} | Adj: {adjacency.shape}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    # Use all sequences (modern + ancient damaged)
    all_seqs = []
    seq_species_map = []  # track which species each sequence belongs to

    for sp_name in species_names:
        if sp_name in simulated:
            # Use damaged sequence for ancient species
            seq = simulated[sp_name].get("damaged_sequence", "")
            if seq:
                all_seqs.append(seq)
                seq_species_map.append(species_names.index(sp_name))
        if sp_name in sequences_raw:
            seq = sequences_raw[sp_name]
            if seq:
                all_seqs.append(seq)
                seq_species_map.append(species_names.index(sp_name))

    if not all_seqs:
        print("  [PHASE5] No sequences available — skipping")
        return None

    dataset = PretrainDataset(all_seqs, vocab, max_len=MAX_SEQ_LEN)
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
    model = TransformerGNNFusion(
        vocab_size=len(vocab),
        n_species_feat=species_feats.shape[1],
        max_len=MAX_SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Optimiser & Losses ────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs, pct_start=0.1,
    )

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    recon_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    bio_loss_fn = BiologicalConstraintLoss()

    # Confidence calibration loss: BCE between predicted confidence
    # and actual correctness
    conf_criterion = nn.BCELoss()

    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history = []
    best_val = float("inf")

    # ── GC contents for bio constraint ────────────────────────────────────────
    gc_contents = []
    for sp in species_names:
        seq = sequences_raw.get(sp, "")
        gc = (seq.upper().count("G") + seq.upper().count("C")) / max(1, len(seq))
        gc_contents.append(gc)
    gc_tensor = torch.tensor(gc_contents, dtype=torch.float).to(device)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_mlm, total_conf, total_bio = 0, 0, 0, 0
        n_batches = 0

        for tokens, labels, att in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            att    = att.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(
                tokens, att,
                species_feats=species_feats,
                adjacency=adjacency,
            )

            # MLM loss
            mlm_logits = outputs["mlm_logits"]
            mlm_loss = mlm_criterion(
                mlm_logits.reshape(-1, len(vocab)),
                labels.reshape(-1),
            )

            # Confidence calibration loss
            with torch.no_grad():
                preds = mlm_logits.argmax(dim=-1)
                valid_mask = (labels != -100)
                correct = (preds == labels).float()
                correct = correct * valid_mask.float()

            per_base_conf = outputs["per_base_conf"]
            if valid_mask.any():
                conf_loss = conf_criterion(
                    per_base_conf[valid_mask],
                    correct[valid_mask],
                )
            else:
                conf_loss = torch.tensor(0.0, device=device)

            # Biological constraint loss
            bio_loss = bio_loss_fn(gc_tensor)

            # Joint loss
            loss = (
                1.0  * mlm_loss +
                0.5  * conf_loss +
                0.05 * bio_loss
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
            total_mlm  += mlm_loss.item()
            total_conf += conf_loss.item()
            total_bio  += bio_loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_mlm  = total_mlm  / max(1, n_batches)
        avg_conf = total_conf / max(1, n_batches)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for tokens, labels, att in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                att    = att.to(device)
                outputs = model(
                    tokens, att,
                    species_feats=species_feats,
                    adjacency=adjacency,
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
            torch.save(
                {
                    "model":       model.state_dict(),
                    "vocab_size":  len(vocab),
                    "n_species":   len(species_names),
                    "species":     species_names,
                },
                os.path.join(MODEL_DIR, "fusion_best.pt"),
            )

        temp = float(outputs.get("temperature",
                                  torch.tensor(1.5)).detach().cpu())
        print(
            f"  [FUSION] Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_loss:.4f} | mlm={avg_mlm:.4f} | "
            f"conf={avg_conf:.4f} | val={avg_val:.4f} | "
            f"temp={temp:.3f}"
        )

        history.append({
            "epoch":    epoch,
            "loss":     round(avg_loss, 6),
            "mlm_loss": round(avg_mlm, 6),
            "conf_loss": round(avg_conf, 6),
            "val_loss": round(avg_val, 6),
            "temperature": round(temp, 4),
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt = os.path.join(MODEL_DIR, "fusion.pt")
    torch.save(
        {
            "model":       model.state_dict(),
            "vocab_size":  len(vocab),
            "n_species":   len(species_names),
            "species":     species_names,
        },
        ckpt,
    )
    with open(os.path.join(MODEL_DIR, "fusion_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [FUSION] Saved → {ckpt}")

    return model
