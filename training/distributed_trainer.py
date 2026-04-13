"""
distributed_trainer.py
======================
DeepSpeed Distributed Training for Large Genome Models.

Supports:
  - DeepSpeed ZeRO Stage 2/3 optimization
  - Mixed precision (FP16/BF16)
  - Gradient checkpointing for memory efficiency
  - Multi-GPU data parallelism
  - Configurable model scaling (4→48 layers, 128→2048 hidden)
  - Automatic fallback to single-GPU/CPU if DeepSpeed unavailable

Usage:
  python main.py train-distributed --profile medium --epochs 10
  
  Or with DeepSpeed launcher:
  deepspeed main.py train-distributed --profile large --epochs 20
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, MAX_SEQ_LEN,
    BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP, BASE_DIR,
)

# Check DeepSpeed availability
_HAS_DEEPSPEED = False
try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ImportError:
    pass


def get_deepspeed_config(profile: str = "medium") -> Dict:
    """
    Generate DeepSpeed configuration based on scale profile.
    """
    base_config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": GRADIENT_CLIP,

        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },

        "bf16": {
            "enabled": False,
        },

        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },

        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },

        "scheduler": {
            "type": "OneCycleLR",
            "params": {
                "max_lr": 1e-4,
                "pct_start": 0.1,
                "anneal_strategy": "cos",
            },
        },

        "wall_clock_breakdown": False,
    }

    # Profile-specific adjustments
    if profile == "small":
        base_config["train_batch_size"] = 16
        base_config["gradient_accumulation_steps"] = 2
        base_config["zero_optimization"]["stage"] = 2
    elif profile == "medium":
        base_config["train_batch_size"] = 32
        base_config["gradient_accumulation_steps"] = 4
        base_config["zero_optimization"]["stage"] = 2
    elif profile == "large":
        base_config["train_batch_size"] = 64
        base_config["gradient_accumulation_steps"] = 8
        base_config["zero_optimization"]["stage"] = 3
        base_config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        base_config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
    elif profile == "xl":
        base_config["train_batch_size"] = 128
        base_config["gradient_accumulation_steps"] = 16
        base_config["zero_optimization"]["stage"] = 3
        base_config["zero_optimization"]["stage3_max_live_parameters"] = 3e9
        base_config["zero_optimization"]["stage3_max_reuse_distance"] = 3e9
        base_config["activation_checkpointing"]["cpu_checkpointing"] = True

    return base_config


def run_distributed_training(
    sequences_raw:  Dict[str, str],
    simulated:      Dict[str, Dict],
    vocab:          Dict[str, int],
    species_names:  List[str],
    evolution_seqs: Dict[str, str] = None,
    epochs:         int   = 5,
    batch_size:     int   = None,
    lr:             float = 1e-4,
    scale_profile:  str   = "medium",
    local_rank:     int   = -1,
) -> Optional["EvoformerGenomeModel"]:
    """
    Train EvoformerGenomeModel with DeepSpeed distributed training.
    Falls back to standard PyTorch training if DeepSpeed is unavailable.
    """
    from models.evoformer_model import (
        EvoformerGenomeModel, SCALE_PROFILES, EvoformerConfig,
    )
    from data.dataset_builder import PretrainDataset
    from data.multi_species_loader import (
        compute_species_features,
        build_phylo_adjacency,
        EVOLUTION_DISTANCES,
    )
    from models.gnn_phylogenetic import BiologicalConstraintLoss

    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 65)
    print("  DISTRIBUTED TRAINING — EvoformerGenomeModel")
    print("=" * 65)

    # ── Merge sequences ───────────────────────────────────────────────────────
    all_sequences = dict(sequences_raw)
    if evolution_seqs:
        all_sequences.update(evolution_seqs)

    all_species = sorted(all_sequences.keys())

    # ── Build features ────────────────────────────────────────────────────────
    features, _ = compute_species_features(all_sequences, k=4)

    from config.settings import PHYLO_DISTANCES
    all_distances = dict(PHYLO_DISTANCES)
    all_distances.update(EVOLUTION_DISTANCES)
    adj = build_phylo_adjacency(all_species, all_distances)

    # ── Dataset ───────────────────────────────────────────────────────────────
    all_seqs_list = []
    for sp_name in all_species:
        if sp_name in all_sequences:
            seq = all_sequences[sp_name]
            if seq:
                all_seqs_list.append(seq)
        if sp_name in simulated:
            dam_seq = simulated[sp_name].get("damaged_sequence", "")
            if dam_seq:
                all_seqs_list.append(dam_seq)

    if not all_seqs_list:
        print("  No sequences — skipping")
        return None

    dataset = PretrainDataset(all_seqs_list, vocab, max_len=MAX_SEQ_LEN)
    val_size = max(1, int(0.1 * len(dataset)))
    trn_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [trn_size, val_size]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    config = SCALE_PROFILES.get(scale_profile, SCALE_PROFILES["small"])
    config.vocab_size = len(vocab)
    config.n_species = len(all_species)
    config.species_feat_dim = features.shape[1]
    config.max_len = MAX_SEQ_LEN

    model = EvoformerGenomeModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Profile: {scale_profile} | Params: {total_params:,}")
    print(f"  DeepSpeed available: {_HAS_DEEPSPEED}")

    # ══════════════════════════════════════════════════════════════════════════
    #  DeepSpeed Path
    # ══════════════════════════════════════════════════════════════════════════
    if _HAS_DEEPSPEED and torch.cuda.is_available():
        print("  Using DeepSpeed distributed training")

        ds_config = get_deepspeed_config(scale_profile)
        ds_config["train_batch_size"] = batch_size
        ds_config["optimizer"]["params"]["lr"] = lr

        # Save config
        config_path = os.path.join(BASE_DIR, "config", "deepspeed_config.json")
        with open(config_path, "w") as f:
            json.dump(ds_config, f, indent=2)

        # Enable gradient checkpointing for large models
        if scale_profile in ("large", "xl"):
            model.gradient_checkpointing_enable = True
            deepspeed.checkpointing.configure(
                None, partition_activations=True
            )

        # Initialize DeepSpeed
        model_engine, optimizer, train_loader_ds, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_ds,
            config=ds_config,
        )

        device = model_engine.device
        species_feats = torch.tensor(features, dtype=torch.float).to(device)
        adjacency_t = torch.tensor(adj, dtype=torch.float).to(device)

        mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        history = []

        for epoch in range(1, epochs + 1):
            model_engine.train()
            total_loss, n_batches = 0, 0

            for tokens, labels, att in train_loader_ds:
                tokens = tokens.to(device)
                labels = labels.to(device)
                att = att.to(device)

                outputs = model_engine(
                    tokens, att,
                    species_feats=species_feats,
                    adjacency=adjacency_t,
                    n_recycles=1,
                )

                loss = mlm_criterion(
                    outputs["mlm_logits"].reshape(-1, len(vocab)),
                    labels.reshape(-1),
                )

                model_engine.backward(loss)
                model_engine.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            print(f"  [DS] Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f}")
            history.append({"epoch": epoch, "loss": round(avg_loss, 6)})

        # Save
        ckpt_dir = os.path.join(MODEL_DIR, "evoformer_ds")
        model_engine.save_checkpoint(ckpt_dir)

        # Also save standard checkpoint
        ckpt = os.path.join(MODEL_DIR, "evoformer.pt")
        torch.save({
            "model": model_engine.module.state_dict(),
            "config": vars(config),
            "species": all_species,
        }, ckpt)

        with open(os.path.join(MODEL_DIR, "evoformer_ds_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"  [DS] Saved → {ckpt_dir}")

        return model_engine.module

    # ══════════════════════════════════════════════════════════════════════════
    #  Standard PyTorch Fallback (with mixed precision + grad checkpointing)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        print("  Using standard PyTorch training (with AMP)")

        device = DEVICE
        model = model.to(device)
        species_feats = torch.tensor(features, dtype=torch.float).to(device)
        adjacency_t = torch.tensor(adj, dtype=torch.float).to(device)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'seq_blocks'):
            for block in model.seq_blocks:
                if hasattr(block, 'gradient_checkpointing'):
                    block.gradient_checkpointing = True

        kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                  pin_memory=False, drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_ds, shuffle=True, **kw
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, shuffle=False, **kw
        )

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr,
            steps_per_epoch=max(1, len(train_loader)),
            epochs=epochs, pct_start=0.1,
        )
        mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        scaler = torch.amp.GradScaler(enabled=USE_AMP)

        history = []
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, n_batches = 0, 0

            for tokens, labels, att in train_loader:
                tokens = tokens.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                att = att.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                if USE_AMP and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = model(
                            tokens, att,
                            species_feats=species_feats,
                            adjacency=adjacency_t,
                            n_recycles=1,
                        )
                        loss = mlm_criterion(
                            outputs["mlm_logits"].reshape(-1, len(vocab)),
                            labels.reshape(-1),
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    scaler.step(opt)
                    scaler.update()
                else:
                    outputs = model(
                        tokens, att,
                        species_feats=species_feats,
                        adjacency=adjacency_t,
                        n_recycles=1,
                    )
                    loss = mlm_criterion(
                        outputs["mlm_logits"].reshape(-1, len(vocab)),
                        labels.reshape(-1),
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    opt.step()

                sched.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)

            # Validate
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
                        adjacency=adjacency_t,
                        n_recycles=1,
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
                    },
                    "species": all_species,
                }, os.path.join(MODEL_DIR, "evoformer_best.pt"))

            print(f"  [STD] Epoch {epoch:02d}/{epochs} | "
                  f"loss={avg_loss:.4f} | val={avg_val:.4f}")
            history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "val_loss": round(avg_val, 6),
            })

        # Save
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
            },
            "species": all_species,
        }, ckpt)

        with open(os.path.join(MODEL_DIR, "evoformer_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"  [STD] Saved → {ckpt}")

        return model
