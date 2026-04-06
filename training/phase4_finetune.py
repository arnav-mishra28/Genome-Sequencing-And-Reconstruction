"""
phase4_finetune.py
==================
Phase 4: Fine-Tuning on real ancient DNA fragments.
Uses Neanderthal + Woolly Mammoth data aligned to modern references.
Lower learning rate, frozen lower layers.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from typing import Dict, List
from config.settings import (
    PHASE4_EPOCHS, BATCH_SIZE, DEVICE, USE_AMP,
    NUM_WORKERS, GRADIENT_CLIP, MAX_SEQ_LEN, MODEL_DIR, REF_MAP,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


def run_phase4(
    bert_model,
    simulated:       Dict[str, Dict],
    modern_seqs:     Dict[str, str],
    vocab:           Dict[str, int],
    epochs:          int = None,
    batch_size:      int = None,
    lr:              float = 5e-5,
) -> object:
    """
    Fine-tune DNABERT-2 on real ancient DNA fragments with known references.
    Uses frozen lower transformer layers + lower learning rate.

    Returns fine-tuned model.
    """
    from data.dataset_builder import AncientDNADataset

    epochs     = epochs or PHASE4_EPOCHS
    batch_size = batch_size or BATCH_SIZE

    print("\n" + "=" * 65)
    print("  PHASE 4 — Fine-Tuning on Real Ancient DNA")
    print(f"  LR: {lr}  |  Epochs: {epochs}")
    print("  Strategy: Freeze lower layers, fine-tune upper + MLM head")
    print("=" * 65)

    # ── Freeze lower transformer layers ───────────────────────────────────────
    n_layers = len(bert_model.blocks)
    freeze_up_to = max(1, n_layers // 2)

    for i, block in enumerate(bert_model.blocks):
        if i < freeze_up_to:
            for param in block.parameters():
                param.requires_grad = False
    print(f"  Frozen layers: 0-{freeze_up_to - 1} of {n_layers}")

    # ── Build dataset from ancient fragments ──────────────────────────────────
    all_fragments = []
    ref_for_frags = ""

    for name, sim_result in simulated.items():
        ref_name = REF_MAP.get(name, "human_mtDNA")
        ref_seq  = modern_seqs.get(ref_name, "ACGT" * 4000)
        frags    = sim_result.get("fragments", [])

        if len(frags) > len(all_fragments):
            # Use the species with most fragments
            all_fragments = frags
            ref_for_frags = ref_seq

    if not all_fragments:
        print("  [WARN] No ancient fragments available — skipping Phase 4.")
        return bert_model

    full_ds  = AncientDNADataset(all_fragments, ref_for_frags, vocab,
                                  max_len=MAX_SEQ_LEN)
    val_size = max(1, int(0.1 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
              pin_memory=False, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    device = DEVICE
    bert_model.to(device)

    # Only optimise unfrozen params
    trainable = [p for p in bert_model.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    crit  = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    for epoch in range(1, epochs + 1):
        bert_model.train()
        total_loss, n_batches = 0.0, 0

        for frag_tok, ref_tok, att in train_loader:
            frag_tok = frag_tok.to(device)
            ref_tok  = ref_tok.to(device)
            att      = att.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = bert_model(frag_tok, att)
                loss   = crit(logits.reshape(-1, logits.shape[-1]),
                              ref_tok.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(1, n_batches)

        # Validate
        bert_model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for frag_tok, ref_tok, att in val_loader:
                frag_tok = frag_tok.to(device)
                ref_tok  = ref_tok.to(device)
                att      = att.to(device)
                logits   = bert_model(frag_tok, att)
                val_loss += crit(logits.reshape(-1, logits.shape[-1]),
                                 ref_tok.reshape(-1)).item()
                vn += 1
        avg_val = val_loss / max(1, vn)

        print(f"  [FT] Epoch {epoch:02d}/{epochs} | "
              f"loss={avg_train:.4f} | val={avg_val:.4f}")

    # Unfreeze all layers
    for param in bert_model.parameters():
        param.requires_grad = True

    # Save fine-tuned checkpoint
    torch.save(
        {"model": bert_model.state_dict(), "vocab_size": len(vocab)},
        os.path.join(MODEL_DIR, "dnabert2_finetuned.pt"),
    )
    print("  ✅ Phase 4 complete — model fine-tuned on ancient DNA.\n")
    return bert_model
