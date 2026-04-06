"""
dnabert2_transformer.py
=======================
DNABERT-2-inspired genomic transformer for masked DNA prediction.

Key architectural advances over DNABERT-1:
  - BPE tokenization (replaces fixed k-mer)
  - ALiBi (Attention with Linear Biases) positional encoding
  - GEGLU activation in feed-forward layers
  - Pre-Norm transformer encoder blocks

References:
  - DNABERT-2: Efficient Foundation Model for Multi-Species Genome
    (Zhou et al., 2023, ICLR 2024)
"""

import os
import sys
import json
import math
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, EMBED_DIM, N_HEADS, N_LAYERS,
    FFN_DIM, MAX_SEQ_LEN, DROPOUT, MASK_PROB, MAX_SAMPLES,
    BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALiBi (Attention with Linear Biases)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute slopes for ALiBi attention bias (Press et al., 2022)."""

    def _get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_get_slopes_power_of_2(n_heads))
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest_pow2)
        extra  = _get_slopes_power_of_2(2 * closest_pow2)
        slopes += extra[0::2][: n_heads - closest_pow2]
        return torch.tensor(slopes)


def build_alibi_bias(n_heads: int, max_len: int) -> torch.Tensor:
    """
    Returns ALiBi attention bias of shape (n_heads, max_len, max_len).
    Each head has a different linear distance penalty.
    """
    slopes = _get_alibi_slopes(n_heads)  # (H,)
    positions = torch.arange(max_len)
    rel_pos   = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
    rel_pos   = rel_pos.float().unsqueeze(0)                      # (1, L, L)
    slopes    = slopes.unsqueeze(1).unsqueeze(2)                  # (H, 1, 1)
    return slopes * rel_pos  # (H, L, L)


# ═══════════════════════════════════════════════════════════════════════════════
#  GEGLU Feed-Forward
# ═══════════════════════════════════════════════════════════════════════════════
class GEGLU(nn.Module):
    """Gated GELU activation (Shazeer, 2020)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class GEGLUFeedForward(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim * 2),  # 2x for gating
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Transformer Block (Pre-Norm + ALiBi)
# ═══════════════════════════════════════════════════════════════════════════════
class DNABERT2Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn   = GEGLUFeedForward(dim, ffn_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                alibi_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # Pre-norm self-attention with ALiBi
        h = self.norm1(x)
        h, _ = self.attn(h, h, h,
                         attn_mask=alibi_bias,
                         key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        # Feed-forward with residual
        x = x + self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  DNABERT-2 Model
# ═══════════════════════════════════════════════════════════════════════════════
class DNABERT2Model(nn.Module):
    """
    DNABERT-2-inspired genomic transformer.

    Key features:
      - No positional embeddings (uses ALiBi instead)
      - GEGLU feed-forward layers
      - Pre-LayerNorm architecture
      - MLM head for masked prediction
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int   = EMBED_DIM,
        n_heads:    int   = N_HEADS,
        n_layers:   int   = N_LAYERS,
        ffn_dim:    int   = FFN_DIM,
        max_len:    int   = MAX_SEQ_LEN,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.n_heads    = n_heads
        self.max_len    = max_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop  = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DNABERT2Block(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        # Pre-compute ALiBi bias (register as buffer so it moves with .to())
        alibi = build_alibi_bias(n_heads, max_len)
        self.register_buffer("alibi_bias", alibi, persistent=False)

    def forward(
        self,
        tokens:         torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor = None,  # (B, L) 1=real, 0=pad
    ) -> torch.Tensor:                        # (B, L, vocab_size)
        B, L = tokens.shape

        x = self.embed_drop(self.token_embed(tokens))

        # Prepare ALiBi bias for current sequence length
        alibi = self.alibi_bias[:, :L, :L]  # (H, L, L)
        # Expand for batched attention: (B*H, L, L)
        alibi = alibi.unsqueeze(0).expand(B, -1, -1, -1)
        alibi = alibi.reshape(B * self.n_heads, L, L)

        key_pad = None
        if attention_mask is not None:
            key_pad = (attention_mask == 0).float()  # match attn_mask dtype

        for block in self.blocks:
            x = block(x, alibi_bias=alibi, key_padding_mask=key_pad)

        x = self.final_norm(x)
        return self.mlm_head(x)

    def get_embeddings(self, tokens: torch.Tensor,
                       attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Return hidden representations without the MLM head."""
        B, L = tokens.shape
        x = self.embed_drop(self.token_embed(tokens))

        alibi = self.alibi_bias[:, :L, :L]
        alibi = alibi.unsqueeze(0).expand(B, -1, -1, -1)
        alibi = alibi.reshape(B * self.n_heads, L, L)

        key_pad = (attention_mask == 0).float() if attention_mask is not None else None

        for block in self.blocks:
            x = block(x, alibi_bias=alibi, key_padding_mask=key_pad)

        return self.final_norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════
def train_dnabert2(
    sequences:  List[str],
    vocab:      Dict[str, int],
    epochs:     int   = 4,
    batch_size: int   = BATCH_SIZE,
    lr:         float = 2e-4,
    max_len:    int   = MAX_SEQ_LEN,
) -> "DNABERT2Model":
    """Train DNABERT-2 with masked language modeling."""
    from data.dataset_builder import PretrainDataset

    device = DEVICE
    print(f"  [DNABERT-2] Device: {device} | AMP: {USE_AMP}")

    # Dataset
    full_ds  = PretrainDataset(sequences, vocab, max_len=max_len)
    val_size = max(1, int(0.1 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
              pin_memory=False, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    # Model
    model = DNABERT2Model(vocab_size=len(vocab), max_len=max_len).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs, pct_start=0.1,
    )
    crit   = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history  = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss, n_batches = 0.0, 0
        for tokens, labels, att in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            att    = att.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(tokens, att)
                loss   = crit(logits.reshape(-1, len(vocab)),
                              labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(1, n_batches)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for tokens, labels, att in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                att    = att.to(device)
                logits = model(tokens, att)
                val_loss += crit(logits.reshape(-1, len(vocab)),
                                 labels.reshape(-1)).item()
                vn += 1
        avg_val = val_loss / max(1, vn)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"model": model.state_dict(), "vocab_size": len(vocab)},
                os.path.join(MODEL_DIR, "dnabert2_best.pt"),
            )

        print(f"  [DNABERT-2] Epoch {epoch:02d}/{epochs} | "
              f"loss={avg_train:.4f} | val={avg_val:.4f}")
        history.append({"epoch": epoch,
                        "loss": round(avg_train, 6),
                        "val_loss": round(avg_val, 6)})

    # Save final
    ckpt = os.path.join(MODEL_DIR, "dnabert2.pt")
    torch.save({"model": model.state_dict(), "vocab_size": len(vocab)}, ckpt)
    with open(os.path.join(MODEL_DIR, "dnabert2_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [DNABERT-2] Saved → {ckpt}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════════
def fill_masked_sequence(
    model:    "DNABERT2Model",
    sequence: str,
    vocab:    Dict[str, int],
    k:        int          = 6,
    device:   torch.device = None,
) -> Tuple[str, List[float]]:
    """Fill N regions using masked token prediction."""
    from preprocessing.encoding import encode_kmer_sequence

    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    inv_vocab = {v: k_str for k_str, v in vocab.items()}
    mask_id   = vocab["[MASK]"]
    cls_id    = vocab["[CLS]"]
    pad_id    = vocab["[PAD]"]
    max_len   = model.max_len

    seq         = list(sequence.upper())
    confidences = [1.0] * len(seq)
    chunk       = (max_len - 1) * k

    for start in range(0, len(seq), chunk):
        end        = min(start + chunk, len(seq))
        window_seq = "".join(seq[start:end])
        kmers      = encode_kmer_sequence(window_seq, vocab, k)

        tokens  = np.concatenate([[cls_id], kmers]).astype(np.int32)
        tokens  = tokens[:max_len]
        pad_len = max_len - len(tokens)
        tokens  = np.pad(tokens, (0, pad_len), constant_values=pad_id)

        for j in range(1, max_len - pad_len):
            kmer_start = start + (j - 1) * k
            kmer_end   = kmer_start + k
            if any(seq[p] == "N"
                   for p in range(kmer_start, min(kmer_end, len(seq)))):
                tokens[j] = mask_id

        t_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        att = torch.tensor(
            (tokens != pad_id).astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(t_tensor, att)
            probs  = torch.softmax(logits[0], dim=-1)

        for j in range(1, max_len - pad_len):
            if int(t_tensor[0, j].cpu()) != mask_id:
                continue
            top_id   = int(probs[j].argmax().cpu())
            top_kmer = inv_vocab.get(top_id, "AAAAAA")
            conf     = float(probs[j].max().cpu())

            kmer_start = start + (j - 1) * k
            for offset, base in enumerate(top_kmer):
                pos = kmer_start + offset
                if pos < len(seq) and seq[pos] == "N" and base in "ACGT":
                    seq[pos]         = base
                    confidences[pos] = round(conf, 4)

    return "".join(seq), confidences
