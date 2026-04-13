"""
denoising_autoencoder.py
========================
Convolutional Denoising Autoencoder for DNA base repair.
Enhanced with U-Net skip connections, multi-scale dilated convolutions,
and channel attention (SE-block).
"""

import os
import sys
import json
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, MAX_SAMPLES,
    BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP,
)

SEQ_LEN = 128
IN_CHAN  = 5
LATENT  = 64


# ── SE Block (Channel Attention) ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3,
                 dilation: int = 1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(in_c, out_c, k, padding=pad,
                              dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_c)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ── Model ─────────────────────────────────────────────────────────────────────
class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_chan: int = IN_CHAN, latent: int = LATENT,
                 seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len

        # Encoder with multi-scale dilations
        self.enc1 = ConvBlock(in_chan, 32, k=7)
        self.enc2 = ConvBlock(32, 64, k=5, dilation=2)
        self.enc3 = ConvBlock(64, latent, k=3)
        self.pool = nn.MaxPool1d(2, return_indices=True)

        # Channel attention
        self.se = SEBlock(latent)

        # Bottleneck attention
        self.attn = nn.MultiheadAttention(
            latent, num_heads=4, batch_first=True, dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(latent)

        # Decoder with U-Net skip connections
        self.unpool = nn.MaxUnpool1d(2)
        self.dec1 = ConvBlock(latent * 2, 64, k=3)  # *2 for skip
        self.dec2 = ConvBlock(64 + 64, 32, k=5)     # +skip from enc2
        self.dec3 = ConvBlock(32 + 32, 32, k=7)     # +skip from enc1
        self.out  = nn.Conv1d(32, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                            # (B, 32, L)
        e2 = self.enc2(e1)                           # (B, 64, L)
        e3 = self.enc3(e2)                           # (B, latent, L)

        # SE attention
        e3 = self.se(e3)

        # Pool
        e3_pooled, indices = self.pool(e3)           # (B, latent, L//2)

        # Bottleneck self-attention
        e3t = e3_pooled.permute(0, 2, 1)
        e3t_attn, _ = self.attn(e3t, e3t, e3t)
        e3t = self.attn_norm(e3t + e3t_attn)
        e3_pooled = e3t.permute(0, 2, 1)

        # Unpool
        d = self.unpool(e3_pooled, indices, output_size=e3.shape)

        # U-Net skip connections
        d = torch.cat([d, e3], dim=1)                # skip from enc3
        d = self.dec1(d)                              # (B, 64, L)
        d = torch.cat([d, e2], dim=1)                # skip from enc2
        d = self.dec2(d)                              # (B, 32, L)
        d = torch.cat([d, e1], dim=1)                # skip from enc1
        d = self.dec3(d)                              # (B, 32, L)

        # Match input length
        if d.shape[-1] > x.shape[-1]:
            d = d[:, :, :x.shape[-1]]
        elif d.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - d.shape[-1]
            d = torch.nn.functional.pad(d, (0, pad))

        return self.out(d)


# ── Training ──────────────────────────────────────────────────────────────────
def train_autoencoder(
    clean_seqs: List[str],
    noisy_seqs: List[str],
    epochs:     int   = 6,
    batch_size: int   = BATCH_SIZE,
    lr:         float = 1e-3,
) -> "DenoisingAutoencoder":
    from data.dataset_builder import CorruptionDataset

    device = DEVICE
    print(f"  [AE] Device: {device} | AMP: {USE_AMP}")

    full_ds  = CorruptionDataset(clean_seqs, noisy_seqs,
                                 max_samples=MAX_SAMPLES)
    val_size = max(1, int(0.1 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
              pin_memory=False, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model  = DenoisingAutoencoder().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5,
    )
    crit   = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history  = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            targets = clean[:, :4, :].argmax(dim=1)
            opt.zero_grad(set_to_none=True)

            if USE_AMP and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(noisy)
                    loss   = crit(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(noisy)
                loss   = crit(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(1, n_batches)

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy   = noisy.to(device)
                clean   = clean.to(device)
                targets = clean[:, :4, :].argmax(dim=1)
                logits  = model(noisy)
                val_loss += crit(logits, targets).item()
                vn += 1
        avg_val = val_loss / max(1, vn)
        sched.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "denoising_ae_best.pt"))

        print(f"  [AE] Epoch {epoch:02d}/{epochs} | "
              f"loss={avg_train:.4f} | val={avg_val:.4f}")
        history.append({"epoch": epoch,
                        "loss": round(avg_train, 6),
                        "val_loss": round(avg_val, 6)})

    ckpt = os.path.join(MODEL_DIR, "denoising_ae.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(MODEL_DIR, "ae_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [AE] Saved → {ckpt}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def denoise_sequence(
    model:    "DenoisingAutoencoder",
    sequence: str,
    device:   torch.device = None,
) -> Tuple[str, List[float], List[Dict]]:
    from preprocessing.encoding import one_hot_encode

    if device is None:
        device = DEVICE
    model.eval()
    model.to(device)

    INT2BASE   = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
    results, confs, repair_log = [], [], []

    for chunk_start in range(0, len(sequence), SEQ_LEN):
        chunk  = sequence[chunk_start : chunk_start + SEQ_LEN]
        padded = chunk.ljust(SEQ_LEN, "N").upper()

        oh = one_hot_encode(padded)
        enc5 = np.zeros((SEQ_LEN, 5), dtype=np.float32)
        enc5[:, :4] = oh
        chars      = np.array(list(padded), dtype=str)
        enc5[:, 4] = (chars == "N").astype(np.float32)

        t = torch.from_numpy(enc5.T.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)
            probs  = torch.softmax(logits[0], dim=0)

        pred = probs[:4].argmax(dim=0).cpu().numpy()
        conf = probs[:4].max(dim=0).values.cpu().numpy()

        for i, (orig, p, c) in enumerate(zip(chunk, pred, conf)):
            base_p     = INT2BASE[int(p)]
            global_pos = chunk_start + i
            if orig == "N" or orig != base_p:
                repair_log.append({
                    "global_pos": global_pos,
                    "original":   orig,
                    "repaired":   base_p,
                    "confidence": round(float(c), 4),
                    "action":     "gap_fill" if orig == "N"
                                  else "base_correction",
                })
            results.append(base_p)
            confs.append(float(c))

    return (
        "".join(results[:len(sequence)]),
        confs[:len(sequence)],
        repair_log,
    )