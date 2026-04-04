"""
denoising_autoencoder.py
========================
Convolutional Denoising Autoencoder for DNA base repair.

FIXES:
  - AttributeError: 'bool' object has no attribute 'astype'
    (comparison result is Python bool, not numpy array — fixed with np.array())
  - num_workers=0 for Windows
  - Dataset size capped
  - Updated torch.amp API (no more torch.cuda.amp.*)
  - AMP properly disabled on CPU
"""

import os
import json
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN     = 128    # reduced from 256
IN_CHAN     = 5      # A C G T N channels
LATENT      = 64     # reduced from 128
MAX_SAMPLES = 2000   # hard cap


# ── Dataset ────────────────────────────────────────────────────────────────────
class DenoisingDataset(Dataset):
    def __init__(
        self,
        clean_seqs: List[str],
        noisy_seqs: List[str],
        seq_len:    int = SEQ_LEN,
        max_samples: int = MAX_SAMPLES,
    ):
        from preprocessing.encoding import one_hot_encode
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for c, n in zip(clean_seqs, noisy_seqs):
            if len(self.pairs) >= max_samples:
                break

            # ── Clean sequence encoding ────────────────────────────────────────
            c_str  = (c[:seq_len]).ljust(seq_len, "A").upper()
            cl_oh  = one_hot_encode(c_str)          # (L, 4)  float32

            cl5 = np.zeros((seq_len, 5), dtype=np.float32)
            cl5[:, :4] = cl_oh                      # channels 0-3 = ACGT
            # channel 4 = N indicator → 0 for clean

            # ── Noisy sequence encoding ────────────────────────────────────────
            n_str  = (n[:seq_len]).ljust(seq_len, "N").upper()
            no_oh  = one_hot_encode(n_str)          # (L, 4)  float32

            no5 = np.zeros((seq_len, 5), dtype=np.float32)
            no5[:, :4] = no_oh

            # ── N-position indicator (channel 4) ──────────────────────────────
            # FIXED: convert character array comparison to float array properly
            n_chars   = np.array(list(n_str), dtype=str)   # shape (L,)
            n_mask    = (n_chars == "N").astype(np.float32) # shape (L,)  ← FIXED
            no5[:, 4] = n_mask

            # channels-first: (5, L)
            self.pairs.append((
                torch.from_numpy(no5.T.copy()),   # noisy  (5, L)
                torch.from_numpy(cl5.T.copy()),   # clean  (5, L)
            ))

        if len(self.pairs) == 0:
            print("  [AE WARN] No valid pairs — using synthetic fallback.")
            for _ in range(100):
                noisy = torch.rand(5, seq_len)
                clean = torch.rand(5, seq_len)
                self.pairs.append((noisy, clean))

        print(f"  [AE] Dataset size: {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


# ── Building blocks ────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(in_c, out_c, k, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_c)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ── Model ──────────────────────────────────────────────────────────────────────
class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        in_chan: int = IN_CHAN,
        latent:  int = LATENT,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len = seq_len

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_chan, 32, k=7)
        self.enc2 = ConvBlock(32,      64, k=5, dilation=2)
        self.enc3 = ConvBlock(64,      latent, k=3)
        self.pool = nn.MaxPool1d(2, return_indices=True)

        # ── Bottleneck attention ───────────────────────────────────────────────
        self.attn = nn.MultiheadAttention(
            latent, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(latent)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.unpool = nn.MaxUnpool1d(2)
        self.dec1   = ConvBlock(latent, 64, k=3)
        self.dec2   = ConvBlock(64,     32, k=5)
        self.dec3   = ConvBlock(32,     32, k=7)
        self.out    = nn.Conv1d(32, 5, kernel_size=1)   # 5 output channels → ACGTN logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, L)
        e1 = self.enc1(x)                              # (B, 32, L)
        e2 = self.enc2(e1)                             # (B, 64, L)
        e3 = self.enc3(e2)                             # (B, latent, L)

        # Pool
        e3_pooled, indices = self.pool(e3)             # (B, latent, L//2)

        # Attention over sequence dimension
        e3t = e3_pooled.permute(0, 2, 1)              # (B, L//2, latent)
        e3t_attn, _ = self.attn(e3t, e3t, e3t)
        e3t = self.attn_norm(e3t + e3t_attn)          # residual
        e3_pooled = e3t.permute(0, 2, 1)              # (B, latent, L//2)

        # Unpool
        d = self.unpool(e3_pooled, indices, output_size=e3.shape)  # (B, latent, L)

        d = self.dec1(d)                               # (B, 64, L)
        d = self.dec2(d)                               # (B, 32, L)
        d = self.dec3(d)                               # (B, 32, L)

        # Crop / pad to match input length exactly
        if d.shape[-1] > x.shape[-1]:
            d = d[:, :, : x.shape[-1]]
        elif d.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - d.shape[-1]
            d = torch.nn.functional.pad(d, (0, pad))

        return self.out(d)                             # (B, 5, L)


# ── Training ───────────────────────────────────────────────────────────────────
def train_autoencoder(
    clean_seqs: List[str],
    noisy_seqs: List[str],
    epochs:     int   = 6,
    batch_size: int   = 16,
    lr:         float = 1e-3,
) -> "DenoisingAutoencoder":

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"  [AE] Device: {device} | AMP: {use_amp}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_ds  = DenoisingDataset(clean_seqs, noisy_seqs, max_samples=MAX_SAMPLES)
    val_size = max(1, int(0.1 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = 0,      # Windows: must be 0
        pin_memory  = False,
        drop_last   = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model + optimizer ──────────────────────────────────────────────────────
    model  = DenoisingAutoencoder().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5
    )
    crit   = nn.CrossEntropyLoss()

    # ── Updated AMP API (no deprecation warnings) ──────────────────────────────
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    history  = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)   # (B, 5, L)
            clean = clean.to(device, non_blocking=True)   # (B, 5, L)

            # Target = argmax of first 4 channels of clean (ACGT, not N)
            targets = clean[:, :4, :].argmax(dim=1)       # (B, L)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(noisy)                  # (B, 5, L)
                    loss   = crit(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(noisy)                      # (B, 5, L)
                loss   = crit(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(1, n_batches)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        vn       = 0
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
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, "denoising_ae_best.pt"),
            )

        print(
            f"  [AE] Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_train:.4f} | val={avg_val:.4f}"
        )
        history.append({
            "epoch":    epoch,
            "loss":     round(avg_train, 6),
            "val_loss": round(avg_val,   6),
        })

    # ── Save final ─────────────────────────────────────────────────────────────
    ckpt = os.path.join(MODEL_DIR, "denoising_ae.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(MODEL_DIR, "ae_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [AE] Saved → {ckpt}")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def denoise_sequence(
    model:    "DenoisingAutoencoder",
    sequence: str,
    device:   torch.device = None,
) -> Tuple[str, List[float], List[Dict]]:
    """
    Denoise a damaged sequence.
    Returns: (reconstructed_seq, per_base_confidence, repair_log)
    """
    from preprocessing.encoding import one_hot_encode

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    INT2BASE   = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
    results    = []
    confs      = []
    repair_log = []

    for chunk_start in range(0, len(sequence), SEQ_LEN):
        chunk  = sequence[chunk_start : chunk_start + SEQ_LEN]
        padded = chunk.ljust(SEQ_LEN, "N").upper()

        # One-hot encode
        oh = one_hot_encode(padded)             # (L, 4)

        enc5 = np.zeros((SEQ_LEN, 5), dtype=np.float32)
        enc5[:, :4] = oh

        # N-channel — FIXED same way as dataset
        chars        = np.array(list(padded), dtype=str)
        enc5[:, 4]   = (chars == "N").astype(np.float32)

        t = torch.from_numpy(enc5.T.copy()).unsqueeze(0).to(device)  # (1, 5, L)

        with torch.no_grad():
            logits = model(t)                            # (1, 5, L)
            probs  = torch.softmax(logits[0], dim=0)    # (5, L)

        # Argmax over first 4 channels only (ignore N channel in output)
        pred = probs[:4].argmax(dim=0).cpu().numpy()    # (L,)
        conf = probs[:4].max(dim=0).values.cpu().numpy()  # (L,)

        for i, (orig, p, c) in enumerate(zip(chunk, pred, conf)):
            base_p = INT2BASE[int(p)]
            global_pos = chunk_start + i

            if orig == "N" or orig != base_p:
                repair_log.append({
                    "global_pos": global_pos,
                    "original":   orig,
                    "repaired":   base_p,
                    "confidence": round(float(c), 4),
                    "action":     "gap_fill" if orig == "N" else "base_correction",
                })

            results.append(base_p)
            confs.append(float(c))

    return (
        "".join(results[: len(sequence)]),
        confs[: len(sequence)],
        repair_log,
    )