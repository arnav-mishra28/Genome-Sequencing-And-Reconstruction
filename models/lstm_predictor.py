"""
lstm_predictor.py
=================
Bidirectional LSTM for DNA sequence completion.
Fixed: deprecated AMP API, dynamic paths, Windows safety.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, MAX_SAMPLES,
    BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP,
)

VOCAB_SIZE  = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LEN     = 64
PREDICT_LEN = 8


class DNADataset(Dataset):
    def __init__(self, sequences: List[str], seq_len: int = SEQ_LEN,
                 pred_len: int = PREDICT_LEN, stride: int = 48,
                 max_samples: int = MAX_SAMPLES):
        from preprocessing.encoding import integer_encode
        self.X, self.Y = [], []

        for seq in sequences:
            if len(seq) < seq_len + pred_len:
                continue
            enc = integer_encode(seq)
            for i in range(0, len(enc) - seq_len - pred_len, stride):
                x = enc[i : i + seq_len].astype(np.int64)
                y = enc[i + seq_len : i + seq_len + pred_len]
                y = np.where(y == 4, np.random.randint(0, 4, y.shape),
                             y).astype(np.int64)
                self.X.append(x)
                self.Y.append(y)
                if len(self.X) >= max_samples:
                    break
            if len(self.X) >= max_samples:
                break

        if len(self.X) == 0:
            print("  [LSTM WARN] No valid sequences — synthetic fallback.")
            for _ in range(200):
                self.X.append(np.random.randint(0, 4, seq_len).astype(np.int64))
                self.Y.append(np.random.randint(0, 4, pred_len).astype(np.int64))

        print(f"  [LSTM] Dataset size: {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.long),
                torch.tensor(self.Y[idx], dtype=torch.long))


class BiLSTMPredictor(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, embed: int = EMBED_DIM,
                 hidden: int = HIDDEN_DIM, layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT, pred_len: int = PREDICT_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.vocab    = vocab

        self.embed = nn.Embedding(vocab, embed, padding_idx=4)
        self.lstm  = nn.LSTM(embed, hidden, layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0,
                             bidirectional=True)
        self.attn_w = nn.Linear(hidden * 2, 1)
        self.fc1    = nn.Linear(hidden * 2, hidden)
        self.fc2    = nn.Linear(hidden, vocab * pred_len)
        self.drop   = nn.Dropout(dropout)
        self.norm   = nn.LayerNorm(hidden * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        h, _ = self.lstm(e)
        h = self.norm(h)
        scores  = self.attn_w(h).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        ctx = (h * weights).sum(dim=1)
        out = self.drop(torch.relu(self.fc1(ctx)))
        out = self.fc2(out)
        return out.view(-1, self.pred_len, self.vocab)


def train_lstm(
    sequences:  List[str],
    epochs:     int   = 5,
    batch_size: int   = 16,
    lr:         float = 1e-3,
) -> "BiLSTMPredictor":

    device = DEVICE
    print(f"  [LSTM] Device: {device} | AMP: {USE_AMP}")

    dataset = DNADataset(sequences, max_samples=MAX_SAMPLES)
    val_size   = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
              pin_memory=False, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model = BiLSTMPredictor().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs, pct_start=0.3,
    )
    crit   = nn.CrossEntropyLoss(ignore_index=4)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total, n_batches = 0.0, 0, 0, 0

        for X, Y in train_loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if USE_AMP and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(X)
                    loss   = crit(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X)
                loss   = crit(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                opt.step()

            sched.step()
            total_loss += loss.item()
            n_batches  += 1

            preds = logits.argmax(-1)
            mask  = Y != 4
            if mask.any():
                correct += (preds[mask] == Y[mask]).sum().item()
                total   += mask.sum().item()

        avg_loss = total_loss / max(1, n_batches)
        acc      = correct / max(1, total)

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                logits   = model(X)
                val_loss += crit(logits.reshape(-1, VOCAB_SIZE),
                                 Y.reshape(-1)).item()
                vn += 1
        avg_val = val_loss / max(1, vn)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "lstm_best.pt"))

        print(f"  [LSTM] Epoch {epoch:02d}/{epochs} | "
              f"loss={avg_loss:.4f} | val={avg_val:.4f} | acc={acc:.4f}")
        history.append({"epoch": epoch, "loss": round(avg_loss, 6),
                        "val_loss": round(avg_val, 6),
                        "acc": round(acc, 6)})

    ckpt = os.path.join(MODEL_DIR, "lstm.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(MODEL_DIR, "lstm_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [LSTM] Saved → {ckpt}")
    return model


def predict_sequence(model: "BiLSTMPredictor", partial: str,
                     steps: int = 5,
                     device: torch.device = None) -> str:
    from preprocessing.encoding import integer_encode, decode_integer
    if device is None:
        device = DEVICE
    model.eval()
    model.to(device)
    seq = partial.upper()

    with torch.no_grad():
        for _ in range(steps):
            window = seq[-SEQ_LEN:]
            if len(window) < SEQ_LEN:
                window = window.ljust(SEQ_LEN, "N")
            enc = torch.tensor(
                integer_encode(window), dtype=torch.long,
            ).unsqueeze(0).to(device)
            logits = model(enc)
            pred   = logits[0].argmax(-1).cpu().numpy()
            new_bases = decode_integer(pred).replace("N", "A")
            seq += new_bases
    return seq