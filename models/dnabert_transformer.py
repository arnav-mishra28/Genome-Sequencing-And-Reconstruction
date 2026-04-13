"""
dnabert_transformer.py
======================
Transformer (BERT-style) for masked DNA prediction (k-mer tokenization).
Inspired by DNABERT / Genomic BERT.

FIXES:
  - Added all typing imports at TOP of file (Tuple, List, Dict)
  - Removed duplicate/dangling import at bottom
  - num_workers=0 for Windows DataLoader safety
  - Dataset size capped to prevent memory freeze
  - Smaller default hyperparameters for CPU safety
"""

# ── Standard library ───────────────────────────────────────────────────────────
import os
import json
import random
from typing import List, Dict, Tuple
from xml.parsers.expat import model   # ← ALL typing imports at the top

# ── Third party ────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Project ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters (CPU-safe defaults) ────────────────────────────────────────
K           = 6
MAX_LEN     = 128    # reduced from 256
EMBED_DIM   = 64     # reduced from 128
N_HEADS     = 4      # reduced from 8
N_LAYERS    = 3      # reduced from 6
FFN_DIM     = 256    # reduced from 512
DROPOUT     = 0.1
MASK_PROB   = 0.15
MAX_SAMPLES = 3000   # hard cap on dataset size


# ── Dataset ────────────────────────────────────────────────────────────────────
class KmerDataset(Dataset):
    def __init__(
        self,
        sequences:   List[str],
        vocab:       Dict[str, int],
        k:           int   = K,
        max_len:     int   = MAX_LEN,
        mask_prob:   float = MASK_PROB,
        max_samples: int   = MAX_SAMPLES,
    ):
        from preprocessing.encoding import encode_kmer_sequence

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        pad_id  = vocab["[PAD]"]
        mask_id = vocab["[MASK]"]
        cls_id  = vocab["[CLS]"]

        for seq in sequences:
            if len(self.samples) >= max_samples:
                break
            if len(seq) < k + 2:
                continue

            ids   = encode_kmer_sequence(seq, vocab, k)
            chunk = max_len - 1   # leave room for [CLS]

            for i in range(0, len(ids), chunk):
                if len(self.samples) >= max_samples:
                    break

                window = ids[i : i + chunk]
                tokens = np.concatenate([[cls_id], window]).astype(np.int32)
                tokens = tokens[:max_len]

                # Pad to max_len
                pad_len = max_len - len(tokens)
                tokens  = np.pad(tokens, (0, pad_len), constant_values=pad_id)

                # MLM masking
                labels   = np.full(max_len, -100, dtype=np.int32)
                seq_len  = max_len - pad_len   # actual non-pad length

                for j in range(1, seq_len):    # skip [CLS] at position 0
                    if random.random() < mask_prob:
                        labels[j] = int(tokens[j])
                        r = random.random()
                        if r < 0.80:
                            tokens[j] = mask_id
                        elif r < 0.90:
                            tokens[j] = random.randint(5, max(5, len(vocab) - 1))
                        # else keep original (10%)

                att_mask = (tokens != pad_id).astype(np.float32)

                self.samples.append((
                    tokens.copy(),
                    labels.copy(),
                    att_mask.copy(),
                ))

        if len(self.samples) == 0:
            print("  [DNABERT WARN] No samples built — using synthetic fallback.")
            for _ in range(100):
                t = np.random.randint(0, min(len(vocab), 100), max_len).astype(np.int32)
                l = np.full(max_len, -100, dtype=np.int32)
                a = np.ones(max_len, dtype=np.float32)
                self.samples.append((t, l, a))

        print(f"  [DNABERT] Dataset size: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, l, a = self.samples[idx]
        return (
            torch.tensor(t, dtype=torch.long),
            torch.tensor(l, dtype=torch.long),
            torch.tensor(a, dtype=torch.float),
        )


# ── Model ──────────────────────────────────────────────────────────────────────
class DNABertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int   = EMBED_DIM,
        n_heads:    int   = N_HEADS,
        n_layers:   int   = N_LAYERS,
        ffn_dim:    int   = FFN_DIM,
        max_len:    int   = MAX_LEN,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_len, embed_dim)
        self.drop        = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers      = n_layers,
            enable_nested_tensor = False,   # ← silences the UserWarning
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(
        self,
        tokens:         torch.Tensor,          # (B, L)
        attention_mask: torch.Tensor = None,   # (B, L)  1=real, 0=pad
    ) -> torch.Tensor:                         # (B, L, vocab_size)
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x    = self.drop(self.token_embed(tokens) + self.pos_embed(pos))

        key_pad = None
        if attention_mask is not None:
            key_pad = (attention_mask == 0)   # True = ignore

        x = self.encoder(x, src_key_padding_mask=key_pad)
        return self.mlm_head(x)               # (B, L, V)


# ── Training ───────────────────────────────────────────────────────────────────
def train_dnabert(
    sequences:  List[str],
    vocab:      Dict[str, int],
    epochs:     int   = 4,
    batch_size: int   = 16,
    lr:         float = 2e-4,
) -> "DNABertModel":

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"  [DNABERT] Device: {device} | AMP: {use_amp}")

    # ── Build dataset ──────────────────────────────────────────────────────────
    full_ds  = KmerDataset(sequences, vocab, max_samples=MAX_SAMPLES)
    val_size = max(1, int(0.1 * len(full_ds)))
    trn_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [trn_size, val_size])

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = 0,       # ← Windows: must be 0
        pin_memory  = False,
        drop_last   = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = DNABertModel(vocab_size=len(vocab)).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr          = lr,
        steps_per_epoch = max(1, len(train_loader)),
        epochs          = epochs,
        pct_start       = 0.1,
    )
    crit   = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history      = []
    best_val     = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for tokens, labels, att in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            att    = att.to(device,    non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device.type, enabled=USE_AMP):
                    logits = model(tokens, att)
                    loss = crit(logits.reshape(-1, len(vocab)), labels.reshape(-1))
            else:
                logits = model(tokens, att)
                loss = crit(logits.reshape(-1, len(vocab)), labels.reshape(-1))

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                if not loss.requires_grad:
                    raise RuntimeError("Loss is not connected to graph!")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            sched.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(1, n_batches)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        vn       = 0
        with torch.no_grad():
            for tokens, labels, att in val_loader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                att    = att.to(device)
                logits = model(tokens, att)
                val_loss += crit(logits.reshape(-1, len(vocab)), labels.reshape(-1)).item()
                vn += 1
        avg_val = val_loss / max(1, vn)

        # Save best checkpoint
        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {"model": model.state_dict(), "vocab_size": len(vocab)},
                os.path.join(MODEL_DIR, "dnabert_best.pt"),
            )

        print(
            f"  [DNABERT] Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_train:.4f} | val={avg_val:.4f}"
        )
        history.append({
            "epoch":    epoch,
            "loss":     round(avg_train, 6),
            "val_loss": round(avg_val,   6),
        })

    # ── Save final ─────────────────────────────────────────────────────────────
    ckpt = os.path.join(MODEL_DIR, "dnabert.pt")
    torch.save({"model": model.state_dict(), "vocab_size": len(vocab)}, ckpt)
    with open(os.path.join(MODEL_DIR, "dnabert_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [DNABERT] Saved → {ckpt}")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
def fill_masked_sequence(
    model:    "DNABertModel",
    sequence: str,
    vocab:    Dict[str, int],
    k:        int          = K,
    device:   torch.device = None,
) -> Tuple[str, List[float]]:
    """
    Fill N regions in a sequence using masked token prediction.
    Returns (reconstructed_sequence, per_position_confidence).
    """
    from preprocessing.encoding import encode_kmer_sequence

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    inv_vocab = {v: k_str for k_str, v in vocab.items()}
    mask_id   = vocab["[MASK]"]
    cls_id    = vocab["[CLS]"]
    pad_id    = vocab["[PAD]"]

    seq         = list(sequence.upper())
    confidences = [1.0] * len(seq)
    chunk       = (MAX_LEN - 1) * k   # characters per chunk

    for start in range(0, len(seq), chunk):
        end         = min(start + chunk, len(seq))
        window_seq  = "".join(seq[start:end])
        kmers       = encode_kmer_sequence(window_seq, vocab, k)

        tokens = np.concatenate([[cls_id], kmers]).astype(np.int32)
        tokens = tokens[:MAX_LEN]
        pad_len = MAX_LEN - len(tokens)
        tokens  = np.pad(tokens, (0, pad_len), constant_values=pad_id)

        # Mask k-mers that overlap any N position
        for j in range(1, MAX_LEN - pad_len):
            kmer_start = start + (j - 1) * k
            kmer_end   = kmer_start + k
            if any(
                seq[p] == "N"
                for p in range(kmer_start, min(kmer_end, len(seq)))
            ):
                tokens[j] = mask_id

        t_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        att      = torch.tensor(
            (tokens != pad_id).astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(t_tensor, att)              # (1, L, V)
            probs  = torch.softmax(logits[0], dim=-1)  # (L, V)

        # Decode masked positions back to bases
        for j in range(1, MAX_LEN - pad_len):
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