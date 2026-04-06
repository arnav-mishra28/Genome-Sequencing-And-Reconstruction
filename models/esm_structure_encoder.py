"""
esm_structure_encoder.py
========================
ESM (Evolutionary Scale Modeling) inspired structure-aware genomic encoder.

Key ideas from Meta AI's ESM / ESM-2:
  - Deep transformer encoder learning evolutionary features from sequence alone
  - Contact prediction head for base-pair interaction modelling
  - Per-residue confidence scoring (analogous to pLDDT)
  - No MSA required — standalone single-sequence inference

Reference:
  - ESM-2: Evolutionary Scale Modeling (Lin et al., Science 2023)
"""

import os
import sys
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN,
    DROPOUT, DEVICE,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rotary Positional Embedding (RoPE)  — used in ESM-2
# ═══════════════════════════════════════════════════════════════════════════════
class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021) used in ESM-2."""
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L = x.shape[1]
        t = torch.arange(L, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, D/2)
        cos_emb = freqs.cos().unsqueeze(0)  # (1, L, D/2)
        sin_emb = freqs.sin().unsqueeze(0)
        return cos_emb, sin_emb


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to query/key tensors."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ESM-style Attention Block
# ═══════════════════════════════════════════════════════════════════════════════
class ESMAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, 3 * dim)
        self.proj   = nn.Linear(dim, dim)
        self.drop1  = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        # Pre-norm
        h = self.norm1(x)

        # QKV projection
        qkv = self.qkv(h).reshape(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, hd)

        # Apply RoPE
        cos, sin = self.rope(x)
        cos = cos.unsqueeze(1)  # (1, 1, L, D/2)
        sin = sin.unsqueeze(1)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(hd)
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # (B, H, L, L)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out   = (attn_probs @ v).transpose(1, 2).reshape(B, L, D)
        h = self.drop1(self.proj(attn_out))

        x = x + h
        x = x + self.ffn(self.norm2(x))

        return x, attn_probs  # return attention weights for contact prediction


# ═══════════════════════════════════════════════════════════════════════════════
#  Contact Prediction Head
# ═══════════════════════════════════════════════════════════════════════════════
class ContactPredictionHead(nn.Module):
    """
    Predicts base-pair contacts from attention weights.
    Inspired by ESM's contact prediction approach where attention
    matrices correlate with structural contacts.
    """
    def __init__(self, n_layers: int, n_heads: int):
        super().__init__()
        self.regression = nn.Sequential(
            nn.Linear(n_layers * n_heads, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, all_attn_weights: torch.Tensor) -> torch.Tensor:
        """
        all_attn_weights: (B, n_layers, n_heads, L, L)
        Returns: (B, L, L) contact probabilities
        """
        B, nL, nH, L, _ = all_attn_weights.shape
        # Symmetrise: (A_{ij} + A_{ji}) / 2
        attn = (all_attn_weights + all_attn_weights.transpose(-2, -1)) / 2
        # Stack layers and heads → (B, L, L, nL*nH)
        attn = attn.permute(0, 3, 4, 1, 2).reshape(B, L, L, nL * nH)
        contacts = self.regression(attn).squeeze(-1)  # (B, L, L)
        return torch.sigmoid(contacts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Residue Confidence Head  (pLDDT-like)
# ═══════════════════════════════════════════════════════════════════════════════
class ConfidenceHead(nn.Module):
    """Per-position confidence prediction, analogous to ESM's pLDDT."""
    def __init__(self, dim: int, n_bins: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, n_bins),
        )
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) hidden states
        Returns: (B, L) confidence scores in [0, 1]
        """
        logits = self.net(x)  # (B, L, n_bins)
        probs  = F.softmax(logits, dim=-1)
        # Expected value as confidence
        bins = torch.linspace(0, 1, self.n_bins, device=x.device)
        return (probs * bins.unsqueeze(0).unsqueeze(0)).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ESM Genomic Encoder  (full model)
# ═══════════════════════════════════════════════════════════════════════════════
class ESMGenomicEncoder(nn.Module):
    """
    ESM-inspired encoder adapted for genomic (DNA) sequences.

    Outputs:
      - hidden_states:  (B, L, D)    sequence representations
      - contacts:       (B, L, L)    predicted base-pair contacts
      - confidence:     (B, L)       per-position confidence scores
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = EMBED_DIM,
        n_heads:    int = N_HEADS,
        n_layers:   int = N_LAYERS,
        ffn_dim:    int = FFN_DIM,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop  = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ESMAttentionBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm    = nn.LayerNorm(embed_dim)
        self.contact_head  = ContactPredictionHead(n_layers, n_heads)
        self.confidence    = ConfidenceHead(embed_dim)

        # Regression head for DNA reconstruction
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(
        self,
        tokens:         torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        Returns dict with keys:
          logits, hidden_states, contacts, confidence, all_attn
        """
        B, L = tokens.shape
        x = self.embed_drop(self.token_embed(tokens))

        key_pad = (attention_mask == 0) if attention_mask is not None else None

        all_attn = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask=key_pad)
            all_attn.append(attn_weights)

        x = self.final_norm(x)

        # Stack attention for contact prediction
        all_attn_tensor = torch.stack(all_attn, dim=1)  # (B, nL, nH, L, L)
        contacts        = self.contact_head(all_attn_tensor)
        conf            = self.confidence(x)

        logits = self.reconstruction_head(x)

        return {
            "logits":        logits,
            "hidden_states": x,
            "contacts":      contacts,
            "confidence":    conf,
            "all_attn":      all_attn_tensor,
        }
