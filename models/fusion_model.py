"""
fusion_model.py
===============
Transformer + Phylogenetic GNN Fusion Model with Cross-Attention.

Architecture:
  1. GNN Encoder: Processes species nodes on the phylogenetic graph →
     produces species-level embeddings that encode evolutionary context.
  2. Transformer Encoder: Processes DNA k-mer tokens with ALiBi attention.
  3. Cross-Attention Fusion: At every transformer layer, GNN species
     embeddings are injected via cross-attention (GNN = keys/values,
     sequence = queries). This lets the model reason about
     "what base is biologically plausible given evolutionary context".
  4. Three output heads:
     - MLM head (masked token prediction)
     - Reconstruction head (per-position base prediction)
     - Confidence head (per-position confidence + overall reliability)

This is the main intelligence engine that combines:
  - DNABERT-2 style sequence understanding
  - Phylogenetic GNN evolutionary reasoning
  - Calibrated confidence output
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MODEL_DIR, DEVICE, USE_AMP, EMBED_DIM, N_HEADS, N_LAYERS,
    FFN_DIM, MAX_SEQ_LEN, DROPOUT, MASK_PROB, MAX_SAMPLES,
    BATCH_SIZE, NUM_WORKERS, GRADIENT_CLIP, PHYLO_DISTANCES,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALiBi (shared with DNABERT-2)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def _pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_pow2(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = _pow2(closest)
    extra = _pow2(2 * closest)
    slopes += extra[0::2][: n_heads - closest]
    return torch.tensor(slopes)


def build_alibi_bias(n_heads: int, max_len: int) -> torch.Tensor:
    slopes = _get_alibi_slopes(n_heads)
    pos = torch.arange(max_len)
    rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).float().unsqueeze(0)
    return slopes.unsqueeze(1).unsqueeze(2) * rel


# ═══════════════════════════════════════════════════════════════════════════════
#  GEGLU Feed-Forward
# ═══════════════════════════════════════════════════════════════════════════════
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class GEGLUFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN Encoder (species-level phylogenetic reasoning)
# ═══════════════════════════════════════════════════════════════════════════════
class PhyloGNNEncoder(nn.Module):
    """
    Graph Neural Network that encodes species nodes on a phylogenetic tree.
    Outputs species embeddings that capture evolutionary relationships.
    """
    def __init__(self, node_feat_dim: int, hidden: int = EMBED_DIM,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GNNBlock(hidden, hidden, dropout))

        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N_species, feat_dim) — k-mer frequency vectors
            adj: (N_species, N_species) — phylogenetic adjacency matrix
        Returns:
            (N_species, hidden) — species embeddings
        """
        h = self.input_proj(x)
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return self.output_norm(h)


class GNNBlock(nn.Module):
    """GCN layer with residual + LayerNorm + gating."""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, adj):
        h = self.norm(x)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
        agg = (adj @ h) / deg

        h_self = self.W_self(h)
        h_neigh = self.W_neigh(agg)

        # Gated fusion
        gate = self.gate(torch.cat([h_self, h_neigh], dim=-1))
        out = gate * h_self + (1 - gate) * h_neigh
        out = self.act(self.drop(out))

        # Residual
        if x.shape == out.shape:
            out = out + x
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-Attention: GNN → Transformer injection
# ═══════════════════════════════════════════════════════════════════════════════
class CrossAttentionLayer(nn.Module):
    """
    Cross-attention that injects phylogenetic GNN embeddings into the
    sequence representation.
    Query: sequence tokens  (B, L, D)
    Key/Value: species embeddings  (1, N_species, D) → broadcast to batch
    """
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.gate_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq: torch.Tensor,
                species_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq:         (B, L, D) — sequence hidden states
            species_emb: (B, N_species, D) — GNN species embeddings
        Returns:
            (B, L, D) — seq with phylogenetic info injected
        """
        q = self.norm_q(seq)
        kv = self.norm_kv(species_emb)
        cross_out, _ = self.cross_attn(q, kv, kv)
        # Gated residual — start with small gate so GNN doesn't dominate
        gate = torch.sigmoid(self.gate_param)
        return seq + gate * self.drop(cross_out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fused Transformer Block (Self-Attention + Cross-Attention + FFN)
# ═══════════════════════════════════════════════════════════════════════════════
class FusedTransformerBlock(nn.Module):
    """
    Pre-Norm transformer block with:
      1. Self-attention (ALiBi) on sequence
      2. Cross-attention with GNN species embeddings
      3. GEGLU feed-forward
    """
    def __init__(self, dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # Self-attention (sequence)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        # Cross-attention (GNN injection)
        self.cross_attn = CrossAttentionLayer(dim, n_heads, dropout)

        # Feed-forward
        self.ffn = GEGLUFeedForward(dim, ffn_dim, dropout)

    def forward(self, x, alibi_bias=None, key_padding_mask=None,
                species_emb=None):
        # 1. Self-attention with ALiBi
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h,
                               attn_mask=alibi_bias,
                               key_padding_mask=key_padding_mask)
        x = x + self.drop1(h)

        # 2. Cross-attention with GNN (if species embeddings provided)
        if species_emb is not None:
            x = self.cross_attn(x, species_emb)

        # 3. Feed-forward
        x = x + self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence Head
# ═══════════════════════════════════════════════════════════════════════════════
class ConfidenceHead(nn.Module):
    """
    Per-position confidence prediction + overall reliability score.

    Outputs:
      - per_base_confidence: (B, L) — sigmoid scores in [0, 1]
      - reliability_score:   (B,)   — overall sequence reliability
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        # Per-position confidence
        self.per_pos = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        # Sequence-level reliability (attention pooling → score)
        self.pool_attn = nn.Linear(dim, 1)
        self.reliability = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def forward(self, h: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Args:
            h: (B, L, D) — fused hidden states
            attention_mask: (B, L) — 1=valid, 0=pad
        Returns:
            per_base_conf: (B, L) — per-position confidence
            reliability:   (B,)   — overall reliability
            temperature:   scalar — current temperature
        """
        # Per-position confidence
        per_base_conf = torch.sigmoid(
            self.per_pos(h).squeeze(-1) / self.temperature.clamp(min=0.1)
        )

        # Attention-pooled representation for reliability
        attn_scores = self.pool_attn(h).squeeze(-1)  # (B, L)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0, float("-inf")
            )
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, L)
        pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        reliability = self.reliability(pooled).squeeze(-1)  # (B,)

        return per_base_conf, reliability, self.temperature


# ═══════════════════════════════════════════════════════════════════════════════
#  TransformerGNNFusion — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class TransformerGNNFusion(nn.Module):
    """
    Unified Transformer + Phylogenetic GNN model with confidence scoring.

    Forward pass:
      1. GNN encodes species nodes → species_embeddings (N_species, D)
      2. Token embedding → (B, L, D)
      3. For each transformer layer:
         a. Self-attention with ALiBi on tokens
         b. Cross-attention: inject GNN species context
         c. GEGLU feed-forward
      4. Three output heads:
         - MLM: (B, L, vocab_size) — masked token logits
         - Reconstruction: (B, L, 5) — per-position base prediction (ACGTN)
         - Confidence: per-base (B, L) + reliability (B,)
    """
    def __init__(
        self,
        vocab_size:     int,
        n_species_feat: int   = 256,  # 4^4 = 256 for 4-mer frequencies
        embed_dim:      int   = EMBED_DIM,
        n_heads:        int   = N_HEADS,
        n_layers:       int   = N_LAYERS,
        ffn_dim:        int   = FFN_DIM,
        max_len:        int   = MAX_SEQ_LEN,
        dropout:        float = DROPOUT,
        n_bases:        int   = 5,    # ACGTN
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.max_len   = max_len
        self.vocab_size = vocab_size

        # ── GNN Encoder ──────────────────────────────────────────────────────
        self.gnn_encoder = PhyloGNNEncoder(
            node_feat_dim=n_species_feat,
            hidden=embed_dim,
            n_layers=3,
            dropout=dropout,
        )

        # ── Token Embedding (no positional — ALiBi instead) ──────────────────
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)

        # ── Fused Transformer Blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FusedTransformerBlock(embed_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # ── ALiBi bias ───────────────────────────────────────────────────────
        alibi = build_alibi_bias(n_heads, max_len)
        self.register_buffer("alibi_bias", alibi, persistent=False)

        # ── Output Heads ─────────────────────────────────────────────────────
        # MLM head (masked token prediction)
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        # Reconstruction head (per-position base prediction: ACGTN)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, n_bases),
        )

        # Confidence head
        self.confidence_head = ConfidenceHead(embed_dim, dropout)

    def encode_species(self, species_feats: torch.Tensor,
                       adj: torch.Tensor) -> torch.Tensor:
        """
        Encode species with GNN.
        Args:
            species_feats: (N_species, feat_dim)
            adj:           (N_species, N_species)
        Returns:
            (N_species, embed_dim)
        """
        return self.gnn_encoder(species_feats, adj)

    def forward(
        self,
        tokens:          torch.Tensor,         # (B, L)
        attention_mask:  torch.Tensor = None,   # (B, L)
        species_feats:   torch.Tensor = None,   # (N_species, feat_dim)
        adjacency:       torch.Tensor = None,   # (N_species, N_species)
        species_idx:     torch.Tensor = None,   # (B,) — which species each sample belongs to
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape
        device = tokens.device

        # ── 1. GNN encoding ──────────────────────────────────────────────────
        species_emb = None
        if species_feats is not None and adjacency is not None:
            all_species_emb = self.gnn_encoder(species_feats, adjacency)
            # (N_species, D)

            if species_idx is not None:
                # Select per-sample species embedding and broadcast
                # species_idx: (B,) → gather (B, D) → (B, 1, D) → (B, L, D)
                per_sample = all_species_emb[species_idx]  # (B, D)
                # Also provide all species as context for cross-attention
                species_emb = all_species_emb.unsqueeze(0).expand(
                    B, -1, -1
                )  # (B, N_species, D)
            else:
                species_emb = all_species_emb.unsqueeze(0).expand(
                    B, -1, -1
                )

        # ── 2. Token embedding ────────────────────────────────────────────────
        x = self.embed_drop(self.token_embed(tokens))

        # ── 3. ALiBi bias ─────────────────────────────────────────────────────
        alibi = self.alibi_bias[:, :L, :L]
        alibi = alibi.unsqueeze(0).expand(B, -1, -1, -1)
        alibi = alibi.reshape(B * self.n_heads, L, L)

        key_pad = None
        if attention_mask is not None:
            key_pad = (attention_mask == 0).float()

        # ── 4. Fused transformer blocks ───────────────────────────────────────
        for block in self.blocks:
            x = block(x, alibi_bias=alibi, key_padding_mask=key_pad,
                      species_emb=species_emb)

        x = self.final_norm(x)

        # ── 5. Output heads ───────────────────────────────────────────────────
        mlm_logits = self.mlm_head(x)           # (B, L, vocab_size)
        recon_logits = self.recon_head(x)        # (B, L, 5)
        per_base_conf, reliability, temperature = self.confidence_head(
            x, attention_mask
        )

        return {
            "mlm_logits":       mlm_logits,
            "recon_logits":     recon_logits,
            "per_base_conf":    per_base_conf,   # (B, L)
            "reliability":      reliability,      # (B,)
            "temperature":      temperature,
            "hidden_states":    x,                # (B, L, D)
        }

    def get_embeddings(self, tokens, attention_mask=None,
                       species_feats=None, adjacency=None,
                       species_idx=None):
        """Return fused hidden states without output heads."""
        out = self.forward(tokens, attention_mask, species_feats,
                           adjacency, species_idx)
        return out["hidden_states"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-Species Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════
def multi_species_reconstruct(
    model:          TransformerGNNFusion,
    sequences:      Dict[str, str],        # species_name → damaged sequence
    vocab:          Dict[str, int],
    species_names:  List[str],
    species_feats:  torch.Tensor,          # (N_species, feat_dim)
    adjacency:      torch.Tensor,          # (N_species, N_species)
    device:         torch.device = None,
    k:              int = 6,
) -> Dict[str, Dict]:
    """
    Reconstruct all species simultaneously using the fused model.
    Returns per-species: reconstructed sequence, confidences, reliability.
    """
    from preprocessing.encoding import encode_kmer_sequence

    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)
    species_feats = species_feats.to(device)
    adjacency = adjacency.to(device)

    inv_vocab = {v: k_str for k_str, v in vocab.items()}
    mask_id = vocab["[MASK]"]
    cls_id  = vocab["[CLS]"]
    pad_id  = vocab["[PAD]"]
    max_len = model.max_len

    results = {}

    for sp_idx, sp_name in enumerate(species_names):
        if sp_name not in sequences:
            continue

        seq = list(sequences[sp_name].upper())
        confidences = [1.0] * len(seq)
        chunk = (max_len - 1) * k

        for start in range(0, len(seq), chunk):
            end = min(start + chunk, len(seq))
            window_seq = "".join(seq[start:end])
            kmers = encode_kmer_sequence(window_seq, vocab, k)

            tokens = np.concatenate([[cls_id], kmers]).astype(np.int32)
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), constant_values=pad_id)

            # Mask N positions
            for j in range(1, max_len - pad_len):
                kmer_start = start + (j - 1) * k
                kmer_end = kmer_start + k
                if any(seq[p] == "N"
                       for p in range(kmer_start, min(kmer_end, len(seq)))):
                    tokens[j] = mask_id

            t_tensor = torch.tensor(
                tokens, dtype=torch.long
            ).unsqueeze(0).to(device)

            att = torch.tensor(
                (tokens != pad_id).astype(np.float32)
            ).unsqueeze(0).to(device)

            sp_idx_tensor = torch.tensor(
                [sp_idx], dtype=torch.long
            ).to(device)

            with torch.no_grad():
                out = model(
                    t_tensor, att,
                    species_feats=species_feats,
                    adjacency=adjacency,
                    species_idx=sp_idx_tensor,
                )
                logits = out["mlm_logits"]
                probs = torch.softmax(logits[0], dim=-1)
                base_conf = out["per_base_conf"][0]

            for j in range(1, max_len - pad_len):
                if int(t_tensor[0, j].cpu()) != mask_id:
                    continue
                top_id = int(probs[j].argmax().cpu())
                top_kmer = inv_vocab.get(top_id, "AAAAAA")
                conf = float(base_conf[j].cpu())

                kmer_start = start + (j - 1) * k
                for offset, base in enumerate(top_kmer):
                    pos = kmer_start + offset
                    if pos < len(seq) and seq[pos] == "N" and base in "ACGT":
                        seq[pos] = base
                        confidences[pos] = round(conf, 4)

        reliability = float(
            out["reliability"][0].cpu()
        ) if "reliability" in out else 0.5

        results[sp_name] = {
            "reconstructed_seq": "".join(seq),
            "confidences":       confidences,
            "reliability_score": round(reliability, 4),
            "gaps_remaining":    "".join(seq).count("N"),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: build phylo graph for the fusion model
# ═══════════════════════════════════════════════════════════════════════════════
def build_fusion_phylo_graph(
    species_names: List[str],
    sequences:     Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build species feature matrix and adjacency matrix for the fusion model.
    Returns: (species_feats, adjacency) as tensors.
    """
    from models.gnn_phylogenetic import kmer_frequency_vector

    N = len(species_names)
    feats = np.zeros((N, 4**4), dtype=np.float32)
    adj = np.zeros((N, N), dtype=np.float32)

    for i, sp in enumerate(species_names):
        seq = sequences.get(sp, "ACGT" * 100)
        feats[i] = kmer_frequency_vector(seq, k=4)

    for (sp1, sp2), dist in PHYLO_DISTANCES.items():
        if sp1 in species_names and sp2 in species_names:
            i = species_names.index(sp1)
            j = species_names.index(sp2)
            weight = 1.0 / (1.0 + dist)
            adj[i, j] = adj[j, i] = weight

    np.fill_diagonal(adj, 1.0)
    return torch.tensor(feats), torch.tensor(adj)
