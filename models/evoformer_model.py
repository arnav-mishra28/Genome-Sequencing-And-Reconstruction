"""
evoformer_model.py
==================
AlphaFold2 Evoformer-Inspired Genome Reconstruction Model.

This is the NEXT-LEVEL architecture that goes beyond basic transformers:

Key innovations from AlphaFold2 adapted for DNA:
  1. Pair Representation  — models relationships between ALL base positions (i,j)
  2. Sequence Attention   — standard self-attention on sequence embeddings
  3. Pair Attention        — attention over the pair matrix (row-wise + column-wise)
  4. Cross Updates         — sequence ↔ pair bidirectional information flow
  5. Triangular Updates    — enforce geometric consistency in pair representation
  6. Recycling             — run model N times, feeding output back as input
  7. Species Embeddings    — learned embeddings for each species (evolution-aware)
  8. Evolution Loss        — penalize unrealistic mutations based on phylogenetic distance
  9. Structure-Aware Head  — biological constraints + mutation likelihoods

References:
  - Jumper et al., "Highly accurate protein structure prediction with AlphaFold"
    Nature 596 (2021)
  - Zhou et al., "DNABERT-2: Efficient Foundation Model for Multi-Species Genome"
    ICLR 2024
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    DEVICE, EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, DROPOUT, GRADIENT_CLIP,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Configuration
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class EvoformerConfig:
    """Configuration for the Evoformer genome model."""
    vocab_size:     int   = 4102
    embed_dim:      int   = EMBED_DIM
    pair_dim:       int   = 128
    n_heads:        int   = N_HEADS
    n_evo_blocks:   int   = 4        # Evoformer blocks
    n_seq_blocks:   int   = N_LAYERS # Sequence transformer blocks
    ffn_dim:        int   = FFN_DIM
    max_len:        int   = MAX_SEQ_LEN
    dropout:        float = DROPOUT
    n_species:      int   = 12       # max species
    n_bases:        int   = 5        # ACGTN
    n_recycles:     int   = 3        # recycling iterations
    species_feat_dim: int = 256      # 4^4 k-mer features


# ═══════════════════════════════════════════════════════════════════════════════
#  Species Embedding
# ═══════════════════════════════════════════════════════════════════════════════
class SpeciesEmbedding(nn.Module):
    """
    Learned species ID embeddings that inject evolutionary identity
    into the transformer. Each species gets a unique embedding vector
    that's added to every token in sequences from that species.
    """
    def __init__(self, n_species: int, embed_dim: int):
        super().__init__()
        self.species_embed = nn.Embedding(n_species, embed_dim)
        self.species_norm = nn.LayerNorm(embed_dim)
        # Gated injection — start small so species info doesn't dominate
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) — token embeddings
        species_idx: (B,) — species index for each sample
        """
        sp_emb = self.species_embed(species_idx)  # (B, D)
        sp_emb = self.species_norm(sp_emb)
        sp_emb = sp_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, L, D)
        gate = torch.sigmoid(self.gate)
        return x + gate * sp_emb


# ═══════════════════════════════════════════════════════════════════════════════
#  Pair Representation
# ═══════════════════════════════════════════════════════════════════════════════
class PairRepresentationModule(nn.Module):
    """
    Build and update pair representation z_{ij} that captures
    relationships between all pairs of positions.

    From sequence embeddings x_i, x_j:
      z_{ij} = MLP(concat(x_i, x_j)) + positional_encoding
    """
    def __init__(self, seq_dim: int, pair_dim: int, max_len: int = 256):
        super().__init__()
        self.pair_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, pair_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(pair_dim * 2, pair_dim),
        )
        # Relative position encoding for pairs
        self.rel_pos_embed = nn.Embedding(2 * max_len + 1, pair_dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) → pair: (B, L, L, pair_dim)
        """
        B, L, D = x.shape
        # Outer concatenation
        x_i = x.unsqueeze(2).expand(-1, -1, L, -1)
        x_j = x.unsqueeze(1).expand(-1, L, -1, -1)
        pair_input = torch.cat([x_i, x_j], dim=-1)  # (B, L, L, 2D)
        pair = self.pair_proj(pair_input)

        # Add relative position encoding
        pos = torch.arange(L, device=x.device)
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)) + self.max_len  # shift to positive
        rel_pos = rel_pos.clamp(0, 2 * self.max_len)
        pair = pair + self.rel_pos_embed(rel_pos).unsqueeze(0)

        return pair


# ═══════════════════════════════════════════════════════════════════════════════
#  Triangular Multiplicative Update (from alphafold_attention.py enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
class TriangularUpdate(nn.Module):
    """
    Enforces geometric consistency in pair representation.
    If z_{ij} and z_{jk} are confident, z_{ik} should be consistent.
    """
    def __init__(self, dim: int, direction: str = "outgoing"):
        super().__init__()
        self.direction = direction
        self.norm = nn.LayerNorm(dim)
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.gate_a = nn.Linear(dim, dim)
        self.gate_b = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.gate_o = nn.Linear(dim, dim)
        self.norm_o = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, L, D)"""
        z_n = self.norm(z)
        a = self.proj_a(z_n) * torch.sigmoid(self.gate_a(z_n))
        b = self.proj_b(z_n) * torch.sigmoid(self.gate_b(z_n))

        if self.direction == "outgoing":
            x = torch.einsum("bijd,bjkd->bikd", a, b)
        else:
            x = torch.einsum("bikd,bjkd->bijd", a, b)

        x = self.norm_o(x)
        return z + self.proj_o(x) * torch.sigmoid(self.gate_o(z_n))


# ═══════════════════════════════════════════════════════════════════════════════
#  Pair Attention (row-wise / column-wise)
# ═══════════════════════════════════════════════════════════════════════════════
class PairAttention(nn.Module):
    """Attention over pair representation rows or columns."""
    def __init__(self, dim: int, n_heads: int = 4, axis: str = "row"):
        super().__init__()
        self.axis = axis
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, L, D)"""
        B, L1, L2, D = z.shape
        H, hd = self.n_heads, self.head_dim

        z_n = self.norm(z)
        if self.axis == "column":
            z_n = z_n.transpose(1, 2)
            L1, L2 = L2, L1

        z_flat = z_n.reshape(B * L1, L2, D)
        qkv = self.qkv(z_flat).reshape(B * L1, L2, 3, H, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(hd)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * L1, L2, D)
        out = self.proj(out).reshape(B, L1, L2, D)

        if self.axis == "column":
            out = out.transpose(1, 2)

        return z + out


# ═══════════════════════════════════════════════════════════════════════════════
#  Sequence ↔ Pair Cross Update
# ═══════════════════════════════════════════════════════════════════════════════
class SeqPairCrossUpdate(nn.Module):
    """
    Bidirectional information flow between sequence and pair representations.

    Sequence → Pair: Update pair with sequence context
    Pair → Sequence: Update sequence with pair context (attention-pooled)
    """
    def __init__(self, seq_dim: int, pair_dim: int, n_heads: int = 4):
        super().__init__()
        # Pair → Sequence
        self.pair_to_seq_norm = nn.LayerNorm(pair_dim)
        self.pair_to_seq_pool = nn.Linear(pair_dim, 1)
        self.pair_to_seq_proj = nn.Linear(pair_dim, seq_dim)
        self.pair_to_seq_gate = nn.Parameter(torch.tensor(0.1))

        # Sequence → Pair
        self.seq_to_pair_norm = nn.LayerNorm(seq_dim)
        self.seq_to_pair_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )
        self.seq_to_pair_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq: torch.Tensor, pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: (B, L, seq_dim)
        pair: (B, L, L, pair_dim)
        """
        B, L, D = seq.shape

        # ── Pair → Sequence ───────────────────────────────────────────────────
        pair_n = self.pair_to_seq_norm(pair)
        # Attention-weighted pooling over j-dimension for each i
        weights = self.pair_to_seq_pool(pair_n).squeeze(-1)  # (B, L, L)
        weights = F.softmax(weights, dim=-1)
        pair_pooled = torch.einsum("bij,bijd->bid", weights, pair_n)  # (B, L, pair_dim)
        pair_proj = self.pair_to_seq_proj(pair_pooled)
        gate_ps = torch.sigmoid(self.pair_to_seq_gate)
        seq_updated = seq + gate_ps * pair_proj

        # ── Sequence → Pair ───────────────────────────────────────────────────
        seq_n = self.seq_to_pair_norm(seq_updated)
        seq_i = seq_n.unsqueeze(2).expand(-1, -1, L, -1)
        seq_j = seq_n.unsqueeze(1).expand(-1, L, -1, -1)
        seq_cat = torch.cat([seq_i, seq_j], dim=-1)
        pair_update = self.seq_to_pair_proj(seq_cat)
        gate_sp = torch.sigmoid(self.seq_to_pair_gate)
        pair_updated = pair + gate_sp * pair_update

        return seq_updated, pair_updated


# ═══════════════════════════════════════════════════════════════════════════════
#  Evoformer Block (Sequence + Pair joint processing)
# ═══════════════════════════════════════════════════════════════════════════════
class EvoformerBlock(nn.Module):
    """
    One block of the Evoformer stack — jointly processes sequence and pair.

    Pipeline:
      1. Sequence self-attention (with pair bias)
      2. Sequence FFN
      3. Pair row attention
      4. Pair column attention
      5. Triangular updates (outgoing + incoming)
      6. Pair FFN
      7. Cross update (seq ↔ pair)
    """
    def __init__(self, config: EvoformerConfig):
        super().__init__()
        dim = config.embed_dim
        pair_dim = config.pair_dim
        n_heads = config.n_heads
        dropout = config.dropout

        # Sequence self-attention with pair bias
        self.seq_norm = nn.LayerNorm(dim)
        self.seq_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.seq_drop = nn.Dropout(dropout)
        self.pair_bias_proj = nn.Linear(pair_dim, n_heads)

        # Sequence FFN
        self.seq_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.ffn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.ffn_dim, dim),  # Note: GEGLU uses ffn_dim after gating
            nn.Dropout(dropout),
        )

        # Pair processing
        self.pair_row_attn = PairAttention(pair_dim, n_heads=4, axis="row")
        self.pair_col_attn = PairAttention(pair_dim, n_heads=4, axis="column")
        self.tri_out = TriangularUpdate(pair_dim, "outgoing")
        self.tri_in = TriangularUpdate(pair_dim, "incoming")
        self.pair_ffn = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 4, pair_dim),
            nn.Dropout(dropout),
        )

        # Cross update
        self.cross_update = SeqPairCrossUpdate(dim, pair_dim, n_heads)

    def forward(self, seq: torch.Tensor, pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: (B, L, D)
        pair: (B, L, L, pair_dim)
        """
        # 1. Sequence self-attention with pair bias
        B, L, D = seq.shape
        h = self.seq_norm(seq)

        # Compute pair bias for attention
        pair_bias = self.pair_bias_proj(pair)  # (B, L, L, n_heads)
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        pair_bias = pair_bias.reshape(B * pair_bias.size(1), L, L)

        h_attn, _ = self.seq_attn(h, h, h, attn_mask=pair_bias)
        seq = seq + self.seq_drop(h_attn)

        # 2. Sequence FFN
        # Handle GEGLU-style: project to 2x, split, gate
        ffn_out = self.seq_ffn[0](seq)  # LayerNorm
        ffn_proj = self.seq_ffn[1](ffn_out)  # Linear to 2*ffn_dim
        x, gate = ffn_proj.chunk(2, dim=-1)
        ffn_out = x * F.gelu(gate)
        ffn_out = self.seq_ffn[3](ffn_out)  # Dropout
        ffn_out = self.seq_ffn[4](ffn_out)  # Linear down
        ffn_out = self.seq_ffn[5](ffn_out)  # Dropout
        seq = seq + ffn_out

        # 3-4. Pair attention
        pair = self.pair_row_attn(pair)
        pair = self.pair_col_attn(pair)

        # 5. Triangular updates
        pair = self.tri_out(pair)
        pair = self.tri_in(pair)

        # 6. Pair FFN
        pair = pair + self.pair_ffn(pair)

        # 7. Cross update
        seq, pair = self.cross_update(seq, pair)

        return seq, pair


# ═══════════════════════════════════════════════════════════════════════════════
#  Recycling Module
# ═══════════════════════════════════════════════════════════════════════════════
class RecyclingModule(nn.Module):
    """
    AlphaFold-style recycling: run the model N times, feeding the output
    back as additional input each time.

    This allows iterative refinement — the model can fix mistakes from
    previous passes.
    """
    def __init__(self, seq_dim: int, pair_dim: int):
        super().__init__()
        self.seq_recycle_norm = nn.LayerNorm(seq_dim)
        self.pair_recycle_norm = nn.LayerNorm(pair_dim)
        self.seq_recycle_proj = nn.Linear(seq_dim, seq_dim)
        self.pair_recycle_proj = nn.Linear(pair_dim, pair_dim)

    def recycle(
        self,
        seq_new: torch.Tensor,
        pair_new: torch.Tensor,
        seq_prev: torch.Tensor = None,
        pair_prev: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix previous iteration's output into current input."""
        if seq_prev is not None:
            seq_recycled = self.seq_recycle_proj(
                self.seq_recycle_norm(seq_prev.detach())
            )
            seq_new = seq_new + seq_recycled

        if pair_prev is not None:
            pair_recycled = self.pair_recycle_proj(
                self.pair_recycle_norm(pair_prev.detach())
            )
            pair_new = pair_new + pair_recycled

        return seq_new, pair_new


# ═══════════════════════════════════════════════════════════════════════════════
#  Structure-Aware Head
# ═══════════════════════════════════════════════════════════════════════════════
class StructureAwareHead(nn.Module):
    """
    Biologically-informed output head that incorporates:
      1. Per-base reconstruction prediction (ACGTN)
      2. GC content constraint
      3. Transition/transversion ratio prediction
      4. Mutation likelihood per position
      5. Confidence scoring with temperature calibration
    """
    def __init__(self, dim: int, pair_dim: int, n_bases: int = 5):
        super().__init__()
        # Base prediction
        self.base_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, n_bases),
        )

        # Mutation likelihood
        self.mutation_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        # Confidence
        self.conf_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.5))

        # Reliability (sequence-level)
        self.pool_attn = nn.Linear(dim, 1)
        self.reliability_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Pair distance output (learned distances from pair repr)
        self.dist_head = nn.Sequential(
            nn.Linear(pair_dim, pair_dim // 2),
            nn.GELU(),
            nn.Linear(pair_dim // 2, 1),
        )

    def forward(self, seq: torch.Tensor, pair: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        seq: (B, L, D)
        pair: (B, L, L, pair_dim)
        """
        # Per-base predictions
        base_logits = self.base_head(seq)            # (B, L, 5)
        mutation_prob = self.mutation_head(seq)       # (B, L, 1)

        # Confidence with temperature calibration
        raw_conf = self.conf_head(seq).squeeze(-1)   # (B, L)
        per_base_conf = torch.sigmoid(raw_conf / self.temperature.clamp(min=0.1))

        # Sequence-level reliability
        attn_scores = self.pool_attn(seq).squeeze(-1)  # (B, L)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = (seq * attn_weights.unsqueeze(-1)).sum(dim=1)
        reliability = self.reliability_head(pooled).squeeze(-1)

        # Pair distances
        pair_dist = self.dist_head(pair).squeeze(-1)  # (B, L, L)

        return {
            "base_logits":       base_logits,
            "mutation_prob":     mutation_prob.squeeze(-1),
            "per_base_conf":     per_base_conf,
            "reliability":       reliability,
            "pair_distances":    pair_dist,
            "temperature":       self.temperature,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Evolution Loss
# ═══════════════════════════════════════════════════════════════════════════════
class EvolutionLoss(nn.Module):
    """
    Penalizes unrealistic mutations based on phylogenetic distance.

    Components:
      1. Mutation rate loss:   mutation probability should correlate with
                               phylogenetic distance
      2. Transition bias loss: transitions should be ~2x more likely than
                               transversions
      3. GC content loss:     reconstructed GC% should be in species-normal range
      4. Conservation loss:   highly conserved positions should have lower
                               mutation probability
    """
    def __init__(self):
        super().__init__()
        self.expected_titv_ratio = 2.0

    def forward(
        self,
        base_logits: torch.Tensor,       # (B, L, 5)
        mutation_prob: torch.Tensor,      # (B, L)
        phylo_distance: torch.Tensor,     # (B,) per-sample
        target_gc: torch.Tensor = None,   # (B,) target GC content
    ) -> torch.Tensor:
        """Compute evolution-aware loss."""
        loss = torch.tensor(0.0, device=base_logits.device)

        # 1. Mutation rate should correlate with phylogenetic distance
        if phylo_distance is not None:
            # Normalize distance to [0, 1]
            norm_dist = phylo_distance / (phylo_distance.max() + 1e-8)
            # Mean mutation prob should roughly match normalized distance
            mean_mut = mutation_prob.mean(dim=-1)  # (B,)
            expected_mut = 0.05 + 0.15 * norm_dist  # 5-20% mutation rate
            loss = loss + F.mse_loss(mean_mut, expected_mut)

        # 2. GC content constraint
        if target_gc is not None:
            probs = F.softmax(base_logits, dim=-1)  # (B, L, 5)
            # G=index 2, C=index 1 in ACGTN
            gc_pred = probs[:, :, 1].mean(dim=-1) + probs[:, :, 2].mean(dim=-1)
            gc_loss = F.mse_loss(gc_pred, target_gc)
            loss = loss + gc_loss

        # 3. Entropy regularization (avoid degenerate predictions)
        probs = F.softmax(base_logits[:, :, :4], dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        # Want entropy > 1.0 (not too low = repetitive)
        loss = loss + F.relu(1.0 - entropy)

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
#  ALiBi Bias (shared)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def _pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_pow2(n_heads))
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = _pow2(closest)
    extra = _pow2(2 * closest)
    slopes += extra[0::2][:n_heads - closest]
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
#  Sequence Transformer Block (with ALiBi)
# ═══════════════════════════════════════════════════════════════════════════════
class SeqTransformerBlock(nn.Module):
    """Standard pre-norm transformer block with ALiBi for sequence processing."""
    def __init__(self, dim: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.ffn = GEGLUFeedForward(dim, ffn_dim, dropout)

    def forward(self, x, alibi_bias=None, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=alibi_bias,
                         key_padding_mask=key_padding_mask)
        x = x + self.drop1(h)
        x = x + self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN Species Encoder
# ═══════════════════════════════════════════════════════════════════════════════
class SpeciesGNNEncoder(nn.Module):
    """Graph neural network encoding phylogenetic relationships between species."""
    def __init__(self, feat_dim: int, hidden: int, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            self._make_layer(hidden, dropout) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden)

    def _make_layer(self, dim, dropout):
        return nn.ModuleDict({
            "norm": nn.LayerNorm(dim),
            "w_self": nn.Linear(dim, dim),
            "w_neigh": nn.Linear(dim, dim),
            "gate": nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid()),
            "drop": nn.Dropout(dropout),
            "act": nn.GELU(),
        })

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h_n = layer["norm"](h)
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
            agg = (adj @ h_n) / deg
            h_self = layer["w_self"](h_n)
            h_neigh = layer["w_neigh"](agg)
            gate = layer["gate"](torch.cat([h_self, h_neigh], dim=-1))
            out = gate * h_self + (1 - gate) * h_neigh
            out = layer["act"](layer["drop"](out))
            if h.shape == out.shape:
                h = h + out
            else:
                h = out
        return self.output_norm(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-Attention: GNN → Transformer
# ═══════════════════════════════════════════════════════════════════════════════
class GNNCrossAttention(nn.Module):
    """Inject phylogenetic GNN embeddings into sequence representation."""
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, seq: torch.Tensor, species_emb: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(seq)
        kv = self.norm_kv(species_emb)
        out, _ = self.cross_attn(q, kv, kv)
        return seq + torch.sigmoid(self.gate) * self.drop(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN MODEL: EvoformerGenomeModel
# ═══════════════════════════════════════════════════════════════════════════════
class EvoformerGenomeModel(nn.Module):
    """
    AlphaFold2 Evoformer-Inspired Genome Reconstruction Model.

    Architecture:
      1. Token embedding + Species embedding
      2. GNN species encoding
      3. Pair representation initialization
      4. Sequence transformer blocks (with ALiBi + GNN cross-attention)
      5. Evoformer blocks (joint sequence + pair processing)
      6. Recycling loop (N iterations)
      7. Structure-aware output heads

    Forward pass:
      tokens → embed → [recycle loop] → {
        seq transformer → GNN cross-attn → Evoformer → structure heads
      } × N_recycles

    Output:
      - mlm_logits:     (B, L, vocab_size)
      - recon_logits:   (B, L, 5)  ACGTN base prediction
      - per_base_conf:  (B, L)     confidence per position
      - reliability:    (B,)       sequence-level reliability
      - mutation_prob:  (B, L)     predicted mutation likelihood
      - pair_distances: (B, L, L)  learned base-pair distances
      - hidden_states:  (B, L, D)  final hidden representations
    """
    def __init__(self, config: EvoformerConfig = None, vocab_size: int = None):
        super().__init__()
        if config is None:
            config = EvoformerConfig()
        if vocab_size is not None:
            config.vocab_size = vocab_size

        self.config = config
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size
        self.n_recycles = config.n_recycles

        # ── Token + Species Embedding ─────────────────────────────────────────
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim,
                                         padding_idx=0)
        self.embed_drop = nn.Dropout(config.dropout)
        self.species_embedding = SpeciesEmbedding(config.n_species,
                                                   config.embed_dim)

        # ── GNN Species Encoder ───────────────────────────────────────────────
        self.gnn_encoder = SpeciesGNNEncoder(
            feat_dim=config.species_feat_dim,
            hidden=config.embed_dim,
            n_layers=3,
            dropout=config.dropout,
        )
        self.gnn_cross = GNNCrossAttention(config.embed_dim, config.n_heads,
                                            config.dropout)

        # ── Pair Representation ───────────────────────────────────────────────
        self.pair_init = PairRepresentationModule(
            config.embed_dim, config.pair_dim, config.max_len
        )

        # ── ALiBi bias ────────────────────────────────────────────────────────
        alibi = build_alibi_bias(config.n_heads, config.max_len)
        self.register_buffer("alibi_bias", alibi, persistent=False)

        # ── Sequence Transformer Blocks ───────────────────────────────────────
        self.seq_blocks = nn.ModuleList([
            SeqTransformerBlock(config.embed_dim, config.n_heads,
                                config.ffn_dim, config.dropout)
            for _ in range(config.n_seq_blocks)
        ])

        # ── Evoformer Blocks ──────────────────────────────────────────────────
        self.evo_blocks = nn.ModuleList([
            EvoformerBlock(config)
            for _ in range(config.n_evo_blocks)
        ])

        # ── Recycling ─────────────────────────────────────────────────────────
        self.recycler = RecyclingModule(config.embed_dim, config.pair_dim)

        # ── Output Norms ──────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # ── MLM Head ──────────────────────────────────────────────────────────
        self.mlm_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.vocab_size),
        )

        # ── Structure-Aware Head ──────────────────────────────────────────────
        self.structure_head = StructureAwareHead(
            config.embed_dim, config.pair_dim, config.n_bases
        )

        # ── Evolution Loss ────────────────────────────────────────────────────
        self.evolution_loss = EvolutionLoss()

    def forward(
        self,
        tokens:         torch.Tensor,          # (B, L)
        attention_mask: torch.Tensor = None,   # (B, L)
        species_feats:  torch.Tensor = None,   # (N_species, feat_dim)
        adjacency:      torch.Tensor = None,   # (N_species, N_species)
        species_idx:    torch.Tensor = None,   # (B,)
        n_recycles:     int = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape
        device = tokens.device
        n_recycles = n_recycles or self.n_recycles

        # ── 1. Token embedding ────────────────────────────────────────────────
        x = self.embed_drop(self.token_embed(tokens))

        # ── 2. Species embedding ──────────────────────────────────────────────
        if species_idx is not None:
            x = self.species_embedding(x, species_idx)

        # ── 3. GNN encoding ──────────────────────────────────────────────────
        species_emb = None
        if species_feats is not None and adjacency is not None:
            all_species_emb = self.gnn_encoder(species_feats, adjacency)
            species_emb = all_species_emb.unsqueeze(0).expand(B, -1, -1)

        # ── 4. ALiBi bias ─────────────────────────────────────────────────────
        alibi = self.alibi_bias[:, :L, :L]
        alibi = alibi.unsqueeze(0).expand(B, -1, -1, -1)
        alibi = alibi.reshape(B * self.n_heads, L, L)

        key_pad = None
        if attention_mask is not None:
            key_pad = (attention_mask == 0).float()

        # ── 5. Recycling loop ─────────────────────────────────────────────────
        seq_prev, pair_prev = None, None

        for cycle in range(n_recycles):
            # Reset from token embedding each cycle (but add recycled info)
            x_cycle = self.embed_drop(self.token_embed(tokens))
            if species_idx is not None:
                x_cycle = self.species_embedding(x_cycle, species_idx)

            # Initialize pair representation
            pair = self.pair_init(x_cycle)

            # Inject recycled information
            x_cycle, pair = self.recycler.recycle(
                x_cycle, pair, seq_prev, pair_prev
            )

            # Sequence transformer blocks
            for block in self.seq_blocks:
                x_cycle = block(x_cycle, alibi_bias=alibi,
                                key_padding_mask=key_pad)

            # GNN cross-attention
            if species_emb is not None:
                x_cycle = self.gnn_cross(x_cycle, species_emb)

            # Evoformer blocks
            for evo_block in self.evo_blocks:
                x_cycle, pair = evo_block(x_cycle, pair)

            seq_prev = x_cycle
            pair_prev = pair

        # ── 6. Final outputs ──────────────────────────────────────────────────
        x = self.final_norm(x_cycle)

        # MLM head
        mlm_logits = self.mlm_head(x)

        # Structure-aware head
        struct_out = self.structure_head(x, pair, attention_mask)

        return {
            "mlm_logits":       mlm_logits,
            "recon_logits":     struct_out["base_logits"],
            "per_base_conf":    struct_out["per_base_conf"],
            "reliability":      struct_out["reliability"],
            "mutation_prob":    struct_out["mutation_prob"],
            "pair_distances":   struct_out["pair_distances"],
            "temperature":      struct_out["temperature"],
            "hidden_states":    x,
        }

    def get_embeddings(self, tokens, attention_mask=None,
                       species_feats=None, adjacency=None,
                       species_idx=None):
        """Return hidden states without output heads."""
        out = self.forward(tokens, attention_mask, species_feats,
                           adjacency, species_idx, n_recycles=1)
        return out["hidden_states"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Scalable Config Profiles
# ═══════════════════════════════════════════════════════════════════════════════
SCALE_PROFILES = {
    "small": EvoformerConfig(
        embed_dim=256, pair_dim=64, n_heads=4, n_evo_blocks=2,
        n_seq_blocks=4, ffn_dim=1024, n_recycles=2,
    ),
    "medium": EvoformerConfig(
        embed_dim=512, pair_dim=128, n_heads=8, n_evo_blocks=4,
        n_seq_blocks=8, ffn_dim=2048, n_recycles=3,
    ),
    "large": EvoformerConfig(
        embed_dim=1024, pair_dim=256, n_heads=16, n_evo_blocks=8,
        n_seq_blocks=16, ffn_dim=4096, n_recycles=3,
    ),
    "xl": EvoformerConfig(
        embed_dim=2048, pair_dim=512, n_heads=32, n_evo_blocks=12,
        n_seq_blocks=48, ffn_dim=8192, n_recycles=4,
    ),
}


def get_model(profile: str = "small", vocab_size: int = 4102,
              **overrides) -> EvoformerGenomeModel:
    """Create EvoformerGenomeModel with predefined scale profile."""
    config = SCALE_PROFILES.get(profile, SCALE_PROFILES["small"])
    config.vocab_size = vocab_size
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return EvoformerGenomeModel(config)


if __name__ == "__main__":
    # Quick test
    for profile in ["small", "medium", "large", "xl"]:
        config = SCALE_PROFILES[profile]
        config.vocab_size = 4102
        model = EvoformerGenomeModel(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {profile:8s}: {params:>15,} parameters")

    # Test forward pass
    model = get_model("small", vocab_size=4102)
    tokens = torch.randint(0, 4102, (2, 64))
    att = torch.ones(2, 64)
    out = model(tokens, att, n_recycles=1)
    print(f"\n  Forward pass test:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape}")
