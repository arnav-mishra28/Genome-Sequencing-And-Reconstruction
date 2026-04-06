"""
alphafold_attention.py
======================
AlphaFold Evoformer-inspired attention modules for DNA sequence analysis.

Key ideas from AlphaFold2:
  - Pairwise representation learning (residue-pair distances)
  - MSA row/column attention (adapted for DNA MSAs)
  - Triangular multiplicative updates for geometric consistency
  - Iterative refinement blocks

Reference:
  - Jumper et al., "Highly accurate protein structure prediction with
    AlphaFold", Nature 596 (2021)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Triangular Multiplicative Update
# ═══════════════════════════════════════════════════════════════════════════════
class TriangularMultiplicativeUpdate(nn.Module):
    """
    Enforces geometric consistency in pairwise representations.
    If z_{ij} and z_{jk} are both confident, z_{ik} should be consistent.

    direction: "outgoing" or "incoming"
    """
    def __init__(self, dim: int, direction: str = "outgoing"):
        super().__init__()
        self.direction = direction
        self.norm   = nn.LayerNorm(dim)
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.gate_a = nn.Linear(dim, dim)
        self.gate_b = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)
        self.gate_o = nn.Linear(dim, dim)
        self.norm_o = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, L, D) pairwise representation."""
        z_n = self.norm(z)
        a = self.proj_a(z_n) * torch.sigmoid(self.gate_a(z_n))
        b = self.proj_b(z_n) * torch.sigmoid(self.gate_b(z_n))

        if self.direction == "outgoing":
            # z_{ik} = sum_j a_{ij} * b_{jk}
            x = torch.einsum("bijd,bjkd->bikd", a, b)
        else:
            # z_{ij} = sum_k a_{ik} * b_{jk}
            x = torch.einsum("bikd,bjkd->bijd", a, b)

        x = self.norm_o(x)
        return z + self.proj_o(x) * torch.sigmoid(self.gate_o(z_n))


# ═══════════════════════════════════════════════════════════════════════════════
#  Pairwise Attention (row-wise / column-wise)
# ═══════════════════════════════════════════════════════════════════════════════
class PairwiseAttention(nn.Module):
    """
    Attention over pairwise representation rows or columns.
    Used to propagate information along one axis of the pair matrix.
    """
    def __init__(self, dim: int, n_heads: int = 4,
                 axis: str = "row"):
        super().__init__()
        self.axis     = axis
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.norm     = nn.LayerNorm(dim)
        self.qkv      = nn.Linear(dim, 3 * dim)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, L, D)"""
        B, L1, L2, D = z.shape
        H, hd = self.n_heads, self.head_dim

        z_n = self.norm(z)

        if self.axis == "column":
            z_n = z_n.transpose(1, 2)  # treat columns as sequences
            L1, L2 = L2, L1

        # Reshape to (B*L1, L2, D) — attend along L2
        z_flat = z_n.reshape(B * L1, L2, D)
        qkv = self.qkv(z_flat).reshape(B * L1, L2, 3, H, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(hd)
        attn  = (q @ k.transpose(-2, -1)) / scale
        attn  = F.softmax(attn, dim=-1)
        out   = (attn @ v).transpose(1, 2).reshape(B * L1, L2, D)
        out   = self.proj(out).reshape(B, L1, L2, D)

        if self.axis == "column":
            out = out.transpose(1, 2)

        return z + out


# ═══════════════════════════════════════════════════════════════════════════════
#  MSA Row Attention  (adapted for DNA MSAs)
# ═══════════════════════════════════════════════════════════════════════════════
class MSARowAttention(nn.Module):
    """
    Self-attention across rows of a DNA multiple sequence alignment.
    Each row is a different species' sequence.
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.norm     = nn.LayerNorm(dim)
        self.qkv      = nn.Linear(dim, 3 * dim)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, msa: torch.Tensor,
                pair_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        msa: (B, N_seq, L, D)
        pair_bias: (B, L, L, n_heads) optional bias from pairwise repr
        """
        B, N, L, D = msa.shape
        H, hd = self.n_heads, self.head_dim

        m = self.norm(msa)
        # Attend within each column position across sequences
        m_flat = m.reshape(B * N, L, D)
        qkv = self.qkv(m_flat).reshape(B * N, L, 3, H, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(hd)
        attn  = (q @ k.transpose(-2, -1)) / scale

        if pair_bias is not None:
            # pair_bias: (B, L, L, H) → (B, H, L, L) → broadcast
            bias = pair_bias.permute(0, 3, 1, 2)  # (B, H, L, L)
            bias = bias.unsqueeze(1).expand(-1, N, -1, -1, -1)
            bias = bias.reshape(B * N, H, L, L)
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B * N, L, D)
        out  = self.proj(out).reshape(B, N, L, D)

        return msa + out


# ═══════════════════════════════════════════════════════════════════════════════
#  Evoformer Block
# ═══════════════════════════════════════════════════════════════════════════════
class EvoformerBlock(nn.Module):
    """
    One block of the Evoformer stack.
    Processes both MSA and pairwise representations jointly.
    """
    def __init__(self, dim: int, pair_dim: int, n_heads: int = 4):
        super().__init__()
        # MSA processing
        self.msa_row_attn = MSARowAttention(dim, n_heads)
        self.msa_ffn      = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Pairwise processing
        self.pair_row_attn = PairwiseAttention(pair_dim, n_heads, axis="row")
        self.pair_col_attn = PairwiseAttention(pair_dim, n_heads, axis="column")
        self.tri_out       = TriangularMultiplicativeUpdate(pair_dim, "outgoing")
        self.tri_in        = TriangularMultiplicativeUpdate(pair_dim, "incoming")
        self.pair_ffn      = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim * 4),
            nn.GELU(),
            nn.Linear(pair_dim * 4, pair_dim),
        )

    def forward(self, msa: torch.Tensor,
                pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        msa:  (B, N_seq, L, D)
        pair: (B, L, L, pair_dim)
        """
        # MSA row attention with pair bias
        pair_bias = nn.Linear(pair.shape[-1], self.msa_row_attn.n_heads,
                              device=pair.device)(pair)
        msa = self.msa_row_attn(msa, pair_bias=pair_bias)
        msa = msa + self.msa_ffn(msa)

        # Pairwise updates
        pair = self.tri_out(pair)
        pair = self.tri_in(pair)
        pair = self.pair_row_attn(pair)
        pair = self.pair_col_attn(pair)
        pair = pair + self.pair_ffn(pair)

        return msa, pair


# ═══════════════════════════════════════════════════════════════════════════════
#  Evoformer Stack  (full module)
# ═══════════════════════════════════════════════════════════════════════════════
class EvoformerStack(nn.Module):
    """
    AlphaFold Evoformer adapted for DNA sequence analysis.

    Takes single-sequence representations and optional pairwise features,
    then iteratively refines both through Evoformer blocks.
    """
    def __init__(
        self,
        seq_dim:   int = 256,
        pair_dim:  int = 128,
        n_blocks:  int = 4,
        n_heads:   int = 4,
        max_len:   int = 256,
    ):
        super().__init__()
        self.seq_dim  = seq_dim
        self.pair_dim = pair_dim
        self.max_len  = max_len

        # Project single sequence to pairwise representation
        self.pair_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )

        self.blocks = nn.ModuleList([
            EvoformerBlock(seq_dim, pair_dim, n_heads)
            for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(seq_dim)

    def _init_pair_repr(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialise pairwise representation from sequence embeddings.
        x: (B, L, D) → pair: (B, L, L, pair_dim)
        """
        B, L, D = x.shape
        # Outer concatenation
        x_i = x.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, D)
        x_j = x.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, D)
        pair_input = torch.cat([x_i, x_j], dim=-1)   # (B, L, L, 2D)
        return self.pair_proj(pair_input)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, D) sequence embeddings

        Returns:
          msa:  (B, 1, L, D)  refined sequence (as single-row MSA)
          pair: (B, L, L, pair_dim) pairwise representation
        """
        B, L, D = x.shape
        # Treat single sequence as 1-row MSA
        msa  = x.unsqueeze(1)  # (B, 1, L, D)
        pair = self._init_pair_repr(x)

        for block in self.blocks:
            msa, pair = block(msa, pair)

        msa = self.output_norm(msa.squeeze(1)).unsqueeze(1)
        return msa, pair
