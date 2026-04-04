## 📄 `models/gnn_phylogenetic.py`

"""
gnn_phylogenetic.py
===================
Graph Neural Network over a phylogenetic tree.
Nodes  : species (embedding = mean k-mer frequency vector)
Edges  : evolutionary distance (from pairwise alignment scores)
Task   : Refine per-species sequence embeddings using graph convolution,
         then decode a "consensus healthy sequence" for reconstruction.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from itertools import product

BASE_DIR   = r"D:\Genome Sequencing And Reconstruction"
MODEL_DIR  = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

# Phylogenetic distances (millions of years) — from published literature
PHYLO_DISTANCES = {
    ("neanderthal_mtDNA",   "human_mtDNA"):        0.6,
    ("mammoth_mtDNA",       "elephant_mtDNA"):      5.0,
    ("woolly_rhino",        "elephant_mtDNA"):     58.0,
    ("cave_bear_mtDNA",     "gray_wolf_mtDNA"):    40.0,
    ("thylacine_mtDNA",     "gray_wolf_mtDNA"):    160.0,
    ("passenger_pigeon",    "rock_pigeon_mtDNA"):   30.0,
    ("dodo_partial",        "rock_pigeon_mtDNA"):   25.0,
    ("saber_tooth_cat",     "gray_wolf_mtDNA"):     90.0,
    ("human_mtDNA",         "elephant_mtDNA"):      90.0,
}

EMBED_DIM = 256


def kmer_frequency_vector(sequence: str, k: int = 4) -> np.ndarray:
    """Compute normalized 4^k frequency vector."""
    bases = "ACGT"

    # Generate all possible k-mers
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_index = {kmer: i for i, kmer in enumerate(all_kmers)}

    vec = np.zeros(len(all_kmers), dtype=np.float32)

    seq = sequence.upper().replace("N", "")

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1

    # Normalize
    if vec.sum() > 0:
        vec /= vec.sum()

    return vec


class GNNLayer(nn.Module):
    """Simple Graph Convolutional layer (mean aggregation)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.bn      = nn.BatchNorm1d(out_dim)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)   adj: (N, N) weighted adjacency
        deg  = adj.sum(dim=1, keepdim=True).clamp(min=1e-9)
        agg  = (adj @ x) / deg
        out  = self.W_self(x) + self.W_neigh(agg)
        return self.act(self.bn(out))


class PhyloGNN(nn.Module):
    def __init__(self, node_feat_dim: int, hidden: int = EMBED_DIM,
                 out_dim: int = EMBED_DIM):
        super().__init__()
        self.proj  = nn.Linear(node_feat_dim, hidden)
        self.gnn1  = GNNLayer(hidden, hidden)
        self.gnn2  = GNNLayer(hidden, hidden)
        self.gnn3  = GNNLayer(hidden, out_dim)
        self.drop  = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.drop(torch.relu(self.proj(x)))
        x = self.gnn1(x, adj)
        x = self.gnn2(x, adj)
        x = self.gnn3(x, adj)
        return x     # (N, out_dim) — species embeddings


def build_phylo_graph(
    species_names: List[str],
    sequences:     Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      node_features  : (N, 4^4=256) k-mer freq vectors
      adj_matrix     : (N, N) inverse-distance weights
    """
    N = len(species_names)
    node_feats = np.zeros((N, 4**4), dtype=np.float32)
    adj        = np.zeros((N, N), dtype=np.float32)

    for i, sp in enumerate(species_names):
        seq = sequences.get(sp, "ACGT" * 100)
        node_feats[i] = kmer_frequency_vector(seq, k=4)

    for (sp1, sp2), dist in PHYLO_DISTANCES.items():
        if sp1 in species_names and sp2 in species_names:
            i = species_names.index(sp1)
            j = species_names.index(sp2)
            weight = 1.0 / (1.0 + dist)
            adj[i, j] = adj[j, i] = weight

    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    return node_feats, adj


def train_phylo_gnn(
    species_names: List[str],
    sequences:     Dict[str, str],
    epochs: int = 50,
    lr: float = 1e-3,
) -> Tuple[PhyloGNN, torch.Tensor]:
    """Train GNN in self-supervised mode (predict masked node features)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GNN] Training Phylogenetic GNN on {device}")

    feats, adj = build_phylo_graph(species_names, sequences)
    X   = torch.tensor(feats, dtype=torch.float).to(device)
    A   = torch.tensor(adj,   dtype=torch.float).to(device)

    model = PhyloGNN(node_feat_dim=feats.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for epoch in range(1, epochs+1):
        # Randomly mask some nodes
        mask = torch.rand(X.shape[0]) < 0.3
        X_masked = X.clone()
        X_masked[mask] = 0.

        embeddings = model(X_masked, A)
        # Reconstruct node features
        recon = nn.Linear(EMBED_DIM, feats.shape[1]).to(device)(embeddings)
        loss  = F.mse_loss(recon[mask], X[mask])
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d}  loss={loss.item():.6f}")
        history.append({"epoch": epoch, "loss": float(loss.item())})

    # Final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(X, A)

    ckpt = os.path.join(MODEL_DIR, "phylo_gnn.pt")
    torch.save(model.state_dict(), ckpt)
    with open(os.path.join(MODEL_DIR, "gnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[GNN] Saved → {ckpt}")
    return model, final_embeddings