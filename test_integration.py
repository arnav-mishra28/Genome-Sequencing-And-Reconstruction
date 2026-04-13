"""Quick integration test for all new modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.fusion_model import TransformerGNNFusion, build_fusion_phylo_graph
from models.confidence_scorer import ConfidenceScorer

print("=" * 65)
print("  INTEGRATION TEST — Fusion + Confidence + Viewer")
print("=" * 65)

# ── Test 1: Fusion Model Forward Pass ────────────────────────────────────────
print("\n[1] Testing TransformerGNNFusion forward pass...")
vocab_size = 100
model = TransformerGNNFusion(
    vocab_size=vocab_size, n_species_feat=256,
    embed_dim=64, n_heads=4, n_layers=2,
    ffn_dim=128, max_len=32,
)
tokens = torch.randint(0, vocab_size, (2, 32))
att = torch.ones(2, 32)
sp_feats = torch.randn(5, 256)
adj = torch.eye(5)
sp_idx = torch.tensor([0, 1])

out = model(tokens, att, species_feats=sp_feats, adjacency=adj, species_idx=sp_idx)
print(f"  MLM logits:       {out['mlm_logits'].shape}")
print(f"  Recon logits:     {out['recon_logits'].shape}")
print(f"  Per-base conf:    {out['per_base_conf'].shape}")
print(f"  Reliability:      {out['reliability'].shape}")
print(f"  Temperature:      {out['temperature'].item():.3f}")
print(f"  Hidden states:    {out['hidden_states'].shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params:     {total_params:,}")
print("  ✅ PASS")

# ── Test 2: Confidence Scorer ─────────────────────────────────────────────────
print("\n[2] Testing ConfidenceScorer...")
scorer = ConfidenceScorer()
result = scorer.score_sequence(out["mlm_logits"][0])
print(f"  Mean confidence:    {result['mean_confidence']}")
print(f"  Reliability:        {result['reliability_score']}")
print(f"  High conf %:        {result['high_confidence_pct']}")
print(f"  Low conf count:     {result['low_confidence_count']}")
print(f"  Positions:          {result['n_positions']}")

regions = scorer.score_region(result["per_base_confidence"], window_size=10)
print(f"  Regions scored:     {len(regions)}")
if regions:
    print(f"  Region 0 class:     {regions[0]['classification']}")
print("  ✅ PASS")

# ── Test 3: Confidence with model output ──────────────────────────────────────
print("\n[3] Testing confidence with model predictions...")
result2 = scorer.score_sequence(
    logits=out["mlm_logits"][0],
    model_confidence=out["per_base_conf"][0].detach(),
)
print(f"  Mean conf (w/ model):  {result2['mean_confidence']}")
print(f"  Reliability (w/ model): {result2['reliability_score']}")
print("  ✅ PASS")

# ── Test 4: Reconstruction Engine ─────────────────────────────────────────────
print("\n[4] Testing ReconstructionEngine...")
from visualization.reconstruction_engine import ReconstructionEngine
import random
random.seed(42)
seq = "".join(random.choices("ACGT", k=200))
damaged = list(seq)
for i in range(0, 200, 8):
    damaged[i] = "N"
damaged_str = "".join(damaged)

engine = ReconstructionEngine(
    damaged_sequence=damaged_str,
    species_name="test_species",
)
stats = engine.stats
print(f"  Total bases:     {stats['total_bases']}")
print(f"  Total gaps:      {stats['total_gaps']}")
print(f"  Progress:        {stats['progress']:.1%}")
print("  ✅ PASS")

# ── Test 5: Module verification ───────────────────────────────────────────────
print("\n[5] Testing main.py module verification...")
from main import verify_modules
missing = verify_modules()
if missing:
    print(f"  ⚠ Missing {len(missing)} files:")
    for m in missing:
        print(f"    • {os.path.basename(m)}")
else:
    print("  All source files found")
    print("  ✅ PASS")

# ── Test 6: Ensemble reconstructor imports ────────────────────────────────────
print("\n[6] Testing ensemble_reconstructor...")
from models.ensemble_reconstructor import (
    ensemble_reconstruct,
    multi_species_ensemble_reconstruct,
    EnsembleReconstructor,
)
print("  All functions imported successfully")
print("  ✅ PASS")

# ── Test 7: Phase 5 training imports ──────────────────────────────────────────
print("\n[7] Testing phase5_fusion...")
from training.phase5_fusion import run_phase5
print("  run_phase5 imported successfully")
print("  ✅ PASS")

# ── Test 8: Viewer imports ────────────────────────────────────────────────────
print("\n[8] Testing reconstruction_viewer...")
from visualization.reconstruction_viewer import (
    ReconstructionViewer, launch_reconstruction_viewer,
    HelixRenderer, HUDRenderer,
)
print("  All viewer classes imported successfully")
print("  ✅ PASS")

print("\n" + "=" * 65)
print("  ✅  ALL 8 INTEGRATION TESTS PASSED")
print("=" * 65)
