"""
reconstruction_engine.py
========================
Step-by-step DNA reconstruction engine that drives the live 3D viewer.

Runs the trained models (Transformer+GNN Fusion, Denoising AE, LSTM)
one k-mer window at a time, yielding ReconstructionEvent objects
that the viewer animates.

This is the "brain" behind the live viewer — it feeds reconstruction
events one at a time so the user can watch the AI think.
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass, field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import DEVICE, MAX_SEQ_LEN


# ═══════════════════════════════════════════════════════════════════════════════
#  Reconstruction Event
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ReconstructionEvent:
    """A single reconstruction step result."""
    position:       int
    original_base:  str            # what was there (damaged)
    predicted_base: str            # what the AI predicts
    confidence:     float          # per-base confidence [0, 1]
    model_source:   str            # which model made this prediction
    phase:          str            # current reconstruction phase
    step:           int      = 0
    timestamp:      float    = 0.0
    is_gap_fill:    bool     = False
    is_correction:  bool     = False
    window_start:   int      = 0
    window_end:     int      = 0
    reliability:    float    = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Reconstruction Engine
# ═══════════════════════════════════════════════════════════════════════════════
class ReconstructionEngine:
    """
    Drives step-by-step DNA reconstruction using trained models.
    
    Usage:
        engine = ReconstructionEngine(
            damaged_sequence="ACGTNNNACGT...",
            models={"fusion": fusion_model, ...},
            vocab=vocab,
            species_name="neanderthal_mtDNA",
        )
        
        for event in engine.reconstruct():
            viewer.apply_event(event)
    """

    def __init__(
        self,
        damaged_sequence: str,
        species_name:     str         = "specimen",
        models:           Dict        = None,
        vocab:            Dict        = None,
        species_feats:    torch.Tensor = None,
        adjacency:        torch.Tensor = None,
        species_idx:      int         = 0,
        species_names:    List[str]   = None,
        device:           torch.device = None,
        k:                int         = 6,
    ):
        self.original_damaged = damaged_sequence.upper()
        self.current_seq      = list(self.original_damaged)
        self.species_name     = species_name
        self.models           = models or {}
        self.vocab            = vocab or {}
        self.species_feats    = species_feats
        self.adjacency        = adjacency
        self.species_idx      = species_idx
        self.species_names    = species_names or [species_name]
        self.device           = device or DEVICE
        self.k                = k

        # State
        self.events:    List[ReconstructionEvent] = []
        self.step_count = 0
        self.finished   = False
        self.paused     = False
        self.speed      = 10.0

        # Confidence tracking
        self.confidences = [0.0 if b == "N" else 0.8 for b in self.current_seq]

        # Pre-compute damage positions
        self.damage_positions = [
            i for i, b in enumerate(self.current_seq) if b == "N"
        ]
        self.total_gaps = len(self.damage_positions)

    @property
    def stats(self) -> Dict:
        seq = "".join(self.current_seq)
        gaps = seq.count("N")
        filled = self.total_gaps - gaps
        progress = filled / max(1, self.total_gaps)
        mean_conf = float(np.mean(self.confidences)) if self.confidences else 0

        return {
            "species":       self.species_name,
            "total_bases":   len(self.current_seq),
            "total_gaps":    self.total_gaps,
            "gaps_remaining": gaps,
            "gaps_filled":   filled,
            "progress":      round(progress, 4),
            "steps":         self.step_count,
            "mean_confidence": round(mean_conf, 4),
            "finished":      self.finished,
        }

    def reconstruct(self) -> Generator[ReconstructionEvent, None, None]:
        """
        Generator that yields reconstruction events one at a time.
        Runs through three phases:
          1. Denoising Autoencoder — repair corrupted bases
          2. Transformer+GNN Fusion — fill gaps with evolutionary context
          3. LSTM — extend/complete sequence edges
        """
        # ════════════════════════════════════════════════════════════════════
        #  Phase 1: Denoising Autoencoder
        # ════════════════════════════════════════════════════════════════════
        ae_model = self.models.get("ae")
        if ae_model is not None:
            yield from self._run_ae_phase(ae_model)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 2: Transformer + GNN Fusion
        # ════════════════════════════════════════════════════════════════════
        fusion_model = self.models.get("fusion")
        if fusion_model is not None:
            yield from self._run_fusion_phase(fusion_model)

        # ════════════════════════════════════════════════════════════════════
        #  Phase 3: BERT fill (fallback for remaining gaps)
        # ════════════════════════════════════════════════════════════════════
        bert_model = self.models.get("bert")
        if bert_model is not None:
            yield from self._run_bert_phase(bert_model)

        self.finished = True

    def _run_ae_phase(self, ae_model) -> Generator[ReconstructionEvent, None, None]:
        """Phase 1: Run denoising autoencoder position by position."""
        from models.denoising_autoencoder import SEQ_LEN
        from preprocessing.encoding import one_hot_encode

        ae_model.eval()
        ae_model.to(self.device)

        INT2BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
        seq = "".join(self.current_seq)

        for chunk_start in range(0, len(seq), SEQ_LEN):
            chunk = seq[chunk_start: chunk_start + SEQ_LEN]
            padded = chunk.ljust(SEQ_LEN, "N").upper()

            oh = one_hot_encode(padded)
            enc5 = np.zeros((SEQ_LEN, 5), dtype=np.float32)
            enc5[:, :4] = oh
            chars = np.array(list(padded), dtype=str)
            enc5[:, 4] = (chars == "N").astype(np.float32)

            t = torch.from_numpy(enc5.T.copy()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = ae_model(t)
                probs = torch.softmax(logits[0], dim=0)

            pred = probs[:4].argmax(dim=0).cpu().numpy()
            conf = probs[:4].max(dim=0).values.cpu().numpy()

            for i in range(min(len(chunk), SEQ_LEN)):
                global_pos = chunk_start + i
                if global_pos >= len(self.current_seq):
                    break

                orig = self.current_seq[global_pos]
                predicted = INT2BASE[int(pred[i])]
                c = float(conf[i])

                if orig == "N" or (orig != predicted and c > 0.7):
                    self.current_seq[global_pos] = predicted
                    self.confidences[global_pos] = round(c, 4)

                    event = ReconstructionEvent(
                        position=global_pos,
                        original_base=orig,
                        predicted_base=predicted,
                        confidence=round(c, 4),
                        model_source="DenoisingAE",
                        phase="denoising",
                        step=self.step_count,
                        timestamp=time.time(),
                        is_gap_fill=(orig == "N"),
                        is_correction=(orig != "N" and orig != predicted),
                        window_start=chunk_start,
                        window_end=chunk_start + SEQ_LEN,
                    )
                    self.events.append(event)
                    self.step_count += 1
                    yield event

    def _run_fusion_phase(self, fusion_model) -> Generator[ReconstructionEvent, None, None]:
        """Phase 2: Run Transformer+GNN fusion model window by window."""
        from preprocessing.encoding import encode_kmer_sequence

        fusion_model.eval()
        fusion_model.to(self.device)

        if not self.vocab:
            return

        inv_vocab = {v: k_str for k_str, v in self.vocab.items()}
        mask_id = self.vocab.get("[MASK]", 0)
        cls_id  = self.vocab.get("[CLS]", 1)
        pad_id  = self.vocab.get("[PAD]", 0)
        max_len = min(fusion_model.max_len, MAX_SEQ_LEN)
        k = self.k

        chunk_size = (max_len - 1) * k
        seq = self.current_seq

        sp_feats = self.species_feats
        adj = self.adjacency
        if sp_feats is not None:
            sp_feats = sp_feats.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)

        for start in range(0, len(seq), chunk_size):
            end = min(start + chunk_size, len(seq))
            window_seq = "".join(seq[start:end])

            # Check if there are any N's to fill
            if "N" not in window_seq:
                continue

            kmers = encode_kmer_sequence(window_seq, self.vocab, k)
            tokens = np.concatenate([[cls_id], kmers]).astype(np.int32)
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), constant_values=pad_id)

            # Mask positions with N
            for j in range(1, max_len - pad_len):
                kmer_start = start + (j - 1) * k
                kmer_end = kmer_start + k
                if any(seq[p] == "N"
                       for p in range(kmer_start, min(kmer_end, len(seq)))):
                    tokens[j] = mask_id

            t_tensor = torch.tensor(
                tokens, dtype=torch.long
            ).unsqueeze(0).to(self.device)
            att = torch.tensor(
                (tokens != pad_id).astype(np.float32)
            ).unsqueeze(0).to(self.device)

            sp_idx = torch.tensor(
                [self.species_idx], dtype=torch.long
            ).to(self.device) if sp_feats is not None else None

            with torch.no_grad():
                outputs = fusion_model(
                    t_tensor, att,
                    species_feats=sp_feats,
                    adjacency=adj,
                    species_idx=sp_idx,
                )
                logits = outputs["mlm_logits"]
                probs = torch.softmax(logits[0], dim=-1)
                base_conf = outputs["per_base_conf"][0]
                reliability = float(outputs["reliability"][0].cpu())

            for j in range(1, max_len - pad_len):
                if int(t_tensor[0, j].cpu()) != mask_id:
                    continue

                top_id = int(probs[j].argmax().cpu())
                top_kmer = inv_vocab.get(top_id, "A" * k)
                conf = float(base_conf[j].cpu())

                kmer_start = start + (j - 1) * k
                for offset, base in enumerate(top_kmer):
                    pos = kmer_start + offset
                    if pos < len(seq) and seq[pos] == "N" and base in "ACGT":
                        old = seq[pos]
                        self.current_seq[pos] = base
                        self.confidences[pos] = round(conf, 4)

                        event = ReconstructionEvent(
                            position=pos,
                            original_base=old,
                            predicted_base=base,
                            confidence=round(conf, 4),
                            model_source="Fusion(T+GNN)",
                            phase="fusion_reconstruction",
                            step=self.step_count,
                            timestamp=time.time(),
                            is_gap_fill=True,
                            window_start=start,
                            window_end=end,
                            reliability=round(reliability, 4),
                        )
                        self.events.append(event)
                        self.step_count += 1
                        yield event

    def _run_bert_phase(self, bert_model) -> Generator[ReconstructionEvent, None, None]:
        """Phase 3: BERT fallback for remaining gaps."""
        from preprocessing.encoding import encode_kmer_sequence

        bert_model.eval()
        bert_model.to(self.device)

        if not self.vocab:
            return

        inv_vocab = {v: k_str for k_str, v in self.vocab.items()}
        mask_id = self.vocab.get("[MASK]", 0)
        cls_id  = self.vocab.get("[CLS]", 1)
        pad_id  = self.vocab.get("[PAD]", 0)
        max_len = bert_model.max_len
        k = self.k

        chunk_size = (max_len - 1) * k
        seq = self.current_seq

        for start in range(0, len(seq), chunk_size):
            end = min(start + chunk_size, len(seq))
            window_seq = "".join(seq[start:end])

            if "N" not in window_seq:
                continue

            kmers = encode_kmer_sequence(window_seq, self.vocab, k)
            tokens = np.concatenate([[cls_id], kmers]).astype(np.int32)
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), constant_values=pad_id)

            for j in range(1, max_len - pad_len):
                kmer_start = start + (j - 1) * k
                kmer_end = kmer_start + k
                if any(seq[p] == "N"
                       for p in range(kmer_start, min(kmer_end, len(seq)))):
                    tokens[j] = mask_id

            t_tensor = torch.tensor(
                tokens, dtype=torch.long
            ).unsqueeze(0).to(self.device)
            att = torch.tensor(
                (tokens != pad_id).astype(np.float32)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = bert_model(t_tensor, att)
                probs = torch.softmax(logits[0], dim=-1)

            for j in range(1, max_len - pad_len):
                if int(t_tensor[0, j].cpu()) != mask_id:
                    continue

                top_id = int(probs[j].argmax().cpu())
                top_kmer = inv_vocab.get(top_id, "A" * k)
                conf = float(probs[j].max().cpu())

                kmer_start = start + (j - 1) * k
                for offset, base in enumerate(top_kmer):
                    pos = kmer_start + offset
                    if pos < len(seq) and seq[pos] == "N" and base in "ACGT":
                        old = seq[pos]
                        self.current_seq[pos] = base
                        self.confidences[pos] = round(conf, 4)

                        event = ReconstructionEvent(
                            position=pos,
                            original_base=old,
                            predicted_base=base,
                            confidence=round(conf, 4),
                            model_source="DNABERT-2",
                            phase="bert_fill",
                            step=self.step_count,
                            timestamp=time.time(),
                            is_gap_fill=True,
                            window_start=start,
                            window_end=end,
                        )
                        self.events.append(event)
                        self.step_count += 1
                        yield event

    def get_current_sequence(self) -> str:
        return "".join(self.current_seq)

    def get_confidence_array(self) -> List[float]:
        return self.confidences.copy()

    # ── Control ───────────────────────────────────────────────────────────────
    def toggle_pause(self):
        self.paused = not self.paused

    def speed_up(self):
        self.speed = min(self.speed * 1.5, 200)

    def slow_down(self):
        self.speed = max(self.speed / 1.5, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  Create engine with pre-trained models
# ═══════════════════════════════════════════════════════════════════════════════
def create_reconstruction_engine(
    species_name:   str,
    damaged_seq:    str     = None,
    max_bases:      int     = 2000,
) -> ReconstructionEngine:
    """
    Create a ReconstructionEngine with trained models loaded from checkpoints.
    Falls back to synthetic sequence + untrained models for demo mode.
    """
    import random
    from config.settings import MODEL_DIR, SEQ_DIR

    # ── Load or generate sequence ─────────────────────────────────────────────
    if damaged_seq is None:
        meta_path = os.path.join(SEQ_DIR, "metadata.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                metadata = json.load(f)
            if species_name in metadata:
                fasta_path = metadata[species_name].get("path", "")
                if os.path.exists(fasta_path):
                    from data.fetch_sequences import load_fasta
                    recs = load_fasta(fasta_path)
                    seq = next(iter(recs.values()))[:max_bases]
                    # Simulate some damage
                    damaged = list(seq)
                    rng = random.Random(42)
                    for _ in range(len(damaged) // 10):
                        pos = rng.randint(0, len(damaged) - 1)
                        damaged[pos] = "N"
                    damaged_seq = "".join(damaged)

        if damaged_seq is None:
            # Synthetic fallback
            rng = random.Random(42)
            bases = random.choices("ACGT", k=max_bases)
            for _ in range(max_bases // 8):
                pos = rng.randint(0, max_bases - 1)
                bases[pos] = "N"
            damaged_seq = "".join(bases)
            print(f"  ⚠ Using synthetic damaged sequence ({max_bases} bp)")

    # ── Load models ───────────────────────────────────────────────────────────
    models = {}

    # Try loading AE
    ae_path = os.path.join(MODEL_DIR, "denoising_ae.pt")
    if os.path.exists(ae_path):
        from models.denoising_autoencoder import DenoisingAutoencoder
        ae = DenoisingAutoencoder()
        ae.load_state_dict(torch.load(ae_path, map_location="cpu",
                                       weights_only=True))
        models["ae"] = ae
        print(f"  ✅ Loaded AE from {ae_path}")

    # Try loading BERT
    bert_path = os.path.join(MODEL_DIR, "dnabert2.pt")
    if os.path.exists(bert_path):
        ckpt = torch.load(bert_path, map_location="cpu", weights_only=True)
        from models.dnabert2_transformer import DNABERT2Model
        bert = DNABERT2Model(vocab_size=ckpt.get("vocab_size", 4102))
        bert.load_state_dict(ckpt["model"])
        models["bert"] = bert
        print(f"  ✅ Loaded BERT from {bert_path}")

    # Try loading Fusion
    fusion_path = os.path.join(MODEL_DIR, "fusion.pt")
    if os.path.exists(fusion_path):
        ckpt = torch.load(fusion_path, map_location="cpu", weights_only=True)
        from models.fusion_model import TransformerGNNFusion
        fusion = TransformerGNNFusion(
            vocab_size=ckpt.get("vocab_size", 4102),
        )
        fusion.load_state_dict(ckpt["model"])
        models["fusion"] = fusion
        print(f"  ✅ Loaded Fusion from {fusion_path}")

    # ── Load vocab ────────────────────────────────────────────────────────────
    from config.settings import RESULTS_DIR
    vocab_path = os.path.join(RESULTS_DIR, "kmer_vocab.json")
    vocab = {}
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path) as f:
            vocab = json.load(f)
    else:
        # Build minimal vocab for demo
        from preprocessing.encoding import build_kmer_vocab
        vocab = build_kmer_vocab([damaged_seq], k=6)

    return ReconstructionEngine(
        damaged_sequence=damaged_seq,
        species_name=species_name,
        models=models,
        vocab=vocab,
    )
