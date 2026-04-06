"""
dataset_builder.py
==================
PyTorch Dataset classes for the 4-phase training curriculum:
  Phase 1 — PretrainDataset      (masked prediction on modern genomes)
  Phase 2 — CorruptionDataset    (corrupted → original reconstruction)
  Phase 3 — EvolutionDataset     (sequences with phylogenetic graph context)
  Phase 4 — AncientDNADataset    (real ancient DNA fragments → reference)
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import (
    MAX_SEQ_LEN, MASK_PROB, MAX_SAMPLES,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — PretrainDataset  (masked language modeling)
# ═══════════════════════════════════════════════════════════════════════════════
class PretrainDataset(Dataset):
    """
    Tokenises modern genome sequences using a vocabulary and applies
    masked-language-model (MLM) corruption (BERT-style 15% masking).
    """
    def __init__(
        self,
        sequences:   List[str],
        vocab:       Dict[str, int],
        k:           int   = 6,
        max_len:     int   = MAX_SEQ_LEN,
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
            chunk = max_len - 1

            for i in range(0, len(ids), chunk):
                if len(self.samples) >= max_samples:
                    break
                window = ids[i : i + chunk]
                tokens = np.concatenate([[cls_id], window]).astype(np.int32)
                tokens = tokens[:max_len]

                pad_len = max_len - len(tokens)
                tokens  = np.pad(tokens, (0, pad_len), constant_values=pad_id)

                labels  = np.full(max_len, -100, dtype=np.int32)
                seq_len = max_len - pad_len

                for j in range(1, seq_len):
                    if random.random() < mask_prob:
                        labels[j] = int(tokens[j])
                        r = random.random()
                        if r < 0.80:
                            tokens[j] = mask_id
                        elif r < 0.90:
                            tokens[j] = random.randint(
                                5, max(5, len(vocab) - 1)
                            )

                att_mask = (tokens != pad_id).astype(np.float32)
                self.samples.append((
                    tokens.copy(), labels.copy(), att_mask.copy(),
                ))

        if len(self.samples) == 0:
            print("  [Pretrain WARN] No samples — synthetic fallback.")
            for _ in range(100):
                t = np.random.randint(0, min(len(vocab), 100),
                                      max_len).astype(np.int32)
                l = np.full(max_len, -100, dtype=np.int32)
                a = np.ones(max_len, dtype=np.float32)
                self.samples.append((t, l, a))

        print(f"  [Pretrain] Dataset size: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        t, l, a = self.samples[idx]
        return (
            torch.tensor(t, dtype=torch.long),
            torch.tensor(l, dtype=torch.long),
            torch.tensor(a, dtype=torch.float),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — CorruptionDataset  (damaged → original pairs)
# ═══════════════════════════════════════════════════════════════════════════════
class CorruptionDataset(Dataset):
    """
    Pairs of (corrupted_sequence, original_sequence) encoded as one-hot
    5-channel tensors (ACGTN).  Used to train the denoiser.
    """
    def __init__(
        self,
        clean_seqs:  List[str],
        noisy_seqs:  List[str],
        seq_len:     int = 128,
        max_samples: int = MAX_SAMPLES,
    ):
        from preprocessing.encoding import one_hot_encode

        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for c, n in zip(clean_seqs, noisy_seqs):
            if len(self.pairs) >= max_samples:
                break

            c_str = c[:seq_len].ljust(seq_len, "A").upper()
            n_str = n[:seq_len].ljust(seq_len, "N").upper()

            cl_oh = one_hot_encode(c_str)
            no_oh = one_hot_encode(n_str)

            cl5 = np.zeros((seq_len, 5), dtype=np.float32)
            cl5[:, :4] = cl_oh

            no5 = np.zeros((seq_len, 5), dtype=np.float32)
            no5[:, :4] = no_oh
            n_chars     = np.array(list(n_str), dtype=str)
            no5[:, 4]   = (n_chars == "N").astype(np.float32)

            self.pairs.append((
                torch.from_numpy(no5.T.copy()),
                torch.from_numpy(cl5.T.copy()),
            ))

        if len(self.pairs) == 0:
            print("  [Corruption WARN] No pairs — synthetic fallback.")
            for _ in range(100):
                self.pairs.append((
                    torch.rand(5, seq_len),
                    torch.rand(5, seq_len),
                ))

        print(f"  [Corruption] Dataset size: {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — EvolutionDataset  (sequences + phylogenetic context)
# ═══════════════════════════════════════════════════════════════════════════════
class EvolutionDataset(Dataset):
    """
    Each sample is a tuple of:
      - tokenised sequence chunk
      - species index (int)  → maps to a node in the phylo graph
      - GC content of the chunk (float)  → for biological constraint loss
    """
    def __init__(
        self,
        species_sequences: Dict[str, str],
        species_names:     List[str],
        vocab:             Dict[str, int],
        k:                 int = 6,
        max_len:           int = MAX_SEQ_LEN,
        max_samples:       int = MAX_SAMPLES,
    ):
        from preprocessing.encoding import encode_kmer_sequence

        self.samples: List[Tuple[np.ndarray, int, float]] = []
        pad_id = vocab["[PAD]"]
        cls_id = vocab["[CLS]"]

        for sp_name in species_names:
            seq = species_sequences.get(sp_name, "")
            if not seq:
                continue
            sp_idx = species_names.index(sp_name)
            ids    = encode_kmer_sequence(seq, vocab, k)
            chunk  = max_len - 1

            gc = (seq.upper().count("G") + seq.upper().count("C")) / max(1, len(seq))

            for i in range(0, len(ids), chunk):
                if len(self.samples) >= max_samples:
                    break
                window = ids[i : i + chunk]
                tokens = np.concatenate([[cls_id], window]).astype(np.int32)
                tokens = tokens[:max_len]
                pad_len = max_len - len(tokens)
                tokens  = np.pad(tokens, (0, pad_len), constant_values=pad_id)
                self.samples.append((tokens.copy(), sp_idx, gc))

            if len(self.samples) >= max_samples:
                break

        if len(self.samples) == 0:
            print("  [Evolution WARN] No samples — synthetic fallback.")
            for _ in range(50):
                t = np.random.randint(0, 100, max_len).astype(np.int32)
                self.samples.append((t, 0, 0.5))

        print(f"  [Evolution] Dataset size: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        t, sp_idx, gc = self.samples[idx]
        return (
            torch.tensor(t, dtype=torch.long),
            torch.tensor(sp_idx, dtype=torch.long),
            torch.tensor(gc, dtype=torch.float),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — AncientDNADataset  (real fragments → reference alignment)
# ═══════════════════════════════════════════════════════════════════════════════
class AncientDNADataset(Dataset):
    """
    Pairs of (ancient_fragment, reference_region) for fine-tuning.
    The reference region is the corresponding segment from the modern
    relative at the mapped position.
    """
    def __init__(
        self,
        fragments:    List[Dict],
        reference:    str,
        vocab:        Dict[str, int],
        k:            int = 6,
        max_len:      int = MAX_SEQ_LEN,
        max_samples:  int = MAX_SAMPLES,
    ):
        from preprocessing.encoding import encode_kmer_sequence

        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        pad_id  = vocab["[PAD]"]
        mask_id = vocab["[MASK]"]
        cls_id  = vocab["[CLS]"]

        ref_upper = reference.upper()

        for frag in fragments:
            if len(self.samples) >= max_samples:
                break
            frag_seq = frag["seq"].upper()
            start    = frag.get("start", 0)
            end      = min(start + len(frag_seq), len(ref_upper))
            ref_seg  = ref_upper[start:end]

            if len(frag_seq) < k + 2 or len(ref_seg) < k + 2:
                continue

            # Encode fragment (mask N positions)
            frag_ids = encode_kmer_sequence(frag_seq, vocab, k)
            ref_ids  = encode_kmer_sequence(ref_seg, vocab, k)

            # Pad / truncate
            frag_tok = np.concatenate([[cls_id], frag_ids]).astype(np.int32)[:max_len]
            ref_tok  = np.concatenate([[cls_id], ref_ids]).astype(np.int32)[:max_len]

            pad_f = max_len - len(frag_tok)
            pad_r = max_len - len(ref_tok)
            frag_tok = np.pad(frag_tok, (0, pad_f), constant_values=pad_id)
            ref_tok  = np.pad(ref_tok,  (0, pad_r), constant_values=pad_id)

            # Mask positions with N in fragment
            for j in range(1, max_len - pad_f):
                kmer_start = (j - 1) * k
                kmer_end   = kmer_start + k
                if any(frag_seq[p] == "N"
                       for p in range(kmer_start, min(kmer_end, len(frag_seq)))):
                    frag_tok[j] = mask_id

            att_mask = (frag_tok != pad_id).astype(np.float32)
            self.samples.append((
                frag_tok.copy(), ref_tok.copy(), att_mask.copy(),
            ))

        if len(self.samples) == 0:
            print("  [AncientDNA WARN] No samples — synthetic fallback.")
            for _ in range(50):
                t = np.random.randint(0, 100, max_len).astype(np.int32)
                r = np.random.randint(0, 100, max_len).astype(np.int32)
                a = np.ones(max_len, dtype=np.float32)
                self.samples.append((t, r, a))

        print(f"  [AncientDNA] Dataset size: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        f, r, a = self.samples[idx]
        return (
            torch.tensor(f, dtype=torch.long),
            torch.tensor(r, dtype=torch.long),
            torch.tensor(a, dtype=torch.float),
        )
