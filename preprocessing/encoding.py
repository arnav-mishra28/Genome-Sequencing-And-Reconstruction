"""
encoding.py
===========
DNA → numerical representations:
  1. One-hot encoding   (A=1000, C=0100, G=0010, T=0001, N=0000)
  2. Integer encoding   (A=0, C=1, G=2, T=3, N=4)
  3. k-mer tokenization (for transformer models)
  4. Physicochemical property encoding
  5. BPE tokenization   (DNABERT-2 inspired)
"""

import os
import sys
import json
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

ONE_HOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

PHYSCHEM = {
    "A": [1, 2, 135.13, 1],
    "C": [1, 3, 111.10, 0],
    "G": [2, 3, 151.13, 1],
    "T": [1, 2, 126.11, 0],
    "N": [0, 0, 0.0,    0],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Basic encodings
# ═══════════════════════════════════════════════════════════════════════════════
def integer_encode(sequence: str) -> np.ndarray:
    return np.array([BASE_TO_INT.get(b, 4)
                     for b in sequence.upper()], dtype=np.int32)


def one_hot_encode(sequence: str) -> np.ndarray:
    return np.array([ONE_HOT.get(b, [0, 0, 0, 0])
                     for b in sequence.upper()], dtype=np.float32)


def physchem_encode(sequence: str) -> np.ndarray:
    return np.array([PHYSCHEM.get(b, [0, 0, 0, 0])
                     for b in sequence.upper()], dtype=np.float32)


def decode_integer(arr: np.ndarray) -> str:
    return "".join(INT_TO_BASE.get(int(x), "N") for x in arr)


# ═══════════════════════════════════════════════════════════════════════════════
#  k-mer tokenization (classic DNABERT-1 style)
# ═══════════════════════════════════════════════════════════════════════════════
def kmer_tokenize(sequence: str, k: int = 6) -> List[str]:
    seq = sequence.upper()
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def build_kmer_vocab(sequences: List[str], k: int = 6) -> Dict[str, int]:
    vocab = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2, "[CLS]": 3, "[SEP]": 4}
    idx = len(vocab)
    for seq in sequences:
        for kmer in kmer_tokenize(seq, k):
            if kmer not in vocab:
                vocab[kmer] = idx
                idx += 1
    return vocab


def encode_kmer_sequence(
    sequence: str, vocab: Dict[str, int], k: int = 6,
) -> np.ndarray:
    kmers = kmer_tokenize(sequence, k)
    unk = vocab["[UNK]"]
    return np.array([vocab.get(km, unk) for km in kmers], dtype=np.int32)


def sliding_windows(
    sequence: str, window: int = 128, stride: int = 64,
) -> List[Tuple[int, str]]:
    return [(i, sequence[i : i + window])
            for i in range(0, len(sequence) - window + 1, stride)]


# ═══════════════════════════════════════════════════════════════════════════════
#  BPE Tokenizer  (DNABERT-2 inspired)
# ═══════════════════════════════════════════════════════════════════════════════
class DNABPETokenizer:
    """
    Byte Pair Encoding tokenizer for genomic sequences, inspired by
    DNABERT-2's approach.  Learns sub-word units from DNA data.

    Special tokens:
      [PAD]=0, [MASK]=1, [UNK]=2, [CLS]=3, [SEP]=4
    """
    SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2,
                      "[CLS]": 3, "[SEP]": 4}

    def __init__(self, vocab_size: int = 2048):
        self.target_vocab_size = vocab_size
        self.token2id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}
        self.merges: List[Tuple[str, str]] = []
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, sequences: List[str], max_seq_chars: int = 500_000):
        """Learn BPE merges from a corpus of DNA sequences."""
        print(f"  [BPE] Training tokenizer (target vocab: "
              f"{self.target_vocab_size}) …")

        # Initialise vocab with single bases
        base_chars = list("ACGTN")
        idx = len(self.SPECIAL_TOKENS)
        for ch in base_chars:
            if ch not in self.token2id:
                self.token2id[ch] = idx
                idx += 1

        # Build word list (each word = list of chars)
        corpus_text = "".join(s.upper()[:5000] for s in sequences)
        corpus_text = corpus_text[:max_seq_chars]
        words = [list(corpus_text[i : i + 500])
                 for i in range(0, len(corpus_text), 500)]

        num_merges = self.target_vocab_size - len(self.token2id)

        for merge_n in range(num_merges):
            # Count adjacent pairs
            pair_counts: Counter = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += 1
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if merged not in self.token2id:
                self.token2id[merged] = idx
                idx += 1

            # Apply merge to all words
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if (i < len(word) - 1
                            and word[i] == best_pair[0]
                            and word[i + 1] == best_pair[1]):
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

            if (merge_n + 1) % 200 == 0:
                print(f"    merge {merge_n + 1}/{num_merges} "
                      f"vocab={len(self.token2id)}")

        self.id2token = {v: k for k, v in self.token2id.items()}
        self._trained = True
        print(f"  [BPE] Done — final vocab size: {len(self.token2id)}")

    # ── Encoding ──────────────────────────────────────────────────────────────
    def encode(self, sequence: str,
               add_special: bool = True) -> List[int]:
        """Encode a DNA sequence to a list of token IDs."""
        seq = sequence.upper()
        tokens = list(seq)

        for left, right in self.merges:
            merged = left + right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == left
                        and tokens[i + 1] == right):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        unk_id = self.token2id["[UNK]"]
        ids = [self.token2id.get(t, unk_id) for t in tokens]

        if add_special:
            cls_id = self.token2id["[CLS]"]
            ids = [cls_id] + ids

        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(
            self.id2token.get(i, "")
            for i in ids
            if i >= len(self.SPECIAL_TOKENS)
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id["[PAD]"]

    @property
    def mask_id(self) -> int:
        return self.token2id["[MASK]"]

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str):
        data = {
            "token2id": self.token2id,
            "merges": self.merges,
            "target_vocab_size": self.target_vocab_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DNABPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["target_vocab_size"])
        tok.token2id = data["token2id"]
        tok.id2token = {int(v): k for k, v in tok.token2id.items()}
        tok.merges   = [tuple(m) for m in data["merges"]]
        tok._trained = True
        return tok