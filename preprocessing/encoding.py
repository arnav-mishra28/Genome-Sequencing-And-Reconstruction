"""
encoding.py
===========
DNA → numerical representations:
  1. One-hot encoding  (A=1000, C=0100, G=0010, T=0001, N=0000)
  2. Integer encoding  (A=0, C=1, G=2, T=3, N=4)
  3. k-mer tokenization (for transformer models)
  4. Physicochemical property encoding
"""

import numpy as np
from typing import List, Tuple, Dict

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

ONE_HOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

# Physicochemical properties: [H-bond donors, H-bond acceptors,
#                               molecular weight, purine(1)/pyrimidine(0)]
PHYSCHEM = {
    "A": [1, 2, 135.13, 1],
    "C": [1, 3, 111.10, 0],
    "G": [2, 3, 151.13, 1],
    "T": [1, 2, 126.11, 0],
    "N": [0, 0, 0.0,    0],
}


def integer_encode(sequence: str) -> np.ndarray:
    return np.array([BASE_TO_INT.get(b, 4) for b in sequence.upper()], dtype=np.int32)


def one_hot_encode(sequence: str) -> np.ndarray:
    return np.array([ONE_HOT.get(b, [0,0,0,0]) for b in sequence.upper()],
                    dtype=np.float32)


def physchem_encode(sequence: str) -> np.ndarray:
    return np.array([PHYSCHEM.get(b, [0,0,0,0]) for b in sequence.upper()],
                    dtype=np.float32)


def kmer_tokenize(sequence: str, k: int = 6) -> List[str]:
    """Slide a window of size k over the sequence."""
    seq = sequence.upper()
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


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
    sequence: str, vocab: Dict[str, int], k: int = 6
) -> np.ndarray:
    kmers = kmer_tokenize(sequence, k)
    unk = vocab["[UNK]"]
    return np.array([vocab.get(km, unk) for km in kmers], dtype=np.int32)


def decode_integer(arr: np.ndarray) -> str:
    return "".join(INT_TO_BASE.get(int(x), "N") for x in arr)


def sliding_windows(
    sequence: str, window: int = 128, stride: int = 64
) -> List[Tuple[int, str]]:
    """Yield (start_pos, window_seq) pairs."""
    windows = []
    for i in range(0, len(sequence) - window + 1, stride):
        windows.append((i, sequence[i:i+window]))
    return windows