"""
settings.py
===========
Centralized configuration for the entire Genome Sequencing & Reconstruction
system.  Every module imports paths and hyperparameters from here — no more
hard-coded ``D:\\`` literals anywhere in the codebase.
"""

import os
import torch

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
SEQ_DIR     = os.path.join(DATA_DIR, "sequences")
SIM_DIR     = os.path.join(DATA_DIR, "simulated")
ALIGN_DIR   = os.path.join(DATA_DIR, "alignments")
MAP_DIR     = os.path.join(DATA_DIR, "mappings")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIZ_DIR     = os.path.join(RESULTS_DIR, "visualizations")

# Create every directory on import so nothing crashes at runtime
for _d in [DATA_DIR, SEQ_DIR, SIM_DIR, ALIGN_DIR, MAP_DIR,
           MODEL_DIR, RESULTS_DIR, VIZ_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Device detection ──────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP  = torch.cuda.is_available()
IS_GPU   = torch.cuda.is_available()

# ── Model sizing  (auto-scale based on hardware) ─────────────────────────────
# GPU-class defaults
_GPU_DEFAULTS = dict(
    EMBED_DIM      = 768,
    N_HEADS        = 12,
    N_LAYERS       = 12,
    FFN_DIM        = 3072,
    MAX_SEQ_LEN    = 512,
    BATCH_SIZE     = 32,
    BPE_VOCAB_SIZE = 4096,
)

# CPU-safe defaults  (fits in < 8 GB RAM)
_CPU_DEFAULTS = dict(
    EMBED_DIM      = 256,
    N_HEADS        = 4,
    N_LAYERS       = 6,
    FFN_DIM        = 1024,
    MAX_SEQ_LEN    = 256,
    BATCH_SIZE     = 8,
    BPE_VOCAB_SIZE = 2048,
)

_hw = _GPU_DEFAULTS if IS_GPU else _CPU_DEFAULTS

EMBED_DIM      = _hw["EMBED_DIM"]
N_HEADS        = _hw["N_HEADS"]
N_LAYERS       = _hw["N_LAYERS"]
FFN_DIM        = _hw["FFN_DIM"]
MAX_SEQ_LEN    = _hw["MAX_SEQ_LEN"]
BATCH_SIZE     = _hw["BATCH_SIZE"]
BPE_VOCAB_SIZE = _hw["BPE_VOCAB_SIZE"]

# ── Shared training hyperparameters ───────────────────────────────────────────
LEARNING_RATE      = 2e-4
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.1
DROPOUT            = 0.1
MASK_PROB          = 0.15
GRADIENT_CLIP      = 1.0
MAX_SAMPLES        = 5000      # hard cap on dataset rows (safety)
NUM_WORKERS        = 0         # Windows: must be 0

# ── Phase-default epochs ──────────────────────────────────────────────────────
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5
PHASE3_EPOCHS = 5
PHASE4_EPOCHS = 3
LSTM_EPOCHS   = 5
AE_EPOCHS     = 6
GNN_EPOCHS    = 30
PHASE5_EPOCHS = 3
FUSION_LR     = 1e-4
FUSION_N_MC_SAMPLES = 10   # Monte Carlo dropout samples for confidence

# ── NCBI accessions (real published sequences) ───────────────────────────────
NCBI_SEQUENCES = {
    # Ancient / Extinct
    "neanderthal_mtDNA":   "FM865409",
    "mammoth_mtDNA":       "NC_007596",
    "dodo_partial":        "MH595722",
    "thylacine_mtDNA":     "AY463959",
    "passenger_pigeon":    "KF000598",
    "cave_bear_mtDNA":     "AJ788962",
    "saber_tooth_cat":     "KF748507",
    "woolly_rhino":        "KX534097",
    # Modern relatives
    "human_mtDNA":         "NC_012920",
    "elephant_mtDNA":      "NC_005129",
    "gray_wolf_mtDNA":     "NC_008093",
    "rock_pigeon_mtDNA":   "NC_013979",
}

MODERN_SPECIES  = {"human_mtDNA", "elephant_mtDNA", "gray_wolf_mtDNA",
                   "rock_pigeon_mtDNA"}

REF_MAP = {
    "neanderthal_mtDNA":  "human_mtDNA",
    "mammoth_mtDNA":      "elephant_mtDNA",
    "woolly_rhino":       "elephant_mtDNA",
    "cave_bear_mtDNA":    "gray_wolf_mtDNA",
    "thylacine_mtDNA":    "gray_wolf_mtDNA",
    "passenger_pigeon":   "rock_pigeon_mtDNA",
    "dodo_partial":       "rock_pigeon_mtDNA",
    "saber_tooth_cat":    "gray_wolf_mtDNA",
}

# ── Phylogenetic distances (Myr) — from published literature ─────────────────
PHYLO_DISTANCES = {
    ("neanderthal_mtDNA",  "human_mtDNA"):       0.6,
    ("mammoth_mtDNA",      "elephant_mtDNA"):     5.0,
    ("woolly_rhino",       "elephant_mtDNA"):    58.0,
    ("cave_bear_mtDNA",    "gray_wolf_mtDNA"):   40.0,
    ("thylacine_mtDNA",    "gray_wolf_mtDNA"):  160.0,
    ("passenger_pigeon",   "rock_pigeon_mtDNA"): 30.0,
    ("dodo_partial",       "rock_pigeon_mtDNA"): 25.0,
    ("saber_tooth_cat",    "gray_wolf_mtDNA"):   90.0,
    ("human_mtDNA",        "elephant_mtDNA"):    90.0,
}

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Ensembl REST base ─────────────────────────────────────────────────────────
ENSEMBL_REST = "https://rest.ensembl.org"
UCSC_API     = "https://api.genome.ucsc.edu"

# ── mtDNA gene map (human NC_012920) ──────────────────────────────────────────
MTDNA_GENES = [
    {"gene": "D-loop",        "start": 1,     "end": 576,   "type": "control"},
    {"gene": "tRNA-Phe",      "start": 577,   "end": 647,   "type": "tRNA"},
    {"gene": "12S rRNA",      "start": 648,   "end": 1601,  "type": "rRNA"},
    {"gene": "tRNA-Val",      "start": 1602,  "end": 1670,  "type": "tRNA"},
    {"gene": "16S rRNA",      "start": 1671,  "end": 3229,  "type": "rRNA"},
    {"gene": "ND1",           "start": 3307,  "end": 4262,  "type": "protein_coding"},
    {"gene": "ND2",           "start": 4470,  "end": 5511,  "type": "protein_coding"},
    {"gene": "COX1",          "start": 5904,  "end": 7445,  "type": "protein_coding"},
    {"gene": "COX2",          "start": 7586,  "end": 8269,  "type": "protein_coding"},
    {"gene": "ATP8",          "start": 8366,  "end": 8572,  "type": "protein_coding"},
    {"gene": "ATP6",          "start": 8527,  "end": 9207,  "type": "protein_coding"},
    {"gene": "COX3",          "start": 9207,  "end": 9990,  "type": "protein_coding"},
    {"gene": "ND4L",          "start": 10059, "end": 10404, "type": "protein_coding"},
    {"gene": "ND4",           "start": 10404, "end": 11935, "type": "protein_coding"},
    {"gene": "ND5",           "start": 11742, "end": 13565, "type": "protein_coding"},
    {"gene": "ND6",           "start": 13552, "end": 14070, "type": "protein_coding"},
    {"gene": "Cytochrome-b",  "start": 14747, "end": 15887, "type": "protein_coding"},
    {"gene": "D-loop2",       "start": 15888, "end": 16569, "type": "control"},
]

DISEASE_MUTATIONS = {
    3243:  ("A→G", "MELAS syndrome — mitochondrial encephalomyopathy"),
    8344:  ("A→G", "MERRF syndrome — myoclonic epilepsy"),
    11778: ("G→A", "Leber's Hereditary Optic Neuropathy (LHON)"),
    3460:  ("G→A", "LHON — ND1 subunit"),
    14484: ("T→C", "LHON — cytochrome b region"),
    4160:  ("T→C", "LHON — mild ND1 variant"),
    7444:  ("G→A", "LHON — COX1 variant"),
}
