# 🧬 Genome Sequencing & Reconstruction v2.0

> **Research-grade Deep Learning pipeline for ancient DNA reconstruction**
> Powered by DNABERT-2 · ESM · AlphaFold Attention · Phylogenetic GNN

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                  │
│  NCBI Entrez  ·  Ensembl REST  ·  UCSC Genome Browser           │
│  Species: Neanderthal, Mammoth, Human, Elephant, + 8 more        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│           ANCIENT DNA DAMAGE SIMULATION                          │
│  Position-dependent deamination curves (Briggs et al. 2007)      │
│  Strand bias · Oxidation · Indels · Fragmentation                │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              4-PHASE TRAINING CURRICULUM                         │
│                                                                  │
│  Phase 1 ─ Pre-train DNABERT-2 on modern genomes (MLM)           │
│  Phase 2 ─ Corruption → reconstruction training (Denoising AE)   │
│  Phase 3 ─ Evolution-aware GNN constraints                       │
│  Phase 4 ─ Fine-tune on real ancient DNA fragments               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              ENSEMBLE RECONSTRUCTION                             │
│                                                                  │
│  DNABERT-2 Transformer (ALiBi + GEGLU + BPE)                     │
│     + ESM Structure Encoder (RoPE + Contact Prediction)          │
│     + Convolutional Denoising AE (U-Net + SE-Block)              │
│     + Phylogenetic GNN (GAT + Bio Constraints)                   │
│     + BiLSTM Sequence Predictor                                  │
│  ──────────────────────────────────                              │
│  → Position-wise Gating Fusion                                   │
│  → Temperature-scaled Confidence Calibration                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION (5 Metrics)                              │
│                                                                  │
│  1. Sequence Accuracy          (per-base correctness)            │
│  2. Edit Distance              (Levenshtein)                     │
│  3. Reconstruction Similarity  (BLAST-like identity)             │
│  4. Phylogenetic Consistency   (evolutionary distance check)     │
│  5. Confidence Calibration     (Expected Calibration Error)      │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT                                              │
│                                                                  │
│  FastAPI Server → /docs (Swagger)                                │
│  Dash Dashboard → real-time metrics                              │
│  3D Helix + 2D Genome Maps                                       │
│  JSON Reports + Benchmark results                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full 4-phase pipeline
python main.py train

# 3. Quick test (fewer epochs)
python main.py train --bert-epochs 1 --ae-epochs 2 --gnn-epochs 5 --lstm-epochs 1

# 4. Skip download if cached
python main.py train --skip-download

# 5. Start FastAPI server (after training)
python main.py serve
# → http://localhost:8000/docs

# 6. Run evaluation only
python main.py evaluate

# 7. Dashboard only
python main.py dashboard
```

### Docker

```bash
cd docker
docker-compose up --build
# API → http://localhost:8000/docs
# Dashboard → http://localhost:8050
```

---

## 🗂 Project Structure

```
├── config/
│   └── settings.py              # Centralized config (auto-scales CPU/GPU)
├── data/
│   ├── fetch_sequences.py       # NCBI + Ensembl + UCSC downloads
│   ├── simulate_ancient_dna.py  # Position-dependent damage simulation
│   └── dataset_builder.py       # PyTorch datasets for 4 phases
├── preprocessing/
│   ├── encoding.py              # One-hot, integer, k-mer, BPE tokenizer
│   ├── alignment.py             # Smith-Waterman + MAFFT MSA wrapper
│   └── corruption.py            # Configurable DNA corruption engine
├── models/
│   ├── dnabert2_transformer.py  # DNABERT-2 (ALiBi + GEGLU + BPE)
│   ├── esm_structure_encoder.py # ESM-inspired (RoPE + Contact Pred)
│   ├── alphafold_attention.py   # Evoformer (Triangular Updates + MSA)
│   ├── denoising_autoencoder.py # U-Net AE with SE-Block attention
│   ├── gnn_phylogenetic.py      # GNN + GAT with biological constraints
│   ├── ensemble_reconstructor.py# Multi-model fusion + calibration
│   └── lstm_predictor.py        # BiLSTM sequence completion
├── training/
│   ├── trainer.py               # Shared utilities, early stopping, logging
│   ├── phase1_pretrain.py       # MLM on modern genomes
│   ├── phase2_corruption.py     # Corrupted → original training
│   ├── phase3_evolution_aware.py# GNN biological constraints
│   └── phase4_finetune.py       # Fine-tune on real ancient DNA
├── evaluation/
│   ├── metrics.py               # 5 evaluation metrics
│   └── benchmark.py             # End-to-end benchmarking suite
├── api/
│   ├── app.py                   # FastAPI application
│   ├── routes.py                # API endpoints
│   └── schemas.py               # Pydantic models
├── pipeline/
│   ├── full_pipeline.py         # Pipeline orchestration
│   └── genome_mapper.py         # Fragment → reference mapping
├── visualization/
│   ├── helix_3d.py              # 3D DNA helix animation
│   ├── genome_map_2d.py         # 2D chromosome maps
│   └── realtime_dashboard.py    # Live Dash dashboard
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🧬 Species Included

| Species | Type | NCBI Accession | Modern Relative |
|---------|------|----------------|-----------------|
| Neanderthal | Ancient | FM865409 | Human |
| Woolly Mammoth | Ancient | NC_007596 | Elephant |
| Dodo | Ancient | MH595722 | Rock Pigeon |
| Thylacine | Ancient | AY463959 | Gray Wolf |
| Passenger Pigeon | Ancient | KF000598 | Rock Pigeon |
| Cave Bear | Ancient | AJ788962 | Gray Wolf |
| Saber-tooth Cat | Ancient | KF748507 | Gray Wolf |
| Woolly Rhinoceros | Ancient | KX534097 | Elephant |
| Modern Human | Modern | NC_012920 | — |
| African Elephant | Modern | NC_005129 | — |
| Gray Wolf | Modern | NC_008093 | — |
| Rock Pigeon | Modern | NC_013979 | — |

---

## 🤖 Models

| Model | Architecture | Inspiration | Purpose |
|-------|-------------|-------------|---------|
| **DNABERT-2** | Transformer + ALiBi + GEGLU | DNABERT-2 (Zhou 2023) | Masked DNA prediction |
| **ESM Encoder** | Transformer + RoPE + Contact Head | ESM-2 (Meta AI) | Structure-aware encoding |
| **AlphaFold Attn** | Evoformer + Triangular Updates | AlphaFold2 (DeepMind) | Pairwise representation |
| **Denoising AE** | U-Net Conv1D + SE-Block | — | Base damage repair |
| **Phylo-GNN** | GCN + GAT + Bio Constraints | — | Evolutionary refinement |
| **BiLSTM** | Bidirectional LSTM + Attention | — | Sequence completion |
| **Ensemble** | Position-wise Gating + Temp Scaling | — | Multi-model fusion |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Sequence Accuracy** | Per-base correctness vs reference |
| **Edit Distance** | Normalised Levenshtein distance |
| **Reconstruction Similarity** | BLAST-like sliding-window identity |
| **Phylogenetic Consistency** | k-mer cosine similarity vs expected evolutionary distance |
| **Confidence Calibration** | Expected Calibration Error (ECE) |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/reconstruct` | Upload fragments → get reconstructed genome |
| `GET` | `/api/v1/species` | List available reference species |
| `POST` | `/api/v1/compare` | Compare reconstruction to reference |
| `GET` | `/api/v1/health` | Health check |

**Swagger docs:** http://localhost:8000/docs

---

## 📚 References

- **DNABERT-2**: Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome", ICLR 2024
- **ESM-2**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023
- **AlphaFold2**: Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 2021
- **ALiBi**: Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022
- **Ancient DNA damage**: Briggs et al., "Patterns of damage in genomic DNA sequences from a Neandertal", PNAS 2007