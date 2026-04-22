<div align="center">

# Genome Sequencing & Reconstruction System v3.0

An advanced ancient DNA reconstruction pipeline combining transformer pretraining, denoising, phylogenetic graph learning, AlphaFold-inspired Evoformer modeling, distributed training, and interactive real-time 3D genome visualization.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-1f6feb?style=for-the-badge">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API%20Layer-009688?style=for-the-badge">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge">
  <img alt="Status" src="https://img.shields.io/badge/Repo-Research%20Prototype-6f42c1?style=for-the-badge">
</p>

</div>

> **End-to-end workflow:** fetch sequences (NCBI/Ensembl/UCSC) &rarr; simulate damage &rarr; encode &rarr; train 6-phase AI stack (including Evoformer) &rarr; reconstruct fragments &rarr; evaluate &rarr; interact via Dashboard and Live 3D Viewers.

## Overview

This repository is an end-to-end experimental system for genome reconstruction, specializing in damaged or ancient DNA inputs. It seamlessly integrates:

- **Data Acquisition**: Automated fetching from NCBI Entrez, Ensembl REST, and UCSC Genome Browser.
- **Damage Simulation**: Synthetic ancient DNA damage simulation (deamination, oxidation, missing segments, fragmentation).
- **Core AI Models**: DNABERT-2-style transformers, denoising autoencoders, BiLSTMs, phylogenetic GNNs, a fusion cross-attention model, and a new **AlphaFold-inspired Evoformer** for multi-species sequence completion.
- **DeepSpeed Distributed Training**: Accelerated training workloads capable of scaling across hardware profiles.
- **Evaluation & Benchmarking**: Precision, structural similarity, phylogenetic consistency, and confidence calibration scores.
- **Immersive Visualization**: A real-time 3D DNA reconstruction viewer (PyOpenGL), a live DNA damage simulator, a Streamlit dashboard, and high-quality 2D/3D matplotlib reports.

## Architecture At A Glance

```text
Raw or cached genomes (NCBI, Ensembl, UCSC)
        |
        v
Data Fetchers & Multi-Species Loader
        |
        v
Ancient DNA Simulation (deamination, oxidation, fragmentation)
        |
        v
Preprocessing (K-mer, BPE, one-hot encoding, alignment)
        |
        v
6-Phase Training Curriculum
  1. DNABERT-2 Pretraining (MLM)
  2. Denoising Autoencoder (Corruption-Repair)
  3. Phylogenetic GNN (Evolution-Aware)
  4. Ancient Fragment Fine-Tuning
  5. Transformer-GNN Fusion (Cross-Attention)
  6. Evoformer (Multi-Species Evolutionary Alignment)
        |
        v
Multi-Model Ensemble Reconstructor + Confidence Scorer
        |
        +--> Benchmark Reports & Hotspot Maps
        +--> FastAPI Service
        +--> Streamlit Metrics Dashboard
        +--> Interactive Real-Time 3D Viewers (PyOpenGL)
```

## Core Components

| Area | Main Modules | Functionality |
| --- | --- | --- |
| **Data Acquisition** | `data/fetch_sequences.py`<br>`data/multi_species_loader.py` | Download genomes and prepare evolutionary context sequences. |
| **Damage Simulation** | `data/simulate_ancient_dna.py` | Generate realistically degraded DNA fragments. |
| **Preprocessing** | `preprocessing/*.py` | Build BPE tokenizers and k-mer vocabularies. |
| **Models** | `models/*.py` | DNABERT-2, Denoising AE, BiLSTM, GNN, Fusion module, and Evoformer. |
| **Training Pipeline** | `training/phase*.py`<br>`training/distributed_trainer.py` | Multi-stage, checkpointed curriculum training including DeepSpeed distributed runs. |
| **Orchestration** | `pipeline/full_pipeline.py`<br>`main.py` | Unified execution entry point. |
| **Interfaces** | `dashboard/`, `visualization/`, `api/`, `simulation/` | Streamlit app, FastAPI, 2D/3D charts, and PyOpenGL live viewers. |

## Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Optional Dependencies**:
> - `MAFFT` (requires external installation) for superior multiple sequence alignment.
> - `DeepSpeed` for distributed multi-GPU training.

### 2. Run the Full Magic Pipeline

```bash
python main.py
```
*This handles everything: downloads data, generates damage, trains all models, reconstructs the genomes, calculates benchmarks, builds visualizations, and automatically opens the glowing 3D Interactive Viewer.*

### 3. Start the Interactive Live 3D Reconstruction Viewer

```bash
python main.py reconstruct-interactive
```
Engage manual mode to pause AI, step through the genome manually, override base pairings, and watch the confidence metrics update in real-time.

### 4. Watch a Simulated DNA Degradation Process

```bash
python main.py simulate
```
Witness a clean genome slowly decay into ancient DNA.

### 5. Launch the Reporting Dashboard

```bash
python main.py dashboard
```
Opens the Streamlit web app in your browser to view loss curves, structural evaluations, species details, and more.

### 6. Start the API Server

```bash
python main.py serve
```
Explore the endpoints at `http://127.0.0.1:8000/docs`.

---

## CLI Reference

The `main.py` entry point acts as a unified CLI:

| Command | Description |
| --- | --- |
| `python main.py` | Run the full automated end-to-end pipeline. |
| `python main.py train` | Run just the training pipeline (optionally skip downloads with `--skip-download`). |
| `python main.py serve` | Start the FastAPI execution server. |
| `python main.py dashboard` | Launch the Streamlit monitoring and metrics dashboard. |
| `python main.py evaluate` | Process `reconstructions.json` to yield benchmarking metrics. |
| `python main.py simulate` | Launch the **Live DNA Damage Simulator**. |
| `python main.py reconstruct-live` | Launch the automated 3D reconstruction viewer. |
| `python main.py reconstruct-interactive`| Launch the fully manual, gamified interactive 3D reconstruction viewer. |
| `python main.py train-evoformer` | Launch the Phase 6 Evoformer standalone training explicitly. |
| `python main.py train-distributed` | Launch training leveraging DeepSpeed (if available in environment). |

## API Integration

The system exposes a FastAPI backend (`api/app.py`).

### Endpoints
- `GET /api/v1/health` - Health check and system configuration
- `GET /api/v1/species` - Enumerate available genome catalogs
- `POST /api/v1/reconstruct` - Feed degraded sequences and receive ensemble transcriptions
- `POST /api/v1/compare` - Compare fragments with structural reference metrics

## Dashboard and Visualizations

- **Streamlit Dashboard**: (`dashboard/streamlit_app.py`) Interactive dashboard displaying genome maps, attention weights, mapping hotspots, phylogenetic trees, and calibration.
- **2D / 3D Static Artifacts**: Saved natively as images inside `results/visualizations`.
- **Live Simulator**: (`simulation/live_simulation.py`) Generates real-time events imitating deamination/oxidation over time.
- **Interactive 3D Viewer**: (`visualization/reconstruction_viewer.py`) A PyOpenGL application allowing for live intervention in the sequence reconstruction process. 

## Configuration

Control hyperparameters directly via `config/settings.py`. It dynamically scales layers, sequence max lengths, embedding dimension, and hardware configuration according to whether the deployment target has a GPU or relies purely on CPU bounds.

## Known Caveats

- **Research Prototype**: This codebase illustrates the viability of using deep learning combinations to repair damaged structural graphs representing sequences. It is a proof of concept, not a validated bio-medical production stack.
- **DeepSpeed Compatibility**: To use `train-distributed`, DeepSpeed must be appropriately configured with native system dependencies (especially on Windows where WSL or native compiler toolchains are heavily suggested).
- **Metadata**: Published phylogenetic distance constants configured within this codebase serve as broad estimates sourced from literature useful for graph construction; please review references diligently for downstream applications.
