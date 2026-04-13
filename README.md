<div align="center">

# Genome Sequencing And Reconstruction

Ancient DNA reconstruction pipeline that combines transformer pretraining, denoising, phylogenetic graph learning, fusion modeling, benchmarking, and interactive genome visualization.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-1f6feb?style=for-the-badge">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API%20Layer-009688?style=for-the-badge">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge">
  <img alt="Status" src="https://img.shields.io/badge/Repo-Research%20Prototype-6f42c1?style=for-the-badge">
</p>

</div>

> End-to-end workflow: fetch sequences -> simulate damage -> align and encode -> train models -> reconstruct fragments -> benchmark outputs -> explore results through an API, dashboard, and live viewers

## Overview

This repository is an end-to-end experimental system for genome reconstruction, with a strong focus on damaged or ancient DNA-like inputs. It includes:

- sequence acquisition from NCBI, Ensembl, and UCSC with caching and fallback generation
- synthetic ancient DNA damage simulation, fragmentation, and corruption pairing
- preprocessing utilities for encoding, k-mer vocab building, and alignment
- multiple reconstruction models, including DNABERT-2-style transformers, denoising autoencoders, a BiLSTM predictor, phylogenetic GNNs, a fusion model, and an Evoformer-style model
- evaluation tools for accuracy, edit distance, similarity, phylogenetic consistency, and calibration
- delivery layers for FastAPI, a Streamlit dashboard, 2D plots, 3D helix views, and interactive reconstruction viewers

## Architecture At A Glance

```text
Raw or cached genomes
        |
        v
Data fetchers + metadata catalog
        |
        v
Ancient DNA simulation
  - deamination
  - oxidative damage
  - substitutions / indels
  - missing segments and fragmentation
        |
        v
Preprocessing
  - alignment
  - integer / one-hot / physicochemical encoding
  - k-mer vocab and BPE-style tokenization
        |
        v
Training stack
  - DNABERT-2 pretraining
  - denoising autoencoder
  - phylogenetic GNN
  - ancient-fragment fine-tuning
  - transformer-GNN fusion
  - Evoformer-style multi-species model
  - BiLSTM sequence completion
        |
        v
Ensemble reconstruction + confidence scoring
        |
        +--> benchmark reports
        +--> genome mapping and hotspot analysis
        +--> FastAPI service
        +--> Streamlit dashboard
        +--> 2D/3D visualization and live viewers
```

## Core Components

| Area | Main modules | What they do |
| --- | --- | --- |
| Data acquisition | `data/fetch_sequences.py`, `data/multi_species_loader.py` | Download or reuse cached genomes and build multi-species training inputs |
| Damage simulation | `data/simulate_ancient_dna.py` | Generate ancient DNA-like corruption, fragmentation, and mutation logs |
| Preprocessing | `preprocessing/encoding.py`, `preprocessing/alignment.py`, `preprocessing/corruption.py` | Encode sequences, train BPE tokenizers, align to references, and make corruption pairs |
| Models | `models/` | DNABERT-2-style transformer, denoising AE, BiLSTM, phylogenetic GNN, fusion model, confidence scorer, ESM-inspired encoder, Evoformer |
| Training | `training/phase1_pretrain.py` through `training/phase6_evoformer.py` | Multi-stage training pipeline with checkpointing and histories |
| Pipeline orchestration | `pipeline/full_pipeline.py`, `pipeline/genome_mapper.py` | Runs the full experiment, reconstruction, benchmarking, and mapping |
| Evaluation | `evaluation/metrics.py`, `evaluation/benchmark.py` | Quantitative metrics and report generation |
| Interfaces | `api/`, `dashboard/`, `visualization/`, `simulation/` | API server, dashboard, 2D/3D outputs, live simulation and interactive viewers |

## Training Pipeline

The main training and reconstruction flow implemented in `pipeline/full_pipeline.py` runs these stages:

| Stage | File | Purpose |
| --- | --- | --- |
| 0 | `data/fetch_sequences.py` | Fetch source genomes and write `data/sequences/metadata.json` |
| 0 | `data/simulate_ancient_dna.py` | Corrupt modern or reference genomes into ancient DNA-like fragments |
| 0 | `preprocessing/alignment.py` | Align damaged reads to reference sequences |
| 0 | `preprocessing/encoding.py` | Build token vocabularies and encoded datasets |
| 1 | `training/phase1_pretrain.py` | Pretrain the DNABERT-2-style model on modern genomes |
| 2 | `training/phase2_corruption.py` | Train the denoising autoencoder on corruption-repair pairs |
| 3 | `training/phase3_evolution_aware.py` | Train the phylogenetic GNN with biological constraint losses |
| 4 | `training/phase4_finetune.py` | Fine-tune transformer weights on ancient fragments |
| 5 | `training/phase5_fusion.py` | Train the transformer-GNN fusion model with confidence outputs |
| 6 | `training/phase6_evoformer.py` | Train the Evoformer-style multi-species model |
| Extra lane | `models/lstm_predictor.py` | Train a BiLSTM for sequence completion and ensemble support |
| Output | `models/ensemble_reconstructor.py` | Combine model outputs into final reconstructions and confidences |

`python main.py` with no subcommand dispatches to the full end-to-end pipeline via `run_everything()`.

## Repository Snapshot

The current checked-in artifacts already include model histories and generated reports:

| Item | Current snapshot |
| --- | --- |
| Cached sequence metadata | 14 catalog entries in `data/sequences/metadata.json` |
| Latest benchmark species count | 10 |
| Average accuracy | `0.4792` |
| Average similarity | `0.4792` |
| Average phylogenetic consistency | `0.6725` |
| Expected calibration error | `0.3551` |
| Latest pipeline runtime | `680.0` seconds |
| Saved vocab size | `4031` tokens |
| Checkpoint histories present | DNABERT-2, denoising AE, phylo GNN, fusion, BiLSTM |

Training history files in `models/checkpoints/` show the following runs:

| Model | Snapshot |
| --- | --- |
| DNABERT-2 | 5 epochs, training loss from about `8.44` to `7.61` |
| Denoising autoencoder | 6 epochs, validation loss down to about `1.10` |
| Phylogenetic GNN | 30 epochs, final loss about `0.0107` |
| Fusion model | 3 epochs, validation loss about `7.79` |
| BiLSTM | 5 epochs, accuracy up to about `0.338` |

<details>
<summary>Cached sequence labels in this repository snapshot</summary>

`neanderthal_mtDNA`, `mammoth_mtDNA`, `dodo_partial`, `thylacine_mtDNA`, `passenger_pigeon`, `cave_bear_mtDNA`, `saber_tooth_cat`, `woolly_rhino`, `human_mtDNA`, `elephant_mtDNA`, `gray_wolf_mtDNA`, `rock_pigeon_mtDNA`, `ensembl_human_mt`, `ucsc_human_mt`

</details>

> Note: these keys are the repository's dataset identifiers. If you need publication-grade provenance, verify the underlying accessions and cached metadata before citing them.

## Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit
```

Optional tools:

- `MAFFT` for external multiple sequence alignment support
- `DeepSpeed` for `train-distributed`

### 2. Run the full pipeline

```bash
python main.py
```

That default run verifies modules, fetches or reuses sequences, simulates damage, trains the model stack, reconstructs sequences, benchmarks results, maps fragments, and generates visualization artifacts.

### 3. Run a lighter training pass with cached data

```bash
python main.py train --skip-download --no-viz --phase1-epochs 2 --bert-epochs 2 --ae-epochs 2 --gnn-epochs 5 --lstm-epochs 2 --batch-size 4
```

### 4. Evaluate saved reconstructions

```bash
python main.py evaluate
```

### 5. Start the API

```bash
python main.py serve
```

Open Swagger docs at `http://127.0.0.1:8000/docs`.

### 6. Launch the dashboard

```bash
python main.py dashboard
```

This starts the Streamlit app in `dashboard/streamlit_app.py`. Unless you pass extra Streamlit flags, it uses Streamlit's default port.

### 7. Run the integration smoke test

```bash
python test_integration.py
```

## CLI Reference

| Command | What it does |
| --- | --- |
| `python main.py` | Run the full end-to-end workflow |
| `python main.py train` | Run the training pipeline |
| `python main.py serve` | Start the FastAPI server on `0.0.0.0:8000` |
| `python main.py dashboard` | Launch the Streamlit metrics dashboard |
| `python main.py evaluate` | Run the benchmark suite |
| `python main.py simulate` | Launch the live DNA damage simulation viewer |
| `python main.py reconstruct-live` | Launch the real-time 3D reconstruction viewer |
| `python main.py reconstruct-interactive` | Launch the manual interactive reconstruction viewer |
| `python main.py train-evoformer` | Train the Evoformer model standalone |
| `python main.py train-distributed` | Train with DeepSpeed if available |

Important CLI options already supported in `main.py`:

- `train`: `--skip-download`, `--no-viz`, `--no-dashboard`, `--phase1-epochs`, `--bert-epochs`, `--ae-epochs`, `--gnn-epochs`, `--lstm-epochs`, `--batch-size`
- `simulate`: `--species`, `--sequence`, `--speed`, `--manual`, `--no-3d`, `--max-bases`, `--max-events`, `--seed`
- `reconstruct-live`: `--species`, `--sequence`, `--max-bases`, `--speed`, `--width`, `--height`
- `reconstruct-interactive`: `--species`, `--sequence`, `--max-bases`, `--speed`, `--width`, `--height`
- `train-evoformer` and `train-distributed`: `--skip-download`, `--epochs`, `--batch-size`, `--profile {small,medium,large,xl}`

## API

The FastAPI app lives in `api/app.py` and mounts routes from `api/routes.py`.

### Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Health check, device info, and model-loaded status |
| `GET` | `/api/v1/species` | Available species catalog from the configured sequence set |
| `POST` | `/api/v1/reconstruct` | Reconstruct a genome from uploaded fragments |
| `POST` | `/api/v1/compare` | Compare uploaded fragments directly against a reference |

### Example request body

```json
{
  "fragments": [
    {
      "seq": "ACGTNNNACGT",
      "start": 0,
      "end": 11
    }
  ],
  "species_hint": "neanderthal_mtDNA",
  "reference": "ACGTAAAACGT"
}
```

### Reconstruction response includes

- reconstructed sequence
- per-position confidence scores
- mean confidence and coverage
- gap counts before and after repair
- up to 50 repair details
- optional evaluation metrics when a reference is supplied
- optional evolutionary comparison when a species hint is supplied

## Dashboard And Visualization

The repository includes several ways to inspect outputs:

- `dashboard/streamlit_app.py`: multi-page dashboard for overview, training curves, sequences, confidence, phylogenetics, calibration, species detail, benchmark results, and upload-based reconstruction
- `visualization/genome_map_2d.py`: ideograms, radar charts, benchmark comparisons, calibration plots, phylogenetic charts
- `visualization/helix_3d.py`: HTML 3D helix generation
- `visualization/live_viewer.py`, `visualization/live_helix_3d.py`, `visualization/live_genome_2d.py`: live simulation displays
- `visualization/reconstruction_viewer.py`: PyOpenGL interactive reconstruction viewer
- `simulation/live_simulation.py`: real-time DNA damage simulator

## Generated Artifacts

Important output files written by the pipeline include:

| File | Description |
| --- | --- |
| `results/kmer_vocab.json` | Learned sequence vocabulary |
| `results/reconstructions.json` | Per-species reconstruction outputs, confidence values, coverage, and mutation summaries |
| `results/benchmark_report.json` | Aggregate and per-species metrics |
| `results/pipeline_log.json` | Run log with elapsed time and completed steps |
| `results/visualizations/` | Saved charts and figures |
| `models/checkpoints/*.pt` | Trained model checkpoints |
| `models/checkpoints/*_history.json` | Training histories for each phase |
| `data/alignments/*.json` | Alignment outputs |
| `data/mappings/*.json` | Fragment mapping and hotspot results |

## Project Layout

```text
api/                  FastAPI app, routes, and request/response schemas
config/               Settings, species catalogs, phylogenetic metadata, hardware-aware defaults
dashboard/            Streamlit dashboard
data/                 Fetchers, simulators, datasets, cached FASTA files, metadata
docker/               Dockerfile and compose definitions
evaluation/           Metrics and benchmark runner
models/               Transformers, autoencoder, GNN, fusion, confidence, Evoformer, ensemble
pipeline/             Full pipeline orchestration and genome mapping
preprocessing/        Alignment, encoding, corruption helpers
results/              Reports, reconstructions, vocab, and generated figures
simulation/           Live simulation logic
training/             Phase-based training scripts, trainers, and distributed entrypoints
visualization/        2D charts, 3D views, live viewers, reconstruction tools
main.py               Main CLI entrypoint
requirements.txt      Python dependency list
test_integration.py   Integration smoke tests for key modules
```

## Configuration Notes

`config/settings.py` centralizes most runtime behavior:

- auto-detects CPU vs GPU and scales embedding size, layer count, max sequence length, and batch size
- defines API host and port
- creates required data, model, and result directories on import
- stores species catalogs, reference mappings, mitochondrial gene annotations, disease mutation locations, and phylogenetic distance priors

## Docker

Build and run with:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Current compose setup:

- `api` exposes port `8000`
- `dashboard` runs `python main.py dashboard`
- project `data/`, `models/checkpoints/`, and `results/` are mounted into the containers

Practical note: the current dashboard entrypoint is Streamlit-based, but `requirements.txt` does not install `streamlit`. If you want the dashboard container to start cleanly, add Streamlit to the image environment first. Also note that `docker/docker-compose.yml` maps the dashboard service to port `8050`, while `main.py dashboard` launches Streamlit without forcing that port.

## Known Caveats

- This repository is best understood as a research prototype and systems demo, not a validated production genomics stack.
- Sequence labels in cached metadata should be treated as project identifiers unless you independently verify the upstream accessions.
- Some optional runtime paths depend on external software such as MAFFT, PyOpenGL, or DeepSpeed.
- The active dashboard implementation is Streamlit-based even though `requirements.txt` currently includes Dash dependencies.

## Why This Repo Is Interesting

What makes the project stand out is not just one model, but the way the pieces are connected:

- classical corruption simulation sits next to modern sequence modeling
- multiple architectures contribute to reconstruction instead of a single monolithic model
- confidence and calibration are treated as first-class outputs
- benchmarking, mapping, visualization, and service layers are already wired into the same repo

If you want one starting point, begin with `main.py`, then read `pipeline/full_pipeline.py`, and finally inspect `models/ensemble_reconstructor.py` to see how the different learned components come together.

