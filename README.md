# 🧬 Genome Sequencing And Reconstruction

> **State-of-the-art Deep Learning pipeline for ancient DNA reconstruction**
> All files stored in `D:\Genome Sequencing And Reconstruction\`

---

## 🗂 Project Structure

D:\Genome Sequencing And Reconstruction
├── data\ # Sequence data, simulations, mappings ├── preprocessing\ # Encoding + alignment tools ├── models\ # LSTM, DNABERT, Autoencoder, GNN ├── pipeline\ # Orchestration + genome mapper ├── visualization\ # 3D helix, 2D maps, live dashboard ├── results\ # All outputs + HTML visualizations ├── main.py # ← RUN THIS └── requirements.txt

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline + visualizations + dashboard
python D:\Genome Sequencing And Reconstruction\main.py

# 3. Skip download if already cached
python main.py --skip-download

# 4. Dashboard only (after pipeline ran once)
python main.py --dashboard-only

# 5. Fast test run (fewer epochs)
python main.py --lstm-epochs 2 --bert-epochs 2 --ae-epochs 3 --gnn-epochs 10
```plaintext

---

## 🧬 Species Included

| Species | Type | Reference |
|---|---|---|
| Neanderthal | Extinct | FM865409 |
| Woolly Mammoth | Extinct | NC_007596 |
| Dodo | Extinct | MH595722 |
| Thylacine | Extinct | AY463959 |
| Passenger Pigeon | Extinct | KF000598 |
| Cave Bear | Extinct | AJ788962 |
| Saber-tooth Cat | Extinct | KF748507 |
| Woolly Rhinoceros | Extinct | KX534097 |
| Modern Human | Modern | NC_012920 |
| African Elephant | Modern | NC_005129 |
| Gray Wolf | Modern | NC_008093 |
| Rock Pigeon | Modern | NC_013979 |

---

## 🤖 Models

| Model | Purpose |
|---|---|
| **BiLSTM** | Sequence completion / prediction |
| **DNABERT** | Masked DNA filling (k-mer transformer) |
| **Denoising AE** | Base damage repair |
| **Phylo-GNN** | Evolutionary context refinement |

---

## 📊 Outputs

- `results/reconstructions.json` — reconstructed sequences + confidence
- `results/visualizations/helix_3d_*.html` — 3D animated helix per species
- `results/visualizations/genome_map_2d_*.html` — 2D chromosome maps
- `results/visualizations/reconstruction_comparison.html` — before/after
- `results/visualizations/phylogenetic_tree.html` — GNN phylo tree
- Dashboard at `http://127.0.0.1:8050`

---

## 🧪 Mutation Types Simulated

| Mutation | Biological Effect |
|---|---|
| C→T deamination | Hallmark ancient DNA damage; false SNPs |
| G→A deamination | 3′-end bias; oncogenic RAS-like variants |
| G→T oxidation | 8-oxoG; TP53-like transversions |
| Random substitution | Background radiation / sequencing error |
| Missing segment | Hydrolytic backbone cleavage |
| Small insertion | Frameshift; repeat expansion |
| Small deletion | Frameshift; exon loss |

---

## 📌 Notes

- All data paths hardcoded to `D:\`
- NCBI accessions are real published sequences
- Synthetic fallback auto-generated if download fails
- Dashboard auto-refreshes every 5 seconds

1. Create folder: D:\Genome Sequencing And Reconstruction\

2. Create subfolders:
   D:\Genome Sequencing And Reconstruction\data\
   D:\Genome Sequencing And Reconstruction\preprocessing\
   D:\Genome Sequencing And Reconstruction\models\
   D:\Genome Sequencing And Reconstruction\pipeline\
   D:\Genome Sequencing And Reconstruction\visualization\
   D:\Genome Sequencing And Reconstruction\results\

3. Create an empty __init__.py in each subfolder.

4. Paste each file's code into its respective path.

5. Open terminal → D:\Genome Sequencing And Reconstruction\
   pip install -r requirements.txt

6. Run:
   python main.py
```plaintext

---

## 🏗️ Architecture Summary

```plaintext
NCBI Data (Real mtDNA)
        ↓
  Ancient DNA Simulation
  (C→T, G→A, oxidation, gaps, fragmentation)
        ↓
  MAFFT-style Alignment → Reference Genome
        ↓
  ┌─────────────────────────────────────┐
  │     ENSEMBLE RECONSTRUCTION         │
  │  1. Denoising CNN Autoencoder       │
  │     (detect + repair damaged bases) │
  │  2. DNABERT (fill N gaps via MLM)   │
  │  3. BiLSTM  (extend + predict)      │
  │  4. Phylo-GNN (evolutionary priors) │
  └─────────────────────────────────────┘
        ↓
  Fragment Mapping → Reference
  (gene regions, variants, disease loci, hotspots)
        ↓
  ┌─────────────────────────────────────┐
  │         VISUALIZATIONS              │
  │  • 3D Double Helix Animation        │
  │  │  (damaged → step-by-step repair) │
  │  • 2D Chromosome / Gene Maps        │
  │  • Phylogenetic Tree                │
  │  • Real-Time Dash Dashboard         │
  └─────────────────────────────────────┘

  ✅ Everything is in D:\Genome Sequencing And Reconstruction\ ✅ Real NCBI accession numbers (with synthetic fallback) ✅ All mutation types documented with biological effects ✅ 3D Animated Helix — damaged → step-by-step → reconstructed ✅ 2D Chromosome Maps — genes, variants, hotspots, disease loci ✅ Live Dash Dashboard — all metrics update every 5 seconds ✅ Zero broken imports — every cross-module reference is valid