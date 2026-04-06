"""
app.py
======
FastAPI application for the Genome Sequencing & Reconstruction API.

Features:
  - Upload DNA fragments
  - Get: Reconstructed genome, confidence scores, evolutionary comparison
  - Swagger docs at /docs
"""

import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import API_HOST, API_PORT

app = FastAPI(
    title="🧬 Genome Sequencing & Reconstruction API",
    description=(
        "Advanced ancient DNA reconstruction system powered by "
        "DNABERT-2, ESM-inspired, and AlphaFold attention architectures.\n\n"
        "**Features:**\n"
        "- Upload DNA fragments\n"
        "- Get reconstructed genome with confidence scores\n"
        "- Evolutionary comparison to modern relatives\n"
        "- 5 evaluation metrics (accuracy, edit distance, similarity, "
        "phylogenetic consistency, confidence calibration)"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routes
from api.routes import router, set_models
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Load models from checkpoints on startup."""
    print("\n" + "=" * 60)
    print("  🧬 Genome Reconstruction API — Starting up …")
    print("=" * 60)

    try:
        from config.settings import MODEL_DIR, DEVICE
        import torch
        import json

        models   = {}
        vocab    = {}
        sequences = {}

        # Try to load vocab
        vocab_path = os.path.join(_PROJECT, "results", "kmer_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path) as f:
                vocab = json.load(f)
            print(f"  Loaded vocab: {len(vocab)} tokens")

        # Try to load models
        bert_path = os.path.join(MODEL_DIR, "dnabert2.pt")
        if os.path.exists(bert_path):
            from models.dnabert2_transformer import DNABERT2Model
            ckpt = torch.load(bert_path, map_location=DEVICE, weights_only=True)
            vs   = ckpt.get("vocab_size", len(vocab))
            model = DNABERT2Model(vocab_size=vs).to(DEVICE)
            model.load_state_dict(ckpt["model"])
            models["bert"] = model
            print("  ✅ DNABERT-2 loaded")

        ae_path = os.path.join(MODEL_DIR, "denoising_ae.pt")
        if os.path.exists(ae_path):
            from models.denoising_autoencoder import DenoisingAutoencoder
            model = DenoisingAutoencoder().to(DEVICE)
            model.load_state_dict(
                torch.load(ae_path, map_location=DEVICE, weights_only=True)
            )
            models["ae"] = model
            print("  ✅ Denoising AE loaded")

        lstm_path = os.path.join(MODEL_DIR, "lstm.pt")
        if os.path.exists(lstm_path):
            from models.lstm_predictor import BiLSTMPredictor
            model = BiLSTMPredictor().to(DEVICE)
            model.load_state_dict(
                torch.load(lstm_path, map_location=DEVICE, weights_only=True)
            )
            models["lstm"] = model
            print("  ✅ BiLSTM loaded")

        # Load sequences
        meta_path = os.path.join(_PROJECT, "data", "sequences", "metadata.json")
        if os.path.exists(meta_path):
            from data.fetch_sequences import load_fasta
            with open(meta_path) as f:
                metadata = json.load(f)
            for name, info in metadata.items():
                if os.path.exists(info["path"]):
                    records = load_fasta(info["path"])
                    sequences[name] = next(iter(records.values()), "")

        set_models(models, vocab, sequences)
        print(f"\n  Models loaded: {list(models.keys())}")
        print(f"  Sequences loaded: {len(sequences)}")
        print(f"  Device: {DEVICE}")

    except Exception as e:
        print(f"  [STARTUP WARN] Could not load models: {e}")
        print("  API will start but reconstruction unavailable until training runs.")

    print(f"\n  🌐 API ready at http://{API_HOST}:{API_PORT}/docs\n")


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_api()
