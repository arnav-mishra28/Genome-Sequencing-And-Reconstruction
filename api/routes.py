"""
routes.py
=========
FastAPI route definitions.

Endpoints:
  POST /api/v1/reconstruct  — Upload DNA fragments → reconstructed genome
  GET  /api/v1/species      — List available reference species
  POST /api/v1/compare      — Compare reconstruction to reference
  GET  /api/v1/health       — Health check
"""

import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional

from api.schemas import (
    DNAUploadRequest,
    ReconstructionResponse,
    RepairDetail,
    SpeciesListResponse,
    SpeciesInfo,
    HealthResponse,
)
from config.settings import (
    DEVICE, NCBI_SEQUENCES, MODERN_SPECIES, REF_MAP, PHYLO_DISTANCES,
)

router = APIRouter(prefix="/api/v1", tags=["genome"])

# ── Global model store (populated at startup) ─────────────────────────────────
_models: Dict = {}
_vocab:  Dict = {}
_sequences: Dict = {}


def set_models(models: Dict, vocab: Dict, sequences: Dict):
    """Called at startup to inject models into the router."""
    global _models, _vocab, _sequences
    _models    = models
    _vocab     = vocab
    _sequences = sequences


# ═══════════════════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        device=str(DEVICE),
        models_loaded=bool(_models),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Species list
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/species", response_model=SpeciesListResponse)
async def list_species():
    species = []
    for name, acc in NCBI_SEQUENCES.items():
        seq = _sequences.get(name, "")
        species.append(SpeciesInfo(
            name=name,
            accession=acc,
            length=len(seq),
            source="ncbi",
            type="modern" if name in MODERN_SPECIES else "ancient",
        ))
    return SpeciesListResponse(species=species, total=len(species))


# ═══════════════════════════════════════════════════════════════════════════════
#  Reconstruct
# ═══════════════════════════════════════════════════════════════════════════════
@router.post("/reconstruct", response_model=ReconstructionResponse)
async def reconstruct(request: DNAUploadRequest):
    if not _models:
        raise HTTPException(503, "Models not loaded. Run training first.")

    # Concatenate fragments
    combined = "".join(f.seq for f in request.fragments)
    if not combined:
        raise HTTPException(400, "No sequence data in fragments.")

    bert_model = _models.get("bert")
    ae_model   = _models.get("ae")
    lstm_model = _models.get("lstm")

    if not all([bert_model, ae_model, lstm_model]):
        raise HTTPException(503, "Not all models are loaded.")

    from models.ensemble_reconstructor import ensemble_reconstruct
    recon_seq, confidences, details = ensemble_reconstruct(
        sequence=combined,
        bert_model=bert_model,
        ae_model=ae_model,
        lstm_model=lstm_model,
        vocab=_vocab,
    )

    # Run metrics if reference provided
    metrics = None
    if request.reference:
        from evaluation.metrics import evaluate_reconstruction
        exp_dist = 90.0
        if request.species_hint:
            for (sp1, sp2), d in PHYLO_DISTANCES.items():
                if sp1 == request.species_hint or sp2 == request.species_hint:
                    exp_dist = d
                    break

        ref_name = REF_MAP.get(request.species_hint, "human_mtDNA")
        rel_seq  = _sequences.get(ref_name, "")

        metrics = evaluate_reconstruction(
            reconstructed=recon_seq,
            reference=request.reference,
            relative_seq=rel_seq,
            expected_distance=exp_dist,
            confidences=confidences,
            species_name=request.species_hint or "unknown",
        )

    # Evolutionary comparison
    evo_comp = None
    if request.species_hint and request.species_hint in REF_MAP:
        ref_name = REF_MAP[request.species_hint]
        ref_seq  = _sequences.get(ref_name, "")
        if ref_seq:
            from evaluation.metrics import phylogenetic_consistency
            exp_dist = 90.0
            for (sp1, sp2), d in PHYLO_DISTANCES.items():
                if sp1 == request.species_hint:
                    exp_dist = d
                    break
            evo_comp = phylogenetic_consistency(
                recon_seq, ref_seq, exp_dist, request.species_hint,
            )

    repairs = [
        RepairDetail(**r)
        for r in details.get("ae_repairs", [])[:50]
    ]

    return ReconstructionResponse(
        reconstructed_sequence=recon_seq[:10000],  # cap for response size
        sequence_length=len(recon_seq),
        confidence_scores=confidences[:10000],
        mean_confidence=details.get("mean_confidence", 0.5),
        coverage=details.get("coverage", 0.0),
        gaps_before=details.get("gaps_before", 0),
        gaps_after=details.get("gaps_after", 0),
        repair_details=repairs,
        metrics=metrics,
        evolutionary_comparison=evo_comp,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Compare
# ═══════════════════════════════════════════════════════════════════════════════
@router.post("/compare")
async def compare(request: DNAUploadRequest):
    """Compare uploaded fragments directly to reference."""
    if not request.reference:
        raise HTTPException(400, "Reference sequence required for comparison.")

    combined = "".join(f.seq for f in request.fragments)
    from evaluation.metrics import (
        sequence_accuracy, edit_distance, reconstruction_similarity,
    )

    return {
        "accuracy":   sequence_accuracy(combined, request.reference),
        "edit_dist":  edit_distance(combined, request.reference),
        "similarity": reconstruction_similarity(combined, request.reference),
    }
