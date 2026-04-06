"""
schemas.py
==========
Pydantic request/response models for the FastAPI genome reconstruction API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class DNAFragment(BaseModel):
    """A single DNA fragment."""
    seq:    str = Field(..., description="DNA sequence (ACGTN)")
    start:  int = Field(0, description="Start position in reference")
    end:    int = Field(0, description="End position in reference")


class DNAUploadRequest(BaseModel):
    """Upload DNA fragments for reconstruction."""
    fragments:    List[DNAFragment] = Field(
        ..., description="List of DNA fragments to reconstruct"
    )
    species_hint: Optional[str] = Field(
        None, description="Hint for species (e.g. 'neanderthal_mtDNA')"
    )
    reference:    Optional[str] = Field(
        None, description="Optional reference sequence for comparison"
    )


class RepairDetail(BaseModel):
    """Detail of a single base repair."""
    global_pos: int
    original:   str
    repaired:   str
    confidence: float
    action:     str


class ReconstructionResponse(BaseModel):
    """Response with reconstructed genome + confidence + comparison."""
    reconstructed_sequence: str = Field(
        ..., description="Full reconstructed genome sequence"
    )
    sequence_length:        int
    confidence_scores:      List[float] = Field(
        ..., description="Per-position confidence [0, 1]"
    )
    mean_confidence:        float
    coverage:               float = Field(
        ..., description="Fraction of sequence that is non-N"
    )
    gaps_before:            int
    gaps_after:             int
    repair_details:         List[RepairDetail] = Field(
        default_factory=list,
        description="Details of each base repair performed"
    )
    metrics:                Optional[Dict] = Field(
        None, description="Evaluation metrics if reference provided"
    )
    evolutionary_comparison: Optional[Dict] = Field(
        None, description="Comparison to related species"
    )


class SpeciesInfo(BaseModel):
    """Information about a reference species."""
    name:      str
    accession: str
    length:    int
    source:    str
    type:      str  # "ancient" or "modern"


class SpeciesListResponse(BaseModel):
    """List of available reference species."""
    species: List[SpeciesInfo]
    total:   int


class HealthResponse(BaseModel):
    """API health check."""
    status:    str = "healthy"
    version:   str = "2.0.0"
    device:    str
    models_loaded: bool
