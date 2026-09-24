"""Shared schemas for the detection pipeline (Phase 4) and the frozen
`AnalysisResult` response contract (docs/contract.md).

Pydantic models throughout, not dataclasses, because `ExtractedClaims` and
`RawVerdict` are passed straight to
`halludetect.llm.structured.complete_structured`, which needs
`model_json_schema()`/`model_validate()` - the plain dataclasses used in
`llm/base.py` and `evidence/base.py` don't support that.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    FACTUAL = "FACTUAL"
    OPINION = "OPINION"
    INSTRUCTION = "INSTRUCTION"
    META = "META"


class Label(str, Enum):
    """Per-claim verification label. Deliberately smaller than `Verdict`
    below - there is no per-claim NOT_VERIFIABLE (docs/contract.md); that
    concept only applies to the whole response.
    """

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Verdict(str, Enum):
    GROUNDED = "GROUNDED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


class ExtractedClaim(BaseModel):
    """One claim as returned by the extraction LLM call, before a stable
    claim_id is assigned (see claims.py - claim_id is never model-
    generated, to avoid collisions or omissions).
    """

    text: str
    claim_type: ClaimType


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]


class Claim(BaseModel):
    """An extracted claim with its service-assigned, guaranteed-unique id."""

    claim_id: str
    text: str
    claim_type: ClaimType


class RawClaimVerdict(BaseModel):
    """One verdict as returned by the verification LLM call, joined back
    to its `Claim` by `claim_id` - never by position (Phase 4.2).
    """

    claim_id: str
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: list[str]
    quote: str


class RawVerdict(BaseModel):
    claim_verdicts: list[RawClaimVerdict]


class ClaimResult(BaseModel):
    claim_id: str
    text: str
    label: Label
    confidence: float
    evidence_chunk_ids: list[str]
    quote: str
    quote_verified: bool


class ModelUsed(BaseModel):
    provider: str
    model: str


class Timings(BaseModel):
    total: int
    retrieval: int
    extraction: int
    verification: int


class AnalysisResult(BaseModel):
    request_id: str
    verdict: Verdict
    p_hallucinated: float
    groundedness: float
    groundedness_ci: tuple[float, float]
    claims: list[ClaimResult]
    n_verifiable_claims: int
    model_used: ModelUsed
    cost_usd: float
    timings_ms: Timings
    calibration_version: str


class Signals(BaseModel):
    """Intermediate per-request counts fuse.py derives scoring from - kept
    separate from AnalysisResult so scoring logic can be unit-tested
    against the counts directly, without constructing a full result.
    """

    n_verifiable_claims: int
    supported: int
    contradicted: int
    not_enough_info: int
