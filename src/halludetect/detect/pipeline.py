"""Detection pipeline orchestration (Phase 4).

Wires together, in order: evidence acquisition -> typed claim extraction
(4.1) -> structured verification joined by claim_id (4.2) -> quote-
grounding downgrade (4.3) -> calibrated fusion (4.4). This is the first
place Phase 2's `LLMProvider`/`complete_structured` and Phase 3's
`EvidenceSource` implementations are actually called together - neither
had a real caller before this module.

The no-evidence rule is enforced here, not just documented: if
`evidence_source.fetch()` returns an empty list, extraction and
verification are skipped entirely and the result is `NOT_VERIFIABLE`.
This is deliberate defense in depth - even if a future API-layer caller
forgot docs/contract.md rule 2, this module cannot produce anything else
when there is no evidence to ground against.

`provider` is a single, already-resolved `LLMProvider` for the whole
request - which concrete model that is (routing via Phase 2's `Router`,
picking a model per free-pool rotation) is the API layer's job (Phase 5),
not this module's; wiring one provider per request keeps `model_used` in
the response attributable to one actual model rather than possibly two
different free-pool picks across the extraction and verification calls.
"""
from __future__ import annotations

from time import monotonic

from halludetect.detect.claims import DEFAULT_MAX_CLAIMS, extract_claims
from halludetect.detect.fuse import CALIBRATION_VERSION, fuse, summarize
from halludetect.detect.quote_check import quote_is_grounded
from halludetect.detect.schemas import (
    AnalysisResult,
    Claim,
    ClaimResult,
    ClaimType,
    Label,
    ModelUsed,
    Timings,
    Verdict,
)
from halludetect.detect.verify import verify_claims
from halludetect.evidence.base import Evidence, EvidenceSource
from halludetect.llm.base import LLMProvider
from halludetect.llm.pricing import estimate_cost_usd
from halludetect.llm.usage import UsageTrackingProvider


def _elapsed_ms(start: float) -> int:
    return int((monotonic() - start) * 1000)


def _not_verifiable_result(
    request_id: str,
    model_used: ModelUsed,
    cost_usd: float,
    timings: Timings,
) -> AnalysisResult:
    return AnalysisResult(
        request_id=request_id,
        verdict=Verdict.NOT_VERIFIABLE,
        p_hallucinated=1.0,
        groundedness=0.0,
        groundedness_ci=(0.0, 0.0),
        claims=[],
        n_verifiable_claims=0,
        model_used=model_used,
        cost_usd=cost_usd,
        timings_ms=timings,
        calibration_version=CALIBRATION_VERSION,
    )


def _verify_and_ground(
    provider: LLMProvider,
    factual_claims: list[Claim],
    evidence: list[Evidence],
) -> list[ClaimResult]:
    if not factual_claims:
        return []

    chunks_by_id = {e.chunk_id: e.text for e in evidence}
    raw_verdicts = verify_claims(provider, factual_claims, evidence)

    results = []
    for claim, raw in zip(factual_claims, raw_verdicts):
        grounded = quote_is_grounded(raw.quote, raw.evidence_chunk_ids, chunks_by_id)
        label = raw.label
        if label == Label.SUPPORTED and not grounded:
            label = Label.NOT_ENOUGH_INFO
        results.append(
            ClaimResult(
                claim_id=claim.claim_id,
                text=claim.text,
                label=label,
                confidence=raw.confidence,
                evidence_chunk_ids=raw.evidence_chunk_ids,
                quote=raw.quote,
                quote_verified=grounded,
            )
        )
    return results


def run(
    *,
    answer: str,
    question: str | None,
    evidence_source: EvidenceSource,
    provider: LLMProvider,
    request_id: str,
    model_used: ModelUsed,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> AnalysisResult:
    total_start = monotonic()
    # Wraps `provider` so every complete() call made during this request -
    # extraction, verification, and any complete_structured repair retries
    # in between - is counted for cost_usd (Phase 5.3), without claims.py/
    # verify.py/structured.py needing to know cost tracking exists.
    tracked_provider = UsageTrackingProvider(provider)

    retrieval_start = monotonic()
    evidence = evidence_source.fetch(question or answer)
    retrieval_ms = _elapsed_ms(retrieval_start)

    if not evidence:
        timings = Timings(
            total=_elapsed_ms(total_start),
            retrieval=retrieval_ms,
            extraction=0,
            verification=0,
        )
        cost_usd = estimate_cost_usd(model_used.provider, model_used.model, tracked_provider.usage)
        return _not_verifiable_result(request_id, model_used, cost_usd, timings)

    extraction_start = monotonic()
    claims = extract_claims(tracked_provider, answer, question=question, max_claims=max_claims)
    factual_claims = [c for c in claims if c.claim_type == ClaimType.FACTUAL]
    extraction_ms = _elapsed_ms(extraction_start)

    verification_start = monotonic()
    claim_results = _verify_and_ground(tracked_provider, factual_claims, evidence)
    verification_ms = _elapsed_ms(verification_start)

    signals = summarize(claim_results)
    verdict, p_hallucinated, groundedness, ci = fuse(signals)

    timings = Timings(
        total=_elapsed_ms(total_start),
        retrieval=retrieval_ms,
        extraction=extraction_ms,
        verification=verification_ms,
    )
    cost_usd = estimate_cost_usd(model_used.provider, model_used.model, tracked_provider.usage)

    return AnalysisResult(
        request_id=request_id,
        verdict=verdict,
        p_hallucinated=p_hallucinated,
        groundedness=groundedness,
        groundedness_ci=ci,
        claims=claim_results,
        n_verifiable_claims=signals.n_verifiable_claims,
        model_used=model_used,
        cost_usd=cost_usd,
        timings_ms=timings,
        calibration_version=CALIBRATION_VERSION,
    )
