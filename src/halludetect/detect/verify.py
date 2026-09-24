"""Structured-JSON claim verification (Phase 4.2).

Sends only FACTUAL claims (claims.py already filtered out
OPINION/INSTRUCTION/META) plus the evidence chunks to the LLM, gets back a
`RawVerdict`, and joins each `RawClaimVerdict` back to its `Claim` by
`claim_id` - never by array position.

v1's fact_verifier.py parsed one verdict line per claim by looking for a
`CLAIM_N` marker, and when that marker was missing fell back to matching
by raw line index (`lines[i]`) - so a dropped, reordered, or blank line in
the model's output silently verified the wrong claim. Joining by an
explicit `claim_id` the model must echo back removes that failure mode
entirely: if the model doesn't return a verdict for a given claim_id, that
claim gets an explicit NOT_ENOUGH_INFO default here, never a claim it
never actually addressed.
"""
from __future__ import annotations

from halludetect.detect.schemas import Claim, Label, RawClaimVerdict, RawVerdict
from halludetect.evidence.base import Evidence
from halludetect.llm.base import LLMProvider
from halludetect.llm.structured import complete_structured

_VERIFY_PROMPT = """Verify each CLAIM against EVIDENCE. Use ONLY the evidence given below -
never your own general knowledge. If the evidence doesn't address a claim,
its label MUST be NOT_ENOUGH_INFO, even if you personally know the answer.

For each claim, return:
- claim_id: copied exactly from the CLAIMS list below.
- label: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO.
- confidence: 0.0-1.0.
- evidence_chunk_ids: the chunk_id(s) (from EVIDENCE below) that support your label.
- quote: a VERBATIM quote copied from one cited evidence chunk. Required for
  SUPPORTED and CONTRADICTED; empty string if NOT_ENOUGH_INFO.

EVIDENCE:
{evidence_block}

CLAIMS:
{claims_block}"""


def _format_evidence(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence) or "(none)"


def _format_claims(claims: list[Claim]) -> str:
    return "\n".join(f"[{c.claim_id}] {c.text}" for c in claims)


def _default_verdict(claim: Claim) -> RawClaimVerdict:
    return RawClaimVerdict(
        claim_id=claim.claim_id,
        label=Label.NOT_ENOUGH_INFO,
        confidence=0.0,
        evidence_chunk_ids=[],
        quote="",
    )


def verify_claims(
    provider: LLMProvider,
    claims: list[Claim],
    evidence: list[Evidence],
) -> list[RawClaimVerdict]:
    """Returns exactly one `RawClaimVerdict` per input claim, in input
    order, joined by claim_id - never fewer or more than `len(claims)`,
    and never assigned by position.
    """
    if not claims:
        return []

    prompt = _VERIFY_PROMPT.format(
        evidence_block=_format_evidence(evidence),
        claims_block=_format_claims(claims),
    )
    instance, _ = complete_structured(provider, prompt, RawVerdict)

    by_id = {v.claim_id: v for v in instance.claim_verdicts}
    return [by_id.get(claim.claim_id) or _default_verdict(claim) for claim in claims]
