"""Explicit no-evidence path.

Used for `evidence_source: none` (docs/contract.md), and as the required
fallback when `evidence_source: web` is requested but no web-search
provider key is configured - the contract states that case "is treated
identically to none".

Returning no evidence is deliberate, not a stub: it is what the detection
pipeline (Phase 4) uses to force `verdict: NOT_VERIFIABLE`,
`reason: no_evidence_configured`, instead of any code path silently
answering from the underlying LLM's own world knowledge (plan.md
non-negotiable rule).
"""
from halludetect.evidence.base import Evidence


class NoEvidenceSource:
    def fetch(self, query: str) -> list[Evidence]:
        return []
