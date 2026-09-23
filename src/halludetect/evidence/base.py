"""Evidence acquisition protocol.

Every evidence source (caller-supplied, web search, none, a caller's own
retriever) must satisfy this so the detection pipeline (Phase 4) can verify
against whichever evidence source a request selects (docs/contract.md:
`evidence_source`) without caring which concrete implementation it is.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    source: str


@runtime_checkable
class EvidenceSource(Protocol):
    def fetch(self, query: str) -> list[Evidence]:
        """Return evidence chunks relevant to query.

        An empty list means no evidence is available. Callers must treat
        that as grounds for NOT_VERIFIABLE - never fall back to the LLM's
        own world knowledge to fill the gap (plan.md non-negotiable rule).
        """
        ...
