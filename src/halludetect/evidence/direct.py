"""Caller-supplied evidence (docs/contract.md: request.evidence).

Split into bounded-size chunks so verification (Phase 4) can cite one
chunk at a time instead of one unbounded blob per evidence string.
"""
from halludetect.evidence.base import Evidence
from halludetect.evidence.chunk import chunk_text


class DirectEvidence:
    """EvidenceSource backed by evidence the caller supplied directly.

    Per docs/contract.md ("if evidence is a non-empty array, it is
    authoritative"), `query` is accepted only to satisfy the
    `EvidenceSource` protocol and is otherwise ignored - there is nothing
    to search for, the caller already told us exactly what to verify
    against.
    """

    def __init__(self, evidence: list[str], *, max_chunk_chars: int = 1000) -> None:
        self._evidence = evidence
        self._max_chunk_chars = max_chunk_chars

    def fetch(self, query: str) -> list[Evidence]:
        chunks: list[Evidence] = []
        for i, raw in enumerate(self._evidence):
            pieces = chunk_text(raw, self._max_chunk_chars)
            if not pieces:
                continue
            if len(pieces) == 1:
                chunks.append(Evidence(chunk_id=f"direct-{i}", text=pieces[0], source="direct"))
            else:
                for j, piece in enumerate(pieces):
                    chunks.append(Evidence(chunk_id=f"direct-{i}-{j}", text=piece, source="direct"))
        return chunks
