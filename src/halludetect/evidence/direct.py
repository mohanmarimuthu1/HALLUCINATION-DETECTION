"""Caller-supplied evidence (docs/contract.md: request.evidence).

Split into bounded-size chunks so verification (Phase 4) can cite one
chunk at a time instead of one unbounded blob per evidence string.
"""
import re

from halludetect.evidence.base import Evidence

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


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
            pieces = _chunk_text(raw, self._max_chunk_chars)
            if not pieces:
                continue
            if len(pieces) == 1:
                chunks.append(Evidence(chunk_id=f"direct-{i}", text=pieces[0], source="direct"))
            else:
                for j, piece in enumerate(pieces):
                    chunks.append(Evidence(chunk_id=f"direct-{i}-{j}", text=piece, source="direct"))
        return chunks


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack text into chunks no larger than max_chars, splitting
    on paragraph then sentence boundaries. Never truncates or drops
    content - a single sentence longer than max_chars is kept whole and
    the resulting chunk is allowed to exceed max_chars, since losing
    evidence text silently would be worse than an oversized chunk.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()] or [text]

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(s for s in _SENTENCE_SPLIT.split(para) if s)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_chars:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks
