"""Plug-in hook for a caller's own retriever/RAG (docs/contract.md:
evidence_source == 'custom').

Any object already exposing fetch(query) -> list[Evidence] satisfies the
EvidenceSource Protocol directly and needs no adapter. This module covers
the more common case: a caller's retriever returns plain text chunks
(list[str]) for a query, not typed Evidence objects. CustomEvidenceSource
wraps that callable so it can be used anywhere an EvidenceSource is
expected, chunking oversized strings the same way DirectEvidence does.

Unlike WebSearchEvidence, a retriever exception is not caught here and
degraded to no evidence - it is the caller's own plugin, so a bug in it
is the caller's to see and fix, not ours to hide behind a silent empty
result.
"""
from collections.abc import Callable

from halludetect.evidence.base import Evidence
from halludetect.evidence.chunk import chunk_text

DEFAULT_MAX_CHUNK_CHARS = 1000


class CustomEvidenceSource:
    def __init__(
        self,
        retriever: Callable[[str], list[str]],
        *,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        self._retriever = retriever
        self._max_chunk_chars = max_chunk_chars

    def fetch(self, query: str) -> list[Evidence]:
        raw_chunks = self._retriever(query)
        chunks: list[Evidence] = []
        for i, raw in enumerate(raw_chunks):
            pieces = chunk_text(raw, self._max_chunk_chars)
            if not pieces:
                continue
            if len(pieces) == 1:
                chunks.append(Evidence(chunk_id=f"custom-{i}", text=pieces[0], source="custom"))
            else:
                for j, piece in enumerate(pieces):
                    chunks.append(Evidence(chunk_id=f"custom-{i}-{j}", text=piece, source="custom"))
        return chunks
