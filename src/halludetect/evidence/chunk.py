"""Shared text chunking for evidence sources backed by caller-supplied
strings (DirectEvidence, CustomEvidenceSource) - both need to bound
chunk size the same way, so the logic lives in one place instead of
being duplicated per source.
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack text into chunks no larger than max_chars, splitting
    on paragraph then sentence boundaries. Never truncates or drops
    content - a single sentence longer than max_chars is kept whole and
    the resulting chunk is allowed to exceed max_chars, since losing
    evidence text silently would be worse than one oversized chunk.
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
