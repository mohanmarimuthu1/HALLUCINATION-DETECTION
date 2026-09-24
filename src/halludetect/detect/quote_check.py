"""Quote-grounding check (Phase 4.3).

A claim can only carry `label: SUPPORTED` if its `quote` is actually,
verifiably present in one of its cited evidence chunks - never taken on
the model's word (docs/contract.md rule 3). This is the direct fix for
v1's fact_verifier.py, which accepted a model's SUPPORTED verdict on the
strength of a freeform "explanation" string alone, with no check against
the evidence text at all, and whose prompt explicitly told the model to
fall back on its "vast general knowledge" when evidence didn't cover a
claim - the exact defect plan.md and CLAUDE.md call out as non-negotiable
to avoid.

Matching is exact substring after whitespace normalization only - no
fuzzy/similarity fallback. A rapidfuzz partial_ratio prototype was tried
and rejected: it scored a quote with a materially wrong number (e.g. "...
built in 1999" against evidence reading "... built in 1932.") at 96/100,
above any threshold that would still catch legitimate paraphrasing. A
quote-grounding check that can rate a factually altered quote as
"grounded" is worse than no fuzzy matching at all, so only whitespace
normalization - which cannot mask a content difference - is applied here.
"""
from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def quote_is_grounded(
    quote: str,
    evidence_chunk_ids: list[str],
    chunks_by_id: dict[str, str],
) -> bool:
    """True only if `quote` is a verbatim substring of at least one cited
    chunk that actually exists, modulo whitespace differences. An empty
    quote, no cited chunks, or a chunk id that doesn't resolve to real
    evidence are all treated as ungrounded - never given the benefit of
    the doubt.
    """
    quote = quote.strip()
    if not quote or not evidence_chunk_ids:
        return False

    normalized_quote = _normalize(quote)
    for chunk_id in evidence_chunk_ids:
        text = chunks_by_id.get(chunk_id)
        if text is None:
            continue
        if quote in text:
            return True
        if normalized_quote in _normalize(text):
            return True

    return False
