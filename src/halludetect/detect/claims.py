"""Typed claim extraction (Phase 4.1).

Extracts atomic claims from an answer and tags each with a ClaimType. Only
FACTUAL claims are ever sent to verify.py - OPINION/INSTRUCTION/META
claims aren't checkable against evidence and must never be silently
treated as if they were.

`claim_id` is assigned here, by us, after extraction - never returned by
the model. v1's claim_extractor.py had no id concept at all (claims were
plain strings, joined to verdicts later by array position); a locally
assigned sequential id (`claim-0`, `claim-1`, ...) can't collide or be
omitted the way a model-generated one could, and is what verify.py's
join-by-claim_id (4.2) actually joins against.

`max_claims` is the only cap and it is always honored exactly - v1 also
had a cap, but a second, undocumented one (`claims[:2]`) hardcoded deeper
in `_parse_claims` regardless of what the caller configured; see
tests/test_parsers.py.
"""
from __future__ import annotations

from halludetect.detect.schemas import Claim, ExtractedClaims
from halludetect.llm.base import LLMProvider
from halludetect.llm.structured import complete_structured

DEFAULT_MAX_CLAIMS = 12

_EXTRACTION_PROMPT = """Extract the individual, atomic claims made in ANSWER below.

For each claim, classify its type:
- FACTUAL: a checkable factual assertion about the world.
- OPINION: a subjective judgment, not checkable against evidence.
- INSTRUCTION: a directive/how-to step, not a factual assertion.
- META: a statement about the answer itself (e.g. "I'm not sure, but...").

Do not merge multiple distinct facts into one claim. Do not invent claims
the answer doesn't make.
{question_line}
ANSWER:
{answer}"""


def _build_prompt(answer: str, question: str | None) -> str:
    question_line = f"\nQUESTION (for context only): {question}\n" if question else ""
    return _EXTRACTION_PROMPT.format(question_line=question_line, answer=answer)


def extract_claims(
    provider: LLMProvider,
    answer: str,
    *,
    question: str | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> list[Claim]:
    """Returns at most `max_claims` claims, each with a locally assigned,
    collision-free claim_id.
    """
    prompt = _build_prompt(answer, question)
    instance, _ = complete_structured(provider, prompt, ExtractedClaims)
    extracted = instance.claims[:max_claims]
    return [
        Claim(claim_id=f"claim-{i}", text=c.text, claim_type=c.claim_type)
        for i, c in enumerate(extracted)
    ]
