"""Regression tests for every parser bug in v1's detection/ module
(Phase 4 exit criteria, plan.md). Each test is run against two distinct
providers to confirm the fix isn't accidentally tied to one model's
output quirks.

v1 bugs this file guards against:

1. Substring verdict matching (detection/fact_verifier.py:
   `_parse_single_line`) - `"SUPPORTED" in line_upper` matches the
   substring "SUPPORTED" inside "UNSUPPORTED" too, so a NOT_ENOUGH_INFO
   or CONTRADICTED explanation containing the word "unsupported" was
   silently reported as SUPPORTED.
2. Position-based verdict join (detection/fact_verifier.py:
   `_parse_batch_verification`) - fell back to `lines[i]` by raw index
   when an explicit `CLAIM_N` marker was missing, so a dropped/reordered
   line silently verified the wrong claim.
3. No quote grounding, and an explicit instruction to fall back to the
   model's own general knowledge (detection/fact_verifier.py's prompt:
   "If the EVIDENCE doesn't answer it, rely on your vast general
   knowledge... output SUPPORTED") - the exact non-negotiable defect
   plan.md/CLAUDE.md exist to prevent.
4. A second, undocumented hardcoded claim cap
   (detection/claim_extractor.py: `_parse_claims` returns `claims[:2]`
   regardless of any caller-configured limit).
"""
import json

from halludetect.detect.claims import extract_claims
from halludetect.detect.pipeline import run as run_pipeline
from halludetect.detect.quote_check import quote_is_grounded
from halludetect.detect.schemas import Claim, ClaimType, Label, ModelUsed, Verdict
from halludetect.detect.verify import verify_claims
from halludetect.evidence.base import Evidence
from halludetect.llm.fake import FakeProvider, fake_response

_PROVIDERS = [
    lambda script: FakeProvider(script, provider_name="gemini", model="gemini-pro"),
    lambda script: FakeProvider(script, provider_name="openai", model="gpt-4o-mini"),
]


class _StaticEvidenceSource:
    def __init__(self, evidence: list[Evidence]):
        self._evidence = evidence

    def fetch(self, query: str) -> list[Evidence]:
        return self._evidence


def test_label_is_never_matched_by_substring():
    """v1 bug 1: `"SUPPORTED" in "UNSUPPORTED"` is True. The strict
    Pydantic enum used for RawClaimVerdict.label must reject an invented
    value like "UNSUPPORTED" outright rather than silently coercing it
    to SUPPORTED via substring containment - complete_structured's
    schema validation (Phase 2.4) is what makes this structurally
    impossible in v2.
    """
    claims = [Claim(claim_id="claim-0", text="claim", claim_type=ClaimType.FACTUAL)]

    for make_provider in _PROVIDERS:
        # First attempt returns an invalid label; complete_structured's
        # repair-retry then gets a valid one - the invalid value is never
        # silently accepted as SUPPORTED along the way.
        provider = make_provider(
            [
                fake_response(
                    json.dumps(
                        {
                            "claim_verdicts": [
                                {
                                    "claim_id": "claim-0",
                                    "label": "UNSUPPORTED",
                                    "confidence": 0.5,
                                    "evidence_chunk_ids": [],
                                    "quote": "",
                                }
                            ]
                        }
                    )
                ),
                fake_response(
                    json.dumps(
                        {
                            "claim_verdicts": [
                                {
                                    "claim_id": "claim-0",
                                    "label": "NOT_ENOUGH_INFO",
                                    "confidence": 0.5,
                                    "evidence_chunk_ids": [],
                                    "quote": "",
                                }
                            ]
                        }
                    )
                ),
            ]
        )

        results = verify_claims(provider, claims, [])

        assert results[0].label == Label.NOT_ENOUGH_INFO
        assert results[0].label != Label.SUPPORTED


def test_verdict_join_by_claim_id_survives_missing_and_reordered_entries():
    """v1 bug 2: a dropped or reordered response line silently verified
    the wrong claim via positional fallback. Here, claim-1's verdict is
    omitted and claim-2/claim-0 are returned out of order; every claim
    must still get its own, correctly attributed result.
    """
    claims = [
        Claim(claim_id="claim-0", text="first", claim_type=ClaimType.FACTUAL),
        Claim(claim_id="claim-1", text="second", claim_type=ClaimType.FACTUAL),
        Claim(claim_id="claim-2", text="third", claim_type=ClaimType.FACTUAL),
    ]

    for make_provider in _PROVIDERS:
        provider = make_provider(
            [
                fake_response(
                    json.dumps(
                        {
                            "claim_verdicts": [
                                {
                                    "claim_id": "claim-2",
                                    "label": "CONTRADICTED",
                                    "confidence": 0.7,
                                    "evidence_chunk_ids": [],
                                    "quote": "",
                                },
                                {
                                    "claim_id": "claim-0",
                                    "label": "SUPPORTED",
                                    "confidence": 0.9,
                                    "evidence_chunk_ids": [],
                                    "quote": "",
                                },
                                # claim-1 has no entry at all.
                            ]
                        }
                    )
                )
            ]
        )

        results = verify_claims(provider, claims, [])
        by_id = {r.claim_id: r for r in results}

        assert by_id["claim-0"].label == Label.SUPPORTED
        assert by_id["claim-1"].label == Label.NOT_ENOUGH_INFO
        assert by_id["claim-2"].label == Label.CONTRADICTED


def test_no_evidence_never_falls_back_to_model_world_knowledge():
    """v1 bug 3: fact_verifier.py's own prompt told the model to answer
    SUPPORTED from general knowledge when evidence didn't cover a claim.
    In v2 this is structural: the pipeline forces NOT_VERIFIABLE the
    moment evidence acquisition returns nothing, before any claim
    extraction or verification call is made at all.
    """
    for make_provider in _PROVIDERS:
        provider = make_provider([fake_response("this should never be reached")])

        result = run_pipeline(
            answer="The Eiffel Tower is 330 meters tall.",
            question=None,
            evidence_source=_StaticEvidenceSource([]),
            provider=provider,
            request_id="req-parsers",
            model_used=ModelUsed(provider="test", model="test-model"),
        )

        assert result.verdict == Verdict.NOT_VERIFIABLE
        assert provider.call_count == 0


def test_claim_cap_is_driven_by_caller_config_not_a_second_hidden_limit():
    """v1 bug 4: `_parse_claims` capped at `claims[:2]` regardless of any
    caller-supplied limit, on top of the documented cap already applied
    elsewhere. Confirm max_claims is the only cap and it is honored
    exactly, not silently overridden to a smaller hardcoded value.
    """
    raw_claims = [{"text": f"Fact {i}.", "claim_type": "FACTUAL"} for i in range(8)]

    for make_provider in _PROVIDERS:
        provider = make_provider([fake_response(json.dumps({"claims": raw_claims}))])

        claims = extract_claims(provider, "irrelevant", max_claims=6)

        assert len(claims) == 6


def test_supported_requires_a_verified_quote_not_just_a_label():
    """v1 never checked a quote against evidence at all - a SUPPORTED
    verdict was accepted purely on the model's word. Confirm quote_check
    actually rejects a quote that isn't present in the cited evidence.
    """
    chunks = {"e1": "The bridge was built in 1932."}
    assert quote_is_grounded("The bridge was built in 1999", ["e1"], chunks) is False
    assert quote_is_grounded("The bridge was built in 1932.", ["e1"], chunks) is True
