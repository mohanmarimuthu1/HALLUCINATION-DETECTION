import json

from halludetect.detect import pipeline
from halludetect.detect.schemas import Label, ModelUsed, Verdict
from halludetect.evidence.base import Evidence
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.fake import FakeProvider, fake_response

_MODEL_USED = ModelUsed(provider="fake", model="fake-model")


class _StaticEvidenceSource:
    def __init__(self, evidence: list[Evidence]):
        self._evidence = evidence
        self.queries: list[str] = []

    def fetch(self, query: str) -> list[Evidence]:
        self.queries.append(query)
        return self._evidence


def _extraction_response(claims: list[dict]) -> "fake_response":
    return fake_response(json.dumps({"claims": claims}))


def _verification_response(verdicts: list[dict]) -> "fake_response":
    return fake_response(json.dumps({"claim_verdicts": verdicts}))


def test_no_evidence_forces_not_verifiable_without_calling_the_llm():
    """The pipeline's own defense of docs/contract.md rule 2: even if the
    caller forgot to gate on evidence_source, an empty fetch() must never
    reach claim extraction/verification at all.
    """
    provider = FakeProvider([fake_response("should never be called")])
    evidence_source = _StaticEvidenceSource([])

    result = pipeline.run(
        answer="The Eiffel Tower is in Paris.",
        question=None,
        evidence_source=evidence_source,
        provider=provider,
        request_id="req-1",
        model_used=_MODEL_USED,
    )

    assert result.verdict == Verdict.NOT_VERIFIABLE
    assert result.claims == []
    assert result.n_verifiable_claims == 0
    assert provider.call_count == 0


def test_fewer_than_three_verifiable_claims_forces_not_verifiable():
    evidence = [Evidence(chunk_id="e1", text="The Eiffel Tower is in Paris, France.", source="direct")]
    provider = FakeProvider(
        [
            _extraction_response(
                [{"text": "The Eiffel Tower is in Paris.", "claim_type": "FACTUAL"}]
            ),
            _verification_response(
                [
                    {
                        "claim_id": "claim-0",
                        "label": "SUPPORTED",
                        "confidence": 0.95,
                        "evidence_chunk_ids": ["e1"],
                        "quote": "The Eiffel Tower is in Paris, France.",
                    }
                ]
            ),
        ]
    )

    result = pipeline.run(
        answer="The Eiffel Tower is in Paris.",
        question=None,
        evidence_source=_StaticEvidenceSource(evidence),
        provider=provider,
        request_id="req-2",
        model_used=_MODEL_USED,
    )

    assert result.n_verifiable_claims == 1
    assert result.verdict == Verdict.NOT_VERIFIABLE
    assert result.claims[0].label == Label.SUPPORTED
    assert result.claims[0].quote_verified is True


def test_supported_verdict_with_unverified_quote_is_downgraded():
    """docs/contract.md rule 3: SUPPORTED without a verified quote must
    never reach the caller - it is downgraded server-side.
    """
    evidence = [Evidence(chunk_id=f"e{i}", text=f"Fact {i} is true.", source="direct") for i in range(3)]
    claim_texts = [{"text": f"Claim {i}.", "claim_type": "FACTUAL"} for i in range(3)]
    verdicts = [
        {
            "claim_id": f"claim-{i}",
            "label": "SUPPORTED",
            "confidence": 0.9,
            "evidence_chunk_ids": [f"e{i}"],
            "quote": "this quote does not appear anywhere in the evidence",
        }
        for i in range(3)
    ]
    provider = FakeProvider([_extraction_response(claim_texts), _verification_response(verdicts)])

    result = pipeline.run(
        answer="irrelevant",
        question=None,
        evidence_source=_StaticEvidenceSource(evidence),
        provider=provider,
        request_id="req-3",
        model_used=_MODEL_USED,
    )

    assert all(c.label == Label.NOT_ENOUGH_INFO for c in result.claims)
    assert all(c.quote_verified is False for c in result.claims)


def test_grounded_verdict_when_all_claims_supported_with_verified_quotes():
    evidence = [Evidence(chunk_id=f"e{i}", text=f"Fact {i} is true and verifiable.", source="direct") for i in range(3)]
    claim_texts = [{"text": f"Claim {i}.", "claim_type": "FACTUAL"} for i in range(3)]
    verdicts = [
        {
            "claim_id": f"claim-{i}",
            "label": "SUPPORTED",
            "confidence": 0.9,
            "evidence_chunk_ids": [f"e{i}"],
            "quote": f"Fact {i} is true and verifiable.",
        }
        for i in range(3)
    ]
    provider = FakeProvider([_extraction_response(claim_texts), _verification_response(verdicts)])

    result = pipeline.run(
        answer="irrelevant",
        question=None,
        evidence_source=_StaticEvidenceSource(evidence),
        provider=provider,
        request_id="req-4",
        model_used=_MODEL_USED,
    )

    assert result.verdict == Verdict.GROUNDED
    assert result.groundedness == 1.0
    assert result.n_verifiable_claims == 3


def test_only_factual_claims_are_sent_to_verification():
    evidence = [Evidence(chunk_id="e1", text="irrelevant evidence", source="direct")]
    provider = FakeProvider(
        [
            _extraction_response(
                [
                    {"text": "An opinion.", "claim_type": "OPINION"},
                    {"text": "An instruction.", "claim_type": "INSTRUCTION"},
                    {"text": "A meta statement.", "claim_type": "META"},
                ]
            ),
        ]
    )

    result = pipeline.run(
        answer="irrelevant",
        question=None,
        evidence_source=_StaticEvidenceSource(evidence),
        provider=provider,
        request_id="req-5",
        model_used=_MODEL_USED,
    )

    # Only one provider call (extraction) - verification is skipped entirely
    # since there are no FACTUAL claims to verify.
    assert provider.call_count == 1
    assert result.claims == []
    assert result.n_verifiable_claims == 0
    assert result.verdict == Verdict.NOT_VERIFIABLE


def test_cost_usd_is_computed_from_actual_token_usage_not_hardcoded():
    """Phase 5.3: cost_usd must reflect real usage across every call the
    pipeline makes (extraction + verification here), not a caller-supplied
    constant - the pipeline no longer even accepts one.
    """
    evidence = [Evidence(chunk_id="e1", text="The Eiffel Tower is in Paris, France.", source="direct")]

    class _CostReportingProvider:
        def __init__(self, responses):
            self._responses = iter(responses)
            self.call_count = 0

        def complete(self, prompt, *, max_tokens=1024):
            self.call_count += 1
            return next(self._responses)

        def supports_json_schema(self):
            return True

    responses = [
        LLMResponse(
            text=json.dumps({"claims": [{"text": "The Eiffel Tower is in Paris.", "claim_type": "FACTUAL"}]}),
            provider="openrouter",
            model="free/a",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, cost_usd=0.001),
        ),
        LLMResponse(
            text=json.dumps(
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "claim-0",
                            "label": "SUPPORTED",
                            "confidence": 0.95,
                            "evidence_chunk_ids": ["e1"],
                            "quote": "The Eiffel Tower is in Paris, France.",
                        }
                    ]
                }
            ),
            provider="openrouter",
            model="free/a",
            usage=TokenUsage(prompt_tokens=150, completion_tokens=30, cost_usd=0.002),
        ),
    ]
    provider = _CostReportingProvider(responses)

    result = pipeline.run(
        answer="The Eiffel Tower is in Paris.",
        question=None,
        evidence_source=_StaticEvidenceSource(evidence),
        provider=provider,
        request_id="req-6",
        model_used=ModelUsed(provider="openrouter", model="free/a"),
    )

    assert provider.call_count == 2
    assert result.cost_usd == 0.003


def test_cost_usd_is_zero_when_no_evidence_and_no_calls_are_made():
    provider = FakeProvider([fake_response("should never be called")])
    result = pipeline.run(
        answer="irrelevant",
        question=None,
        evidence_source=_StaticEvidenceSource([]),
        provider=provider,
        request_id="req-7",
        model_used=_MODEL_USED,
    )
    assert result.cost_usd == 0.0
