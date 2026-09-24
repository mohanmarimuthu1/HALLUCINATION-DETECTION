import json

from halludetect.detect.schemas import Claim, ClaimType, Label
from halludetect.detect.verify import verify_claims
from halludetect.evidence.base import Evidence
from halludetect.llm.fake import FakeProvider, fake_response


def _verdict_response(verdicts: list[dict]) -> "fake_response":
    return fake_response(json.dumps({"claim_verdicts": verdicts}))


def _claim(claim_id: str, text: str = "some claim") -> Claim:
    return Claim(claim_id=claim_id, text=text, claim_type=ClaimType.FACTUAL)


def test_no_claims_returns_empty_without_calling_provider():
    provider = FakeProvider([fake_response("should never be used")])
    assert verify_claims(provider, [], []) == []
    assert provider.call_count == 0


def test_joins_verdicts_by_claim_id_even_when_returned_out_of_order():
    claims = [_claim("claim-0", "first"), _claim("claim-1", "second")]
    evidence = [Evidence(chunk_id="e1", text="evidence text", source="direct")]
    provider = FakeProvider(
        [
            _verdict_response(
                [
                    {
                        "claim_id": "claim-1",
                        "label": "CONTRADICTED",
                        "confidence": 0.8,
                        "evidence_chunk_ids": ["e1"],
                        "quote": "evidence text",
                    },
                    {
                        "claim_id": "claim-0",
                        "label": "SUPPORTED",
                        "confidence": 0.9,
                        "evidence_chunk_ids": ["e1"],
                        "quote": "evidence text",
                    },
                ]
            )
        ]
    )

    results = verify_claims(provider, claims, evidence)

    assert [r.claim_id for r in results] == ["claim-0", "claim-1"]
    assert results[0].label == Label.SUPPORTED
    assert results[1].label == Label.CONTRADICTED


def test_missing_verdict_for_a_claim_defaults_to_not_enough_info():
    """v1's _parse_batch_verification fell back to matching a verdict to a
    claim by raw line index when an explicit marker was missing - a
    dropped line silently verified the wrong claim. Here, a claim with no
    matching claim_id in the model's response must get an explicit
    NOT_ENOUGH_INFO default, never a neighboring claim's verdict.
    """
    claims = [_claim("claim-0"), _claim("claim-1")]
    provider = FakeProvider(
        [
            _verdict_response(
                [
                    {
                        "claim_id": "claim-0",
                        "label": "SUPPORTED",
                        "confidence": 0.9,
                        "evidence_chunk_ids": ["e1"],
                        "quote": "text",
                    }
                ]
            )
        ]
    )

    results = verify_claims(provider, claims, [])

    assert len(results) == 2
    assert results[1].claim_id == "claim-1"
    assert results[1].label == Label.NOT_ENOUGH_INFO
    assert results[1].confidence == 0.0


def test_unknown_claim_id_in_response_is_ignored():
    claims = [_claim("claim-0")]
    provider = FakeProvider(
        [
            _verdict_response(
                [
                    {
                        "claim_id": "claim-not-asked-about",
                        "label": "SUPPORTED",
                        "confidence": 0.9,
                        "evidence_chunk_ids": [],
                        "quote": "",
                    }
                ]
            )
        ]
    )

    results = verify_claims(provider, claims, [])

    assert len(results) == 1
    assert results[0].claim_id == "claim-0"
    assert results[0].label == Label.NOT_ENOUGH_INFO
