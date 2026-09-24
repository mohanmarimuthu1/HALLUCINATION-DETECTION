import json

from halludetect.detect.claims import extract_claims
from halludetect.detect.schemas import ClaimType
from halludetect.llm.fake import FakeProvider, fake_response


def _claims_response(claims: list[dict]) -> "fake_response":
    return fake_response(json.dumps({"claims": claims}))


def test_extracts_typed_claims_with_locally_assigned_ids():
    provider = FakeProvider(
        [
            _claims_response(
                [
                    {"text": "Paris is the capital of France.", "claim_type": "FACTUAL"},
                    {"text": "This is a great question.", "claim_type": "OPINION"},
                ]
            )
        ]
    )

    claims = extract_claims(provider, "Paris is the capital of France. This is a great question.")

    assert [c.claim_id for c in claims] == ["claim-0", "claim-1"]
    assert claims[0].claim_type == ClaimType.FACTUAL
    assert claims[1].claim_type == ClaimType.OPINION


def test_max_claims_cap_is_exact_not_hardcoded():
    """v1's _parse_claims silently truncated to 2 claims regardless of any
    caller-configured limit. Verify the cap here is driven entirely by the
    max_claims argument.
    """
    raw_claims = [{"text": f"Fact number {i}.", "claim_type": "FACTUAL"} for i in range(10)]
    provider = FakeProvider([_claims_response(raw_claims)])

    claims = extract_claims(provider, "irrelevant", max_claims=5)

    assert len(claims) == 5
    assert claims[-1].claim_id == "claim-4"


def test_claim_ids_never_collide_across_multiple_claims():
    raw_claims = [{"text": f"Fact {i}.", "claim_type": "FACTUAL"} for i in range(4)]
    provider = FakeProvider([_claims_response(raw_claims)])

    claims = extract_claims(provider, "irrelevant")

    assert len(set(c.claim_id for c in claims)) == len(claims)
