from halludetect.detect.fuse import fuse, summarize, wilson_ci
from halludetect.detect.schemas import ClaimResult, Label, Verdict


def _claim_result(label: Label) -> ClaimResult:
    return ClaimResult(
        claim_id="c",
        text="t",
        label=label,
        confidence=0.9,
        evidence_chunk_ids=["e1"],
        quote="q",
        quote_verified=label == Label.SUPPORTED,
    )


def test_wilson_ci_zero_n_returns_zero_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_all_supported_is_narrow_and_high():
    low, high = wilson_ci(10, 10)
    assert low > 0.6
    assert high == 1.0


def test_below_min_verifiable_forces_not_verifiable():
    signals = summarize([_claim_result(Label.SUPPORTED), _claim_result(Label.SUPPORTED)])
    verdict, *_ = fuse(signals)
    assert verdict == Verdict.NOT_VERIFIABLE


def test_zero_claims_forces_not_verifiable():
    signals = summarize([])
    verdict, p_hallucinated, groundedness, ci = fuse(signals)
    assert verdict == Verdict.NOT_VERIFIABLE
    assert groundedness == 0.0
    assert ci == (0.0, 0.0)


def test_all_supported_at_min_threshold_is_grounded():
    claims = [_claim_result(Label.SUPPORTED) for _ in range(3)]
    signals = summarize(claims)
    verdict, p_hallucinated, groundedness, _ci = fuse(signals)
    assert verdict == Verdict.GROUNDED
    assert groundedness == 1.0
    assert p_hallucinated == 0.0


def test_any_contradiction_forces_contradicted_verdict():
    claims = [_claim_result(Label.SUPPORTED), _claim_result(Label.SUPPORTED), _claim_result(Label.CONTRADICTED)]
    signals = summarize(claims)
    verdict, *_ = fuse(signals)
    assert verdict == Verdict.CONTRADICTED


def test_not_enough_info_without_contradiction():
    claims = [
        _claim_result(Label.SUPPORTED),
        _claim_result(Label.SUPPORTED),
        _claim_result(Label.NOT_ENOUGH_INFO),
    ]
    signals = summarize(claims)
    verdict, *_ = fuse(signals)
    assert verdict == Verdict.NOT_ENOUGH_INFO


def test_contradiction_outranks_not_enough_info():
    claims = [
        _claim_result(Label.CONTRADICTED),
        _claim_result(Label.NOT_ENOUGH_INFO),
        _claim_result(Label.SUPPORTED),
    ]
    signals = summarize(claims)
    verdict, *_ = fuse(signals)
    assert verdict == Verdict.CONTRADICTED
