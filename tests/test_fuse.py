from halludetect.detect.fuse import CALIBRATION_VERSION, P_HALLUCINATED_BY_VERDICT, fuse, rescore, summarize, wilson_ci
from halludetect.detect.schemas import AbstentionReason, AnalysisResult, ClaimResult, Label, ModelUsed, Timings, Verdict


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
    assert p_hallucinated == P_HALLUCINATED_BY_VERDICT[Verdict.GROUNDED]


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


def test_p_hallucinated_orders_verdicts_by_risk():
    rates = P_HALLUCINATED_BY_VERDICT
    assert rates[Verdict.GROUNDED] < rates[Verdict.NOT_ENOUGH_INFO] < rates[Verdict.CONTRADICTED]


def test_fitted_rates_match_committed_golden_sets():
    from halludetect.eval.calibrate import fit

    fitted = {verdict: rate for verdict, (rate, _n) in fit().items()}
    assert fitted == P_HALLUCINATED_BY_VERDICT


def _result(claims: list[ClaimResult], *, version: str, reason=None) -> AnalysisResult:
    return AnalysisResult(
        request_id="r",
        verdict=Verdict.GROUNDED,
        p_hallucinated=0.0,
        groundedness=1.0,
        groundedness_ci=(0.0, 1.0),
        claims=claims,
        n_verifiable_claims=len(claims),
        model_used=ModelUsed(provider="openrouter", model="free/a"),
        cost_usd=0.0,
        timings_ms=Timings(total=1, retrieval=0, extraction=0, verification=0),
        calibration_version=version,
        reason=reason,
    )


def test_rescore_updates_a_result_from_an_older_calibration():
    claims = [_claim_result(Label.SUPPORTED), _claim_result(Label.SUPPORTED), _claim_result(Label.CONTRADICTED)]
    rescored = rescore(_result(claims, version="heuristic-v0"))
    assert rescored.verdict == Verdict.CONTRADICTED
    assert rescored.p_hallucinated == P_HALLUCINATED_BY_VERDICT[Verdict.CONTRADICTED]
    assert rescored.calibration_version == CALIBRATION_VERSION
    assert rescored.claims == claims


def test_rescore_leaves_current_results_alone():
    result = _result([_claim_result(Label.SUPPORTED)] * 3, version=CALIBRATION_VERSION)
    assert rescore(result) is result


def test_rescore_keeps_no_evidence_results_not_verifiable():
    result = _result([], version="heuristic-v0", reason=AbstentionReason.NO_EVIDENCE_CONFIGURED).model_copy(
        update={"verdict": Verdict.NOT_VERIFIABLE, "p_hallucinated": 1.0}
    )
    rescored = rescore(result)
    assert rescored.verdict == Verdict.NOT_VERIFIABLE
    assert rescored.reason == AbstentionReason.NO_EVIDENCE_CONFIGURED
