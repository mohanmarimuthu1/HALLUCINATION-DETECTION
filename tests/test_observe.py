"""observe.ModelObserver: per-model counters behind /v1/models/stats."""
from halludetect.detect.schemas import AnalysisResult, ModelUsed, Timings, Verdict
from halludetect.observe import ModelObserver, Role


def _result(verdict: Verdict, groundedness: float) -> AnalysisResult:
    return AnalysisResult(
        request_id="r",
        verdict=verdict,
        p_hallucinated=0.5,
        groundedness=groundedness,
        groundedness_ci=(0.0, 1.0),
        claims=[],
        n_verifiable_claims=3,
        model_used=ModelUsed(provider="p", model="m"),
        cost_usd=0.0,
        timings_ms=Timings(total=0, retrieval=0, extraction=0, verification=0),
        calibration_version="test",
    )


def test_rates_count_only_scored_verdicts():
    obs = ModelObserver()
    for verdict, g in [
        (Verdict.GROUNDED, 1.0),
        (Verdict.CONTRADICTED, 0.0),
        (Verdict.NOT_ENOUGH_INFO, 0.5),
        (Verdict.GROUNDED, 0.9),
        (Verdict.NOT_VERIFIABLE, 0.0),
    ]:
        obs.record_answer_verdict("p", "m", _result(verdict, g))
    [row] = obs.snapshot()["models"]
    assert row["answer"]["verdicts"] == {
        "GROUNDED": 2,
        "CONTRADICTED": 1,
        "NOT_ENOUGH_INFO": 1,
        "NOT_VERIFIABLE": 1,
    }
    assert row["answer"]["contradicted_rate"] == 0.25
    assert row["answer"]["unsupported_rate"] == 0.5
    assert row["answer"]["mean_groundedness"] == 0.6


def test_no_scored_answers_means_no_rates():
    obs = ModelObserver()
    obs.record_answer_verdict("p", "m", _result(Verdict.NOT_VERIFIABLE, 0.0))
    [row] = obs.snapshot()["models"]
    assert row["answer"]["unsupported_rate"] is None
    assert row["answer"]["mean_groundedness"] is None


def test_roles_are_counted_separately_and_failures_are_calls():
    obs = ModelObserver()
    for latency in range(1, 21):
        obs.record_success("p", "m", Role.VERIFY, float(latency * 100))
    obs.record_failure("p", "m", Role.ANSWER, "rate limited", rate_limited=True)
    [row] = obs.snapshot(in_cooldown={("p", "m"): True})["models"]
    assert row["answer"] == {
        **row["answer"],
        "calls": 1,
        "failures": 1,
        "rate_limited": 1,
        "avg_latency_ms": None,
    }
    assert row["verify"] == {
        **row["verify"],
        "calls": 20,
        "failures": 0,
        "rate_limited": 0,
        "avg_latency_ms": 1050.0,
        "p95_latency_ms": 2000.0,
        "unsupported_rate": None,
    }
    assert row["in_cooldown"] is True
    assert row["last_error"] == "rate limited"
    assert row["last_used"].endswith("Z")


def test_reset_clears_models():
    obs = ModelObserver()
    obs.record_success("p", "m", Role.ANSWER, 1.0)
    obs.reset()
    assert obs.snapshot()["models"] == []


def test_checker_verdicts_are_kept_apart_from_answer_verdicts():
    obs = ModelObserver()
    obs.record_check_verdict("p", "checker", _result(Verdict.NOT_ENOUGH_INFO, 0.0))
    obs.record_check_verdict("p", "checker", _result(Verdict.GROUNDED, 1.0))
    [row] = obs.snapshot()["models"]
    assert row["verify"]["unsupported_rate"] == 0.5
    assert row["verify"]["verdicts"]["NOT_ENOUGH_INFO"] == 1
    assert row["answer"]["unsupported_rate"] is None
