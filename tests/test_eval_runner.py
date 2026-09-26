"""eval/runner.py tests (Phase 7.4) - replay-only path, fully offline: no
provider is ever constructed in these tests, matching the point of replay
mode (the runner must not touch `api.resolve`/network at all when
`record=False`).
"""
from pathlib import Path

import pytest

from halludetect.detect.schemas import AnalysisResult, ModelUsed, Timings, Verdict
from halludetect.eval.datasets import load_golden_set
from halludetect.eval.replay import EvalReplayStore
from halludetect.eval.runner import MissingReplayError, _cache_key_for, run_suite

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden" / "eval_set_a.yaml"


def _stub_result(request_id: str) -> AnalysisResult:
    return AnalysisResult(
        request_id=request_id,
        verdict=Verdict.GROUNDED,
        p_hallucinated=0.0,
        groundedness=1.0,
        groundedness_ci=(0.9, 1.0),
        claims=[],
        n_verifiable_claims=3,
        model_used=ModelUsed(provider="openrouter", model="free/a"),
        cost_usd=0.0,
        timings_ms=Timings(total=10, retrieval=1, extraction=4, verification=5),
        calibration_version="heuristic-v0",
    )


def _prefilled_replay_path(tmp_path) -> Path:
    replay_path = tmp_path / "replay.json"
    store = EvalReplayStore(replay_path)
    for item in load_golden_set(_GOLDEN_PATH):
        store.set(_cache_key_for(item), _stub_result(item.id))
    store.save()
    return replay_path


def test_replay_mode_never_touches_a_provider(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("resolve_provider must not be called in replay mode")

    monkeypatch.setattr("halludetect.eval.runner.resolve_provider", _boom)

    replay_path = _prefilled_replay_path(tmp_path)
    outcomes = run_suite(_GOLDEN_PATH, replay_path, record=False)

    items = load_golden_set(_GOLDEN_PATH)
    assert len(outcomes) == len(items)
    assert {o.item.id for o in outcomes} == {i.id for i in items}


def test_replay_mode_raises_on_missing_entry(tmp_path):
    empty_replay_path = tmp_path / "empty.json"
    with pytest.raises(MissingReplayError, match="golden-a-001"):
        run_suite(_GOLDEN_PATH, empty_replay_path, record=False)


def test_replay_mode_returns_the_recorded_result(tmp_path):
    replay_path = _prefilled_replay_path(tmp_path)
    outcomes = run_suite(_GOLDEN_PATH, replay_path, record=False)
    first = next(o for o in outcomes if o.item.id == "golden-a-001")
    assert first.result.request_id == "golden-a-001"
    assert first.result.verdict == Verdict.GROUNDED
