"""EvalReplayStore tests (Phase 7.4) - offline, real local disk under
pytest's tmp_path, mirroring tests/test_cache_store.py's pattern for
Phase 6.1's DiskCacheStore.
"""
from halludetect.detect.schemas import AnalysisResult, ModelUsed, Timings, Verdict
from halludetect.eval.replay import EvalReplayStore

_RESULT = AnalysisResult(
    request_id="golden-a-001",
    verdict=Verdict.GROUNDED,
    p_hallucinated=0.0,
    groundedness=1.0,
    groundedness_ci=(0.9, 1.0),
    claims=[],
    n_verifiable_claims=3,
    model_used=ModelUsed(provider="openrouter", model="free/a"),
    cost_usd=0.0,
    timings_ms=Timings(total=100, retrieval=10, extraction=40, verification=50),
    calibration_version="heuristic-v0",
)


def test_miss_returns_none(tmp_path):
    store = EvalReplayStore(tmp_path / "replay.json")
    assert store.get("missing-key") is None


def test_set_then_get_roundtrips_without_save(tmp_path):
    store = EvalReplayStore(tmp_path / "replay.json")
    store.set("key-1", _RESULT)
    assert store.get("key-1") == _RESULT


def test_different_keys_do_not_collide(tmp_path):
    store = EvalReplayStore(tmp_path / "replay.json")
    other = _RESULT.model_copy(update={"request_id": "golden-a-002"})
    store.set("key-1", _RESULT)
    store.set("key-2", other)
    assert store.get("key-1").request_id == "golden-a-001"
    assert store.get("key-2").request_id == "golden-a-002"


def test_save_then_reload_from_a_fresh_instance(tmp_path):
    path = tmp_path / "replay.json"
    first = EvalReplayStore(path)
    first.set("key-1", _RESULT)
    first.save()

    reopened = EvalReplayStore(path)
    assert reopened.get("key-1") == _RESULT


def test_saved_file_is_plain_diffable_json(tmp_path):
    path = tmp_path / "replay.json"
    store = EvalReplayStore(path)
    store.set("key-1", _RESULT)
    store.save()

    text = path.read_text(encoding="utf-8")
    assert text.startswith("{")
    assert '"key-1"' in text
    assert '"request_id": "golden-a-001"' in text


def test_loading_a_nonexistent_file_starts_empty(tmp_path):
    store = EvalReplayStore(tmp_path / "does-not-exist.json")
    assert store.get("anything") is None
