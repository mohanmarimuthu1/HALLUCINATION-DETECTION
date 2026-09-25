"""DiskCacheStore tests (Phase 6.1) - offline, real local disk under
pytest's tmp_path (no network, no live keys), one isolated directory per
test so nothing leaks between tests or across test runs.
"""
import time

from halludetect.cache.store import DiskCacheStore
from halludetect.detect.schemas import AnalysisResult, ModelUsed, Timings, Verdict

_RESULT = AnalysisResult(
    request_id="req-1",
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
    store = DiskCacheStore(str(tmp_path))
    assert store.get("missing-key") is None


def test_set_then_get_roundtrips(tmp_path):
    store = DiskCacheStore(str(tmp_path))
    store.set("key-1", _RESULT, ttl_s=60)
    result = store.get("key-1")
    assert result == _RESULT


def test_different_keys_do_not_collide(tmp_path):
    store = DiskCacheStore(str(tmp_path))
    other = _RESULT.model_copy(update={"request_id": "req-2"})
    store.set("key-1", _RESULT, ttl_s=60)
    store.set("key-2", other, ttl_s=60)
    assert store.get("key-1").request_id == "req-1"
    assert store.get("key-2").request_id == "req-2"


def test_entry_expires_after_ttl(tmp_path):
    store = DiskCacheStore(str(tmp_path))
    store.set("key-1", _RESULT, ttl_s=0.05)
    assert store.get("key-1") == _RESULT
    time.sleep(0.2)
    assert store.get("key-1") is None


def test_persists_across_separate_store_instances(tmp_path):
    """The whole point of a file-backed cache over an in-memory one: it
    survives a process restart, not just repeated calls on one instance.
    """
    DiskCacheStore(str(tmp_path)).set("key-1", _RESULT, ttl_s=60)
    reopened = DiskCacheStore(str(tmp_path))
    assert reopened.get("key-1") == _RESULT
