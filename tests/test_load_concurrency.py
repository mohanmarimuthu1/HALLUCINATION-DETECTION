"""Concurrency/load tests for the rate limiter and free-model rotation
(Phase 8.2).

Both pieces of shared state these cover - `RateLimiter`'s per-key buckets
and `HealthTracker`'s per-model counters - are process-global singletons
(`ratelimit._limiter`, `api.resolve._openrouter_health`) touched by every
request. FastAPI runs a *sync* dependency like `enforce_rate_limit` in its
threadpool, so "concurrent" here is real OS threads mutating one dict, not
cooperative async that only yields at awaits.

`sys.setswitchinterval` is lowered for the duration of each test so the
interpreter preempts threads mid-function instead of usually letting a
short critical section run to completion. Without that these tests pass on
racy code most of the time, which is the worst possible outcome for a
regression test: the bug stays in, and the suite says it doesn't.

Correctness under load only - no throughput or latency assertions, which
would be flaky on shared CI hardware. Measured throughput numbers live in
process.md instead.
"""
from __future__ import annotations

import sys
import threading

import pytest
from fastapi.testclient import TestClient

from halludetect.api import main, ratelimit
from halludetect.api.ratelimit import RateLimiter
from halludetect.api.resolve import _openrouter_health
from halludetect.llm import openrouter
from halludetect.llm.base import LLMResponse, TokenUsage
from halludetect.llm.exceptions import LLMResponseError
from halludetect.llm.health import HealthTracker
from halludetect.llm.openrouter import FreeModelCatalog
from halludetect.llm.router import Router
from halludetect.settings import Settings, get_settings

_VALID_KEY = "load-test-key"
_AUTH_HEADERS = {"Authorization": f"Bearer {_VALID_KEY}"}


@pytest.fixture(autouse=True)
def _preempt_aggressively():
    """Force frequent thread switches so an unlocked read-modify-write is
    actually interleaved rather than accidentally atomic.
    """
    original = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(original)


def _run_threads(target, count: int) -> None:
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_rate_limiter_never_admits_more_than_capacity_under_threads():
    """A token bucket that over-admits under a burst is worse than no
    limiter: the whole point is the hard ceiling, and a caller who can
    beat it with concurrency has no ceiling at all.
    """
    capacity = 100
    limiter = RateLimiter(capacity=capacity, refill_per_s=0.0)
    counts: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        allowed = 0
        for _ in range(200):
            # Fixed clock: any admission beyond `capacity` is a lost
            # update, never refill.
            if limiter.allow("one-key", now=0.0):
                allowed += 1
        with lock:
            counts.append(allowed)

    _run_threads(worker, 8)

    assert sum(counts) == capacity


def test_rate_limiter_keeps_keys_independent_under_threads():
    capacity = 20
    keys = [f"key-{i}" for i in range(8)]
    limiter = RateLimiter(capacity=capacity, refill_per_s=0.0)
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def worker_for(key: str):
        def worker() -> None:
            allowed = 0
            for _ in range(100):
                if limiter.allow(key, now=0.0):
                    allowed += 1
            with lock:
                counts[key] = counts.get(key, 0) + allowed

        return worker

    threads = [threading.Thread(target=worker_for(key)) for key in keys for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert counts == dict.fromkeys(keys, capacity)


def test_health_tracker_loses_no_updates_under_threads():
    """`rank_available` sorts on success_rate, and the circuit breaker
    trips on a consecutive-failure count. Both read counters that every
    request thread increments, so a lost update silently distorts routing
    and delays demotion.
    """
    tracker = HealthTracker(failure_threshold=1_000_000, cooldown_s=0.0)
    per_thread = 300
    threads = 8

    def worker() -> None:
        for _ in range(per_thread):
            tracker.record_failure("free/a")
            tracker.record_success("free/b", latency_ms=1.0)

    _run_threads(worker, threads)

    total = threads * per_thread
    assert tracker.get("free/a").attempts == total
    assert tracker.get("free/a").consecutive_failures == total
    assert tracker.get("free/b").attempts == total
    assert tracker.get("free/b").successes == total


def test_health_tracker_demotes_exactly_once_under_threads():
    """Concurrent failures past the threshold must still leave the model
    in cooldown - a torn write to `cooldown_until` would leave a model the
    breaker believes it demoted still being handed out.
    """
    tracker = HealthTracker(failure_threshold=3, cooldown_s=300.0)

    def worker() -> None:
        for _ in range(50):
            tracker.record_failure("free/bad", now=0.0)

    _run_threads(worker, 8)

    assert tracker.is_available("free/bad", now=0.0) is False
    assert tracker.rank_available(["free/bad", "free/ok"], now=0.0) == ["free/ok"]


def test_router_rotation_is_thread_safe_under_load(monkeypatch):
    """Free-model rotation under concurrent load: every caller must get a
    real response from a healthy model while a broken one is being
    demoted underneath them, with no exception escaping and no lost
    health accounting.
    """
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/broken", "free/good"])

    class _ScriptedProvider:
        def __init__(self, model: str):
            self._model = model

        def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
            if self._model == "free/broken":
                raise LLMResponseError("free/broken: no content in response")
            return LLMResponse(
                text="ok",
                provider="openrouter",
                model=self._model,
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            )

        def supports_json_schema(self) -> bool:
            return False

    router = Router(
        catalog=FreeModelCatalog(api_key="key", ttl_s=3600.0),
        provider_factory=_ScriptedProvider,
        health=HealthTracker(failure_threshold=1_000_000, cooldown_s=0.0),
    )

    per_thread = 40
    threads = 8
    results: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        local_ok, local_err = [], []
        for _ in range(per_thread):
            try:
                local_ok.append(router.complete("prompt").model)
            except Exception as exc:  # noqa: BLE001 - any escape is the failure
                local_err.append(repr(exc))
        with lock:
            results.extend(local_ok)
            errors.extend(local_err)

    _run_threads(worker, threads)

    total = threads * per_thread
    assert errors == []
    assert results == ["free/good"] * total
    assert router.health.get("free/good").successes == total
    # The broken model is not tried `total` times: one failure drops its
    # success rate below the proven-good model's, and `rank_available`
    # stops offering it first. How many threads hit it before that
    # propagates is scheduling-dependent, so the assertion is the
    # invariant (it was tried, and it lost) rather than an exact count.
    broken = router.health.get("free/broken")
    assert broken.attempts >= 1
    assert broken.successes == 0
    assert router.health.rank_available(["free/broken", "free/good"]) == ["free/good", "free/broken"]


def test_api_enforces_capacity_exactly_under_concurrent_requests(tmp_path, monkeypatch):
    """End-to-end at the boundary a real caller hits: fire far more
    concurrent requests than the bucket allows and assert the 200/429
    split is exactly the capacity, not merely "roughly".
    """
    capacity = 25
    settings = Settings(
        _env_file=None,
        openrouter_api_key="key",
        client_api_keys=_VALID_KEY,
        cache_dir=str(tmp_path / "cache"),
        rate_limit_capacity=capacity,
        rate_limit_refill_per_s=0.0,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    main.app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(openrouter, "fetch_free_models", lambda api_key: ["free/a"])
    main.app.state.custom_evidence_source = None
    _openrouter_health._health.clear()
    ratelimit._limiter = None
    main._cache_store = None

    client = TestClient(main.app)
    # evidence_source "none" with no evidence short-circuits to
    # NOT_VERIFIABLE before any LLM call, so this exercises auth +
    # rate limiting without needing a provider scripted per request.
    body = {"answer": "irrelevant", "evidence_source": "none"}
    statuses: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        local = []
        for _ in range(10):
            local.append(client.post("/v1/verify", json=body, headers=_AUTH_HEADERS).status_code)
        with lock:
            statuses.extend(local)

    try:
        _run_threads(worker, 10)
    finally:
        main.app.dependency_overrides.clear()

    assert set(statuses) <= {200, 429}
    assert statuses.count(200) == capacity
    assert statuses.count(429) == len(statuses) - capacity


def test_catalog_refetches_once_when_threads_arrive_on_a_cold_cache(monkeypatch):
    """The catalog is shared by every concurrent request. A cold or expired
    cache must cost one upstream call, not one per thread in flight.
    """
    calls: list[str | None] = []
    lock = threading.Lock()

    def _fetch(api_key):
        with lock:
            calls.append(api_key)
        return ["free/a", "free/b"]

    monkeypatch.setattr(openrouter, "fetch_free_models", _fetch)
    catalog = FreeModelCatalog(api_key="key", ttl_s=3600.0)

    seen: list[list[str]] = []

    def worker() -> None:
        local = [catalog.get_models() for _ in range(20)]
        with lock:
            seen.extend(local)

    _run_threads(worker, 8)

    assert len(calls) == 1
    assert all(models == ["free/a", "free/b"] for models in seen)
