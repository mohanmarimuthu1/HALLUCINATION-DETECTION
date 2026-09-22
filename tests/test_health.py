"""HealthTracker tests - rolling success rate/latency and the circuit
breaker's cooldown-based demotion/re-enable (Phase 2.2, 2.5)."""
from halludetect.llm.health import HealthTracker


def test_success_rate_and_latency_tracked():
    tracker = HealthTracker()
    tracker.record_success("model-a", 100.0)
    tracker.record_success("model-a", 300.0)
    tracker.record_failure("model-a")

    health = tracker.get("model-a")
    assert health.attempts == 3
    assert health.successes == 2
    assert health.success_rate == 2 / 3
    assert health.avg_latency_ms == 200.0


def test_consecutive_failures_trigger_cooldown():
    tracker = HealthTracker(failure_threshold=3, cooldown_s=60)
    for _ in range(2):
        tracker.record_failure("model-a", now=0.0)
    assert tracker.is_available("model-a", now=0.0) is True

    tracker.record_failure("model-a", now=0.0)
    assert tracker.is_available("model-a", now=0.0) is False
    assert tracker.is_available("model-a", now=59.0) is False


def test_cooldown_expires_and_model_becomes_available_again():
    tracker = HealthTracker(failure_threshold=2, cooldown_s=60)
    tracker.record_failure("model-a", now=0.0)
    tracker.record_failure("model-a", now=0.0)
    assert tracker.is_available("model-a", now=60.0) is True


def test_success_resets_consecutive_failure_count():
    tracker = HealthTracker(failure_threshold=2, cooldown_s=60)
    tracker.record_failure("model-a", now=0.0)
    tracker.record_success("model-a", 50.0)
    tracker.record_failure("model-a", now=0.0)
    assert tracker.is_available("model-a", now=0.0) is True


def test_rate_limited_failure_records_timestamp():
    tracker = HealthTracker()
    tracker.record_failure("model-a", rate_limited=True, now=42.0)
    assert tracker.get("model-a").last_rate_limited_at == 42.0


def test_rank_available_prefers_higher_success_rate():
    tracker = HealthTracker()
    tracker.record_success("model-a", 100.0)
    tracker.record_failure("model-a")
    tracker.record_success("model-b", 100.0)
    tracker.record_success("model-b", 100.0)

    ranked = tracker.rank_available(["model-a", "model-b"])
    assert ranked == ["model-b", "model-a"]


def test_rank_available_excludes_models_in_cooldown():
    tracker = HealthTracker(failure_threshold=1, cooldown_s=60)
    tracker.record_failure("model-a", now=0.0)
    tracker.record_success("model-b", 100.0)

    ranked = tracker.rank_available(["model-a", "model-b"], now=0.0)
    assert ranked == ["model-b"]


def test_rank_available_gives_untried_models_a_chance():
    tracker = HealthTracker()
    tracker.record_success("model-a", 100.0)
    tracker.record_failure("model-a")

    ranked = tracker.rank_available(["model-a", "model-b"])
    assert ranked[0] == "model-b"
