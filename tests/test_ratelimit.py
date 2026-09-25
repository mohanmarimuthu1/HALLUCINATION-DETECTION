"""Token-bucket rate limiter unit tests (Phase 5.2) - offline, deterministic
clock passed explicitly rather than relying on real elapsed time."""
from halludetect.api.ratelimit import RateLimiter


def test_allows_up_to_capacity_then_blocks():
    limiter = RateLimiter(capacity=3, refill_per_s=0.0)
    assert limiter.allow("key", now=0.0) is True
    assert limiter.allow("key", now=0.0) is True
    assert limiter.allow("key", now=0.0) is True
    assert limiter.allow("key", now=0.0) is False


def test_refills_over_time():
    limiter = RateLimiter(capacity=1, refill_per_s=1.0)
    assert limiter.allow("key", now=0.0) is True
    assert limiter.allow("key", now=0.1) is False
    assert limiter.allow("key", now=1.0) is True


def test_refill_never_exceeds_capacity():
    limiter = RateLimiter(capacity=2, refill_per_s=100.0)
    assert limiter.allow("key", now=0.0) is True
    assert limiter.allow("key", now=1000.0) is True
    assert limiter.allow("key", now=1000.0) is True
    assert limiter.allow("key", now=1000.0) is False


def test_keys_are_tracked_independently():
    limiter = RateLimiter(capacity=1, refill_per_s=0.0)
    assert limiter.allow("key-a", now=0.0) is True
    assert limiter.allow("key-b", now=0.0) is True
    assert limiter.allow("key-a", now=0.0) is False
    assert limiter.allow("key-b", now=0.0) is False
