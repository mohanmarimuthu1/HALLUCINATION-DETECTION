"""Token-bucket rate limiting per client API key (Phase 5.2).

One bucket per key, refilled continuously at
`settings.rate_limit_refill_per_s` tokens/second up to
`settings.rate_limit_capacity`. In-memory and per-process, same scope
tradeoff as `settings.py` documents: correct for a single instance, not a
multi-instance deployment - that needs a shared store, out of scope here.

Chained after `auth.require_api_key` so an invalid key is always a 401,
never a 429 - rate limiting only applies once a key is known to be valid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException

from halludetect.api.auth import require_api_key
from halludetect.settings import Settings, get_settings


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, *, capacity: float, refill_per_s: float):
        self._capacity = capacity
        self._refill_per_s = refill_per_s
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._capacity, last_refill=now)
            self._buckets[key] = bucket
        else:
            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_per_s)
            bucket.last_refill = now

        if bucket.tokens < 1.0:
            return False
        bucket.tokens -= 1.0
        return True


# One shared limiter per process (same pattern as api.resolve's shared
# HealthTracker) - per-key bucket state must persist across requests, not
# reset on every call.
_limiter: RateLimiter | None = None


def _get_limiter(settings: Settings) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(capacity=settings.rate_limit_capacity, refill_per_s=settings.rate_limit_refill_per_s)
    return _limiter


def enforce_rate_limit(
    api_key: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency: validates the key (401) then checks its rate
    limit (429). Returns the validated key on success.
    """
    limiter = _get_limiter(settings)
    if not limiter.allow(api_key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    return api_key
