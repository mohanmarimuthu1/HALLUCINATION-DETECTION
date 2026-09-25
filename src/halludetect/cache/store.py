"""File-backed CacheStore implementation (Phase 6.1), using `diskcache`.

Chosen over Redis for v1: no extra service to run or configure - a plain
directory on disk is enough for a single-instance deployment, and
`diskcache.Cache` already handles TTL expiry, concurrent access, and
process-restart persistence correctly rather than reimplementing any of
that here. A multi-instance deployment sharing one cache would need a
networked store (Redis) instead - same scope tradeoff already documented
for `HealthTracker`/`RateLimiter`, just for the one component in this
project that's actually meant to survive a restart.
"""
from __future__ import annotations

import diskcache

from halludetect.detect.schemas import AnalysisResult


class DiskCacheStore:
    def __init__(self, directory: str):
        self._cache = diskcache.Cache(directory)

    def get(self, key: str) -> AnalysisResult | None:
        raw = self._cache.get(key)
        if raw is None:
            return None
        return AnalysisResult.model_validate_json(raw)

    def set(self, key: str, value: AnalysisResult, *, ttl_s: float) -> None:
        self._cache.set(key, value.model_dump_json(), expire=ttl_s)
