"""Result cache protocol (Phase 6.1).

`api.main` checks this before running the detection pipeline and writes to
it after, keyed by `cache.key.compute_cache_key()`. A miss must return
`None`, never raise - a cache is an optimization, not a dependency the
request path can fail on.
"""
from typing import Protocol, runtime_checkable

from halludetect.detect.schemas import AnalysisResult


@runtime_checkable
class CacheStore(Protocol):
    def get(self, key: str) -> AnalysisResult | None: ...

    def set(self, key: str, value: AnalysisResult, *, ttl_s: float) -> None: ...
