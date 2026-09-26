"""Content-addressed replay store for golden-set eval (Phase 7.4).

Same `get`/`set` shape as `cache.base.CacheStore` (Phase 6.1), but backed by
one committed, human-readable JSON file instead of `diskcache` - CLAUDE.md
forbids committing generated binary artifacts, and a golden-set recording
needs to be diffable in a PR, not opaque. Reuses `cache.key.compute_cache_key`
unchanged, so eval replay and the live result cache share one hashing
scheme instead of eval inventing a second one.

Entries never expire (there is no `ttl_s` here) - a committed recording is
meant to be replayed indefinitely until someone re-records it with
`--record`, not aged out on a clock.
"""
from __future__ import annotations

import json
from pathlib import Path

from halludetect.detect.schemas import AnalysisResult


class EvalReplayStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._entries: dict[str, dict] = {}
        if self._path.exists():
            self._entries = json.loads(self._path.read_text(encoding="utf-8"))

    def get(self, key: str) -> AnalysisResult | None:
        raw = self._entries.get(key)
        if raw is None:
            return None
        return AnalysisResult.model_validate(raw)

    def set(self, key: str, value: AnalysisResult) -> None:
        self._entries[key] = value.model_dump(mode="json")

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self._entries, indent=2, sort_keys=True) + "\n"
        self._path.write_text(serialized, encoding="utf-8")
