"""Golden-set eval orchestration (Phase 7.4).

Two modes:

- **Replay** (default): every golden item's cache key must already be in
  the committed `EvalReplayStore` file. No provider is resolved, no
  network call is made, no API key is required - this is what makes a CI
  run genuinely offline, not just configured to look offline.
- **Record** (`--record`): resolves one real OpenRouter provider *once* for
  the whole run (not per item) via `api.resolve.resolve_provider`, so every
  item in a single recording pass is answered by the same actual model -
  determinism within one recording, the same way `detect.pipeline.run()`
  already keeps `model_used` attributable to one model per request rather
  than mixing models across a request's own extraction/verification calls.
  Every item is re-run for real in this mode (not just cache misses), so a
  `--record` pass fully refreshes the file instead of silently reusing
  stale entries next to freshly recorded ones.

A cache-key miss in replay mode is a hard error, never a silent live
fallback - the project's non-negotiable "no evidence -> never guess" rule,
applied here to eval determinism instead of evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from halludetect.api import resolve as api_resolve
from halludetect.api.resolve import resolve_provider
from halludetect.api.schemas import ModelPrefsIn, ModelProvider
from halludetect.cache.key import compute_cache_key
from halludetect.detect import pipeline
from halludetect.detect.schemas import AnalysisResult, ModelUsed
from halludetect.eval.datasets import GoldenItem, load_golden_set
from halludetect.eval.replay import EvalReplayStore
from halludetect.evidence.direct import DirectEvidence
from halludetect.llm.base import LLMProvider
from halludetect.llm.exceptions import LLMError
from halludetect.settings import Settings, get_settings

# Router.pick_model() (used by resolve_provider for provider: openrouter)
# deliberately records no health signal itself - only Router.complete()'s
# per-call failover does that (see router.py's pick_model docstring). A
# fresh CLI process's HealthTracker starts empty every run, so a model
# that fails mid-item would otherwise get picked again forever. Recording
# is the one caller in this codebase that needs its own bounded retry
# across models: production /v1/verify requests deliberately resolve once
# and let a mid-request failure surface as a 502 (Phase 5.1), which is
# correct for a live request but useless for building an offline replay
# fixture that must eventually get a real answer for every item.
_MAX_RECORD_ATTEMPTS_PER_ITEM = 5

# Golden set A always supplies direct evidence, which docs/contract.md's
# own precedence rule makes authoritative regardless of `evidence_source` -
# "none" is the neutral value a real caller would send alongside direct
# evidence, so cache keys here match what a real /v1/verify request would
# hash to.
_EVIDENCE_SOURCE_FOR_KEY = "none"


class MissingReplayError(Exception):
    """A golden item's cache key wasn't found in the committed replay file
    and `--record` wasn't passed. Raised instead of falling back to a live
    call - an eval run that's supposed to be offline must not silently
    become a live one.
    """


@dataclass(frozen=True)
class EvalOutcome:
    item: GoldenItem
    result: AnalysisResult


def _cache_key_for(item: GoldenItem) -> str:
    return compute_cache_key(
        answer=item.answer,
        question=item.question,
        evidence=item.evidence,
        evidence_source=_EVIDENCE_SOURCE_FOR_KEY,
        model_provider=ModelProvider.OPENROUTER.value,
        allow_free_pool=True,
        pinned_model=None,
        user_api_key=None,
    )


def _resolve_recording_provider(settings: Settings) -> tuple[LLMProvider, ModelUsed]:
    model_prefs = ModelPrefsIn(provider=ModelProvider.OPENROUTER, allow_free_pool=True)
    return resolve_provider(model_prefs, settings)


def run_suite(
    golden_path: str | Path,
    replay_path: str | Path,
    *,
    record: bool,
    settings: Settings | None = None,
) -> list[EvalOutcome]:
    settings = settings or get_settings()
    items = load_golden_set(golden_path)
    store = EvalReplayStore(replay_path)

    provider: LLMProvider | None = None
    model_used: ModelUsed | None = None
    if record:
        provider, model_used = _resolve_recording_provider(settings)

    outcomes = []
    for item in items:
        cache_key = _cache_key_for(item)

        if record:
            result = None
            last_error: LLMError | None = None
            for _ in range(_MAX_RECORD_ATTEMPTS_PER_ITEM):
                assert provider is not None and model_used is not None
                try:
                    result = pipeline.run(
                        answer=item.answer,
                        question=item.question,
                        evidence_source=DirectEvidence(item.evidence),
                        provider=provider,
                        request_id=item.id,
                        model_used=model_used,
                    )
                    break
                except LLMError as exc:
                    last_error = exc
                    api_resolve._openrouter_health.record_failure(model_used.model)
                    provider, model_used = _resolve_recording_provider(settings)
            if result is None:
                assert last_error is not None
                raise last_error
            store.set(cache_key, result)
        else:
            cached = store.get(cache_key)
            if cached is None:
                raise MissingReplayError(
                    f"golden item {item.id!r} has no recorded replay entry "
                    f"(key {cache_key}) and --record was not passed; run "
                    "`python -m halludetect.eval --suite golden --record` to populate it"
                )
            result = cached

        outcomes.append(EvalOutcome(item=item, result=result))

    if record:
        store.save()

    return outcomes
