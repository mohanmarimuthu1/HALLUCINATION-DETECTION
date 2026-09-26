"""Golden-set eval orchestration (Phase 7.4).

Two modes:

- **Replay** (default): every golden item's cache key must already be in
  the committed `EvalReplayStore` file. No provider is resolved, no
  network call is made, no API key is required - this is what makes a CI
  run genuinely offline, not just configured to look offline.
- **Record** (`--record`): resolves one real OpenRouter provider via
  `api.resolve.resolve_provider`, lazily on the first item that actually
  needs a live call, and reuses that same provider for every other item
  that needs recording in the same run - determinism within one recording,
  the same way `detect.pipeline.run()` already keeps `model_used`
  attributable to one model per request rather than mixing models across a
  request's own extraction/verification calls.
  By default, an item already present in the replay file is reused as-is
  rather than re-called - `--record` is meant to be resumable, not just a
  one-shot batch job: recording a 90-item set live against the free tier
  in one sitting reliably hits transient rate limits/blocks partway
  through (discovered live, repeatedly), and re-paying for every
  already-good item on every retry made recovering from that
  disproportionately expensive. Pass `force_record=True` (CLI:
  `--force-record`) to fully refresh every item regardless of what's
  already recorded. The replay file is saved after every newly recorded
  item, not just once at the end, so a mid-batch failure never discards
  progress already made.

A cache-key miss in replay mode is a hard error, never a silent live
fallback - the project's non-negotiable "no evidence -> never guess" rule,
applied here to eval determinism instead of evidence.
"""
from __future__ import annotations

import time
from collections.abc import Callable
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
_MAX_RECORD_ATTEMPTS_PER_ITEM = 8

# A pause between items, record mode only - discovered live, recording a
# 90-item batch back-to-back tripped OpenRouter's free-tier per-minute rate
# limit outright (an LLMRateLimitError that survived RetryingProvider's own
# backoff *and* this module's model-reselection retries). Switching models
# doesn't help a rate limit that's scoped to the account, not one model -
# only slowing down does.
_RECORD_ITEM_DELAY_S = 2.0

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


def _resolve_recording_provider_with_backoff(
    settings: Settings, *, sleep: Callable[[float], None], attempts: int = 3
) -> tuple[LLMProvider, ModelUsed]:
    """Resolution itself (the free-model catalog fetch) can hit the same
    transient network failures a `.complete()` call can - discovered live
    when a bare SSL/transport error during catalog refetch propagated
    straight out of the per-item retry loop below and crashed the whole
    run, since that reselection step wasn't wrapped in its own retry.
    """
    last_error: LLMError | None = None
    for attempt in range(attempts):
        try:
            return _resolve_recording_provider(settings)
        except LLMError as exc:
            last_error = exc
            if attempt < attempts - 1:
                sleep(_RECORD_ITEM_DELAY_S)
    assert last_error is not None
    raise last_error


def run_suite(
    golden_path: str | Path,
    replay_path: str | Path,
    *,
    record: bool,
    force_record: bool = False,
    settings: Settings | None = None,
    # Injectable so tests never actually sleep, matching llm/retry.py's
    # RetryingProvider convention - a fully-mocked test recording 90 items
    # should not take three real minutes just because record mode paces
    # itself against a real free-tier API.
    sleep: Callable[[float], None] = time.sleep,
) -> list[EvalOutcome]:
    settings = settings or get_settings()
    items = load_golden_set(golden_path)
    store = EvalReplayStore(replay_path)

    provider: LLMProvider | None = None
    model_used: ModelUsed | None = None

    outcomes = []
    for item in items:
        cache_key = _cache_key_for(item)
        already_recorded = store.get(cache_key)

        if record and (force_record or already_recorded is None):
            # Resolved lazily, on the first item that actually needs a live
            # call - not once at the top of the run - so a --record
            # invocation where every item is already recorded never
            # touches the network at all.
            if provider is None:
                provider, model_used = _resolve_recording_provider_with_backoff(settings, sleep=sleep)
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
                    sleep(_RECORD_ITEM_DELAY_S)
                    try:
                        provider, model_used = _resolve_recording_provider_with_backoff(settings, sleep=sleep)
                    except LLMError:
                        pass  # transient resolution hiccup - retry the next attempt with the same provider
            if result is None:
                assert last_error is not None
                raise last_error
            store.set(cache_key, result)
            # Saved after every item, not just once at the end - a batch
            # this size can run for minutes against a real free-tier API
            # and *will* occasionally hit a wall (rate limiting, a bad
            # model) partway through. Combined with skipping
            # already-recorded items by default (see this module's
            # docstring), a failed run can simply be re-invoked afterward:
            # it picks up only the items still missing, rather than
            # re-paying for everything already on disk.
            store.save()
            if _RECORD_ITEM_DELAY_S:
                sleep(_RECORD_ITEM_DELAY_S)
        else:
            if already_recorded is None:
                raise MissingReplayError(
                    f"golden item {item.id!r} has no recorded replay entry "
                    f"(key {cache_key}) and --record was not passed; run "
                    "`python -m halludetect.eval --suite golden --record` to populate it"
                )
            result = already_recorded

        outcomes.append(EvalOutcome(item=item, result=result))

    return outcomes
