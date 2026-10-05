"""FastAPI app: `POST /v1/verify`, `GET /healthz` (Phase 5.1), per-key auth
and token-bucket rate limiting (Phase 5.2), a result cache in front of the
pipeline (Phase 6.1), `POST /v1/chat` (a model answers, the pipeline checks
the answer) and `GET /v1/models/stats` (contract v1.3).

This is `detect.pipeline.run()`'s first real external caller. This module
does request parsing, auth/rate-limit enforcement, a cache lookup,
resolution (`api.resolve`), and response serialization against
`docs/contract.md`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.resources import files
from time import monotonic
from typing import TypeVar

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from halludetect.api.auth import require_api_key
from halludetect.api.ratelimit import enforce_rate_limit
from halludetect.api.resolve import (
    Candidate,
    ResolutionError,
    pool_health,
    resolve_candidates,
    resolve_evidence_source,
)
from halludetect.api.schemas import ChatRequestIn, ChatResult, VerifyRequestIn
from halludetect.cache.base import CacheStore
from halludetect.cache.key import compute_cache_key
from halludetect.cache.store import DiskCacheStore
from halludetect.detect import pipeline
from halludetect.detect.answer import generate_answer
from halludetect.detect.fuse import rescore
from halludetect.detect.schemas import AbstentionReason, AnalysisResult, ModelUsed, Timings, Verdict
from halludetect.evidence.base import EvidenceSource
from halludetect.llm.base import LLMProvider, LLMResponse
from halludetect.llm.exceptions import LLMAuthError, LLMError, LLMRateLimitError
from halludetect.logging import bind_request_id, configure_logging, get_logger
from halludetect.observe import Role, observer
from halludetect.settings import Settings, get_settings

T = TypeVar("T")

configure_logging()
_logger = get_logger(__name__)

app = FastAPI(title="HALLUDETECT API", version="1.0.0")

# Registration point for an embedding deployment's own retriever
# (docs/contract.md: evidence_source == "custom", Phase 3.4's plug-in hook).
# A Python callable can't be expressed in a JSON request body, so "custom"
# is only usable when a deployment sets this at startup; otherwise a request
# for it is a 400, never a silent fall-through to NoEvidenceSource.
# Type: EvidenceSource | None (mypy doesn't allow an inline annotation on a
# non-self attribute assignment, and app.state is an untyped attribute bag).
app.state.custom_evidence_source = None

# One shared cache store per process (same pattern as api.resolve's shared
# HealthTracker) - constructed lazily against settings.cache_dir on first
# use, not per request.
_cache_store: CacheStore | None = None
# Set once opening the cache has failed, so later requests don't retry it.
_cache_unavailable = False


def _get_cache_store(settings: Settings) -> CacheStore | None:
    """`None` if the cache directory can't be opened. The cache is an
    optimisation, so an unwritable directory runs the service uncached
    rather than failing every request - which is what happened on Vercel,
    whose filesystem is read-only outside /tmp (point CACHE_DIR there to
    keep caching).
    """
    global _cache_store, _cache_unavailable
    if _cache_store is None and not _cache_unavailable:
        try:
            _cache_store = DiskCacheStore(settings.cache_dir)
        except OSError as exc:
            _cache_unavailable = True
            _logger.warning("cache_unavailable", cache_dir=settings.cache_dir, error=str(exc))
    return _cache_store


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """The browser client (halludetect/web/index.html). It holds no key:
    visitors enter their own, which the page sends as a Bearer token.
    """
    page = files("halludetect.web").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(page, headers={"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


class _CallCounter:
    """Counts calls, so health is only updated for a model that was asked
    something (a no-evidence request returns without calling it).
    """

    def __init__(self, provider: LLMProvider):
        self._provider = provider
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        return self._provider.complete(prompt, max_tokens=max_tokens)

    def supports_json_schema(self) -> bool:
        return self._provider.supports_json_schema()


def _run_with_failover(
    candidates: Iterator[Candidate],
    attempt: Callable[[LLMProvider, ModelUsed], T],
    settings: Settings,
    role: Role,
    *,
    started: float | None = None,
) -> tuple[T, ModelUsed]:
    """Calls `attempt` on each candidate model in turn until one succeeds,
    reporting every outcome to that model's pool health (so later requests
    rank it accordingly) and to the model observer. One result always
    comes from one model. A rejected key (401) skips the rest of that
    provider's models, since they share the key.

    `started` lets several loops in one request share one
    `free_pool_budget_s`.
    """
    errors: list[str] = []
    rejected_providers: set[str] = set()
    attempts = 0
    started = monotonic() if started is None else started
    try:
        for candidate in candidates:
            model_used = candidate.model_used
            if model_used.provider in rejected_providers:
                continue
            if attempts >= settings.free_pool_max_attempts:
                break
            if attempts and monotonic() - started >= settings.free_pool_budget_s:
                errors.append(f"stopped after {settings.free_pool_budget_s:.0f}s")
                break
            attempts += 1
            start = monotonic()
            counted = _CallCounter(candidate.provider)
            try:
                result = attempt(counted, model_used)
            except LLMAuthError as exc:
                rejected_providers.add(model_used.provider)
                observer.record_failure(model_used.provider, model_used.model, role, str(exc))
                errors.append(f"{model_used.provider}/{model_used.model}: {exc}")
            except LLMError as exc:
                rate_limited = isinstance(exc, LLMRateLimitError)
                if candidate.health is not None:
                    candidate.health.record_failure(model_used.model, rate_limited=rate_limited)
                observer.record_failure(
                    model_used.provider, model_used.model, role, str(exc), rate_limited=rate_limited
                )
                errors.append(f"{model_used.provider}/{model_used.model}: {exc}")
                _logger.warning(
                    "llm_attempt_failed",
                    role=role.value,
                    model=model_used.model,
                    provider=model_used.provider,
                    error=str(exc),
                )
            else:
                if counted.calls:
                    latency_ms = (monotonic() - start) * 1000
                    if candidate.health is not None:
                        candidate.health.record_success(model_used.model, latency_ms)
                    observer.record_success(model_used.provider, model_used.model, role, latency_ms)
                return result, model_used
    except LLMError as exc:
        # resolve_provider failed for a pinned or named provider.
        raise HTTPException(status_code=503, detail=f"no LLM provider available: {exc}") from exc

    if attempts == 0 and not errors:
        raise HTTPException(status_code=503, detail="no LLM provider available: no free-pool model is configured")
    raise HTTPException(status_code=502, detail="LLM provider failed: " + "; ".join(errors))


def _verify_attempt(
    *, answer: str, question: str | None, evidence_source: EvidenceSource, request_id: str
) -> Callable[[LLMProvider, ModelUsed], AnalysisResult]:
    def attempt(provider: LLMProvider, model_used: ModelUsed) -> AnalysisResult:
        return pipeline.run(
            answer=answer,
            question=question,
            evidence_source=evidence_source,
            provider=provider,
            request_id=request_id,
            model_used=model_used,
        )

    return attempt


def _other_models_first(candidates: Iterator[Candidate], avoid: ModelUsed) -> Iterator[Candidate]:
    """Yields `avoid` last, so a chat answer is checked by a different
    model whenever the pool has one.
    """
    deferred: list[Candidate] = []
    for candidate in candidates:
        if candidate.model_used == avoid:
            deferred.append(candidate)
        else:
            yield candidate
    yield from deferred


def _record_check(checker: ModelUsed, result: AnalysisResult) -> None:
    # A no-evidence result never reached the checking model.
    if result.reason != AbstentionReason.NO_EVIDENCE_CONFIGURED:
        observer.record_check_verdict(checker.provider, checker.model, result)


@app.post("/v1/verify", response_model=AnalysisResult)
def verify(request: VerifyRequestIn, api_key: str = Depends(enforce_rate_limit)) -> AnalysisResult:
    request_id = bind_request_id()
    settings = get_settings()

    cache_key = compute_cache_key(
        answer=request.answer,
        question=request.question,
        evidence=request.evidence,
        evidence_source=request.evidence_source.value,
        model_provider=request.model_prefs.provider.value,
        allow_free_pool=request.model_prefs.allow_free_pool,
        pinned_model=request.model_prefs.pinned_model,
        user_api_key=request.model_prefs.user_api_key,
    )
    cache_store = _get_cache_store(settings) if settings.cache_enabled else None

    if cache_store is not None:
        cached = cache_store.get(cache_key)
        if cached is not None:
            _logger.info("verify_cache_hit", cache_key=cache_key)
            return rescore(cached).model_copy(
                update={
                    "request_id": request_id,
                    "timings_ms": Timings(total=0, retrieval=0, extraction=0, verification=0),
                }
            )

    try:
        evidence_source = resolve_evidence_source(request, settings, app.state.custom_evidence_source)
        candidates = resolve_candidates(request.model_prefs, settings)
        attempt = _verify_attempt(
            answer=request.answer,
            question=request.question,
            evidence_source=evidence_source,
            request_id=request_id,
        )
        result, checker = _run_with_failover(candidates, attempt, settings, Role.VERIFY)
    except ResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _record_check(checker, result)

    # A NOT_ENOUGH_INFO verdict isn't cached: it's the only verdict a flaky
    # model response can produce (a missing quote downgrades SUPPORTED to
    # NOT_ENOUGH_INFO; it can never fake SUPPORTED, and CONTRADICTED wins
    # the verdict regardless), so caching it would serve one bad response
    # to every identical request for cache_ttl_s. Seen in production: a
    # three-claim answer the evidence fully supports got NOT_ENOUGH_INFO
    # once and GROUNDED on every retry.
    if cache_store is not None and result.verdict != Verdict.NOT_ENOUGH_INFO:
        cache_store.set(cache_key, result, ttl_s=settings.cache_ttl_s)

    return result


@app.post("/v1/chat", response_model=ChatResult)
def chat(request: ChatRequestIn, api_key: str = Depends(enforce_rate_limit)) -> ChatResult:
    """A model answers `question`; the answer is then checked like any
    `/v1/verify` answer, against `evidence` or `evidence_source` only. Not
    cached: a fresh answer is the point of asking.
    """
    request_id = bind_request_id()
    settings = get_settings()
    # One budget for both steps, so a chat request stays inside the same
    # time limit as a /v1/verify request plus one answer call.
    answer_start = monotonic()
    try:
        evidence_source = resolve_evidence_source(request, settings, app.state.custom_evidence_source)
        response, answer_model = _run_with_failover(
            resolve_candidates(request.model_prefs, settings),
            lambda provider, _model_used: generate_answer(
                provider, request.question, history=request.history, evidence=request.evidence
            ),
            settings,
            Role.ANSWER,
            started=answer_start,
        )
        answer_ms = int((monotonic() - answer_start) * 1000)
        answer = response.text.strip()

        attempt = _verify_attempt(
            answer=answer,
            question=request.question,
            evidence_source=evidence_source,
            request_id=request_id,
        )
        candidates = _other_models_first(resolve_candidates(request.model_prefs, settings), answer_model)
        verification, checker = _run_with_failover(
            candidates, attempt, settings, Role.VERIFY, started=answer_start
        )
    except ResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    observer.record_answer_verdict(answer_model.provider, answer_model.model, verification)
    _record_check(checker, verification)
    return ChatResult(
        request_id=request_id,
        answer=answer,
        answer_model=answer_model,
        answer_ms=answer_ms,
        verification=verification,
    )


@app.get("/v1/models/stats")
def model_stats(api_key: str = Depends(require_api_key)) -> dict:
    """Per-model counters since this process started (`observe.py`)."""
    return observer.snapshot(in_cooldown=pool_health())
