"""FastAPI app: `POST /v1/verify`, `GET /healthz` (Phase 5.1), per-key auth
and token-bucket rate limiting on `/v1/verify` (Phase 5.2), a result cache
in front of the pipeline (Phase 6.1).

This is `detect.pipeline.run()`'s first real external caller. This module
does request parsing, auth/rate-limit enforcement, a cache lookup,
resolution (`api.resolve`), and response serialization against
`docs/contract.md`.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from halludetect.api.ratelimit import enforce_rate_limit
from halludetect.api.resolve import ResolutionError, resolve_evidence_source, resolve_provider
from halludetect.api.schemas import VerifyRequestIn
from halludetect.cache.base import CacheStore
from halludetect.cache.key import compute_cache_key
from halludetect.cache.store import DiskCacheStore
from halludetect.detect import pipeline
from halludetect.detect.schemas import AnalysisResult, Timings
from halludetect.llm.exceptions import LLMError
from halludetect.logging import bind_request_id, configure_logging, get_logger
from halludetect.settings import Settings, get_settings

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


def _get_cache_store(settings: Settings) -> CacheStore:
    global _cache_store
    if _cache_store is None:
        _cache_store = DiskCacheStore(settings.cache_dir)
    return _cache_store


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


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
            return cached.model_copy(
                update={
                    "request_id": request_id,
                    "timings_ms": Timings(total=0, retrieval=0, extraction=0, verification=0),
                }
            )

    try:
        evidence_source = resolve_evidence_source(request, settings, app.state.custom_evidence_source)
        provider, model_used = resolve_provider(request.model_prefs, settings)
    except ResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"no LLM provider available: {exc}") from exc

    try:
        result = pipeline.run(
            answer=request.answer,
            question=request.question,
            evidence_source=evidence_source,
            provider=provider,
            request_id=request_id,
            model_used=model_used,
        )
    except LLMError as exc:
        _logger.warning("verify_llm_error", error=str(exc))
        raise HTTPException(status_code=502, detail=f"LLM provider failed: {exc}") from exc

    if cache_store is not None:
        cache_store.set(cache_key, result, ttl_s=settings.cache_ttl_s)

    return result
