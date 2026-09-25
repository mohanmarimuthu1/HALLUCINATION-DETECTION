"""FastAPI app: `POST /v1/verify`, `GET /healthz` (Phase 5.1), per-key auth
and token-bucket rate limiting on `/v1/verify` (Phase 5.2).

This is `detect.pipeline.run()`'s first real external caller. Cost/usage
tracking (5.3) is a separate, not-yet-built concern - this module does
request parsing, auth/rate-limit enforcement, resolution (`api.resolve`),
and response serialization against `docs/contract.md`.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from halludetect.api.ratelimit import enforce_rate_limit
from halludetect.api.resolve import ResolutionError, resolve_evidence_source, resolve_provider
from halludetect.api.schemas import VerifyRequestIn
from halludetect.detect import pipeline
from halludetect.detect.schemas import AnalysisResult
from halludetect.evidence.base import EvidenceSource
from halludetect.llm.exceptions import LLMError
from halludetect.logging import bind_request_id, configure_logging, get_logger
from halludetect.settings import get_settings

configure_logging()
_logger = get_logger(__name__)

app = FastAPI(title="HALLUDETECT API", version="1.0.0")

# Registration point for an embedding deployment's own retriever
# (docs/contract.md: evidence_source == "custom", Phase 3.4's plug-in hook).
# A Python callable can't be expressed in a JSON request body, so "custom"
# is only usable when a deployment sets this at startup; otherwise a request
# for it is a 400, never a silent fall-through to NoEvidenceSource.
app.state.custom_evidence_source: EvidenceSource | None = None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/verify", response_model=AnalysisResult)
def verify(request: VerifyRequestIn, api_key: str = Depends(enforce_rate_limit)) -> AnalysisResult:
    request_id = bind_request_id()
    settings = get_settings()

    try:
        evidence_source = resolve_evidence_source(request, settings, app.state.custom_evidence_source)
        provider, model_used = resolve_provider(request.model_prefs, settings)
    except ResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=f"no LLM provider available: {exc}") from exc

    try:
        return pipeline.run(
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
