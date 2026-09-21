"""Shared HTTP error classification for provider implementations.

Centralized so every provider maps transport failures to the same
halludetect.llm.exceptions hierarchy by status code / exception type,
not by inspecting response text.
"""
import httpx

from halludetect.llm.exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

DEFAULT_TIMEOUT_S = 30.0


def raise_for_provider_error(response: httpx.Response, provider: str) -> None:
    if response.status_code in (401, 403):
        raise LLMAuthError(f"{provider}: authentication failed (HTTP {response.status_code})")
    if response.status_code == 429:
        raise LLMRateLimitError(f"{provider}: rate limited (HTTP 429)")
    if response.status_code >= 400:
        raise LLMResponseError(f"{provider}: HTTP {response.status_code}: {response.text[:300]}")


def wrap_transport_error(exc: httpx.HTTPError, provider: str) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"{provider}: request timed out")
    return LLMResponseError(f"{provider}: transport error: {exc}")
