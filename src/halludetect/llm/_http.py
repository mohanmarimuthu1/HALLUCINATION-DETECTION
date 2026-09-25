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


def extract_chat_content(data: dict, provider: str, model: str) -> str:
    """Pulls `choices[0].message.content` out of an OpenAI-shaped chat
    completion response (openrouter.py, openai.py,
    custom_openai_compat.py all use this exact response shape).

    A free/reasoning model can return HTTP 200 with `content: null` when it
    exhausts `max_tokens` while still "thinking" (`finish_reason: length`,
    the actual answer never written) - confirmed live against a real
    OpenRouter free model, not a hypothetical. Raising here instead of
    returning `None` as `LLMResponse.text` matters because
    `complete_structured` calls `.strip()` on that text unconditionally;
    without this check, that failure mode surfaces as a confusing
    `AttributeError` deep in JSON parsing instead of the
    `LLMResponseError` every other provider failure already produces.
    """
    choices = data.get("choices") or []
    if not choices:
        raise LLMResponseError(f"{provider}: no choices in response for model {model}")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        finish_reason = choices[0].get("finish_reason", "unknown")
        raise LLMResponseError(
            f"{provider}: no content in response for model {model} (finish_reason={finish_reason})"
        )
    return content
