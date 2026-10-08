"""Shared HTTP error classification for provider implementations.

Centralized so every provider maps transport failures to the same
halludetect.llm.exceptions hierarchy by status code / exception type,
not by inspecting response text.
"""
import time
from typing import Any

import httpx

from halludetect.llm.exceptions import (
    LLMAuthError,
    LLMError,
    LLMModelAccessError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

DEFAULT_TIMEOUT_S = 30.0
# Total wall-clock limit on one chat call. httpx's timeout is per network
# operation, and OpenRouter keeps a slow non-streaming completion alive by
# sending whitespace, so without this a call can run for minutes. Free
# models answer in well under this when they answer at all (p95 per
# request, several calls, is ~46s on golden set B).
CALL_DEADLINE_S = 60.0
# A 429 whose limit resets further out than this is a used-up quota, not
# a burst limit worth retrying.
QUOTA_RESET_MIN_S = 15 * 60


def _quota_reset_at(response: httpx.Response) -> float | None:
    """Epoch seconds the quota resets, if the 429's rate-limit headers say
    nothing is left until well past a retry's horizon.
    """
    if response.headers.get("x-ratelimit-remaining") != "0":
        return None
    try:
        reset = float(response.headers["x-ratelimit-reset"])
    except (KeyError, ValueError):
        return None
    reset_s = reset / 1000 if reset > 1e11 else reset  # OpenRouter sends milliseconds
    return reset_s if reset_s - time.time() >= QUOTA_RESET_MIN_S else None


def raise_for_provider_error(response: httpx.Response, provider: str) -> None:
    if response.status_code == 401:
        raise LLMAuthError(f"{provider}: authentication failed (HTTP 401)")
    if response.status_code in (402, 403):
        # 402: OpenRouter lists some models at $0 that still need purchased
        # credits; like 403, another model on the same key can work.
        raise LLMModelAccessError(
            f"{provider}: access to this model denied (HTTP {response.status_code}): {response.text[:300]}"
        )
    if response.status_code == 429:
        reset_at = _quota_reset_at(response)
        if reset_at is not None:
            raise LLMQuotaExhaustedError(
                f"{provider}: quota used up until {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(reset_at))}"
                " (HTTP 429)",
                reset_at=reset_at,
            )
        raise LLMRateLimitError(f"{provider}: rate limited (HTTP 429)")
    if response.status_code >= 400:
        raise LLMResponseError(f"{provider}: HTTP {response.status_code}: {response.text[:300]}")


def post_json(
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    provider: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> httpx.Response:
    """POST and read the whole body, raising LLMTimeoutError once
    CALL_DEADLINE_S has passed. Checked between chunks, so a server that
    goes fully silent can overrun it by up to `timeout_s`.
    """
    deadline = time.monotonic() + CALL_DEADLINE_S
    try:
        with httpx.stream("POST", url, headers=headers, json=json, timeout=timeout_s) as response:
            body = bytearray()
            # Raw bytes: the Response built below decodes them per its headers.
            for chunk in response.iter_raw():
                body += chunk
                if time.monotonic() > deadline:
                    raise LLMTimeoutError(f"{provider}: no complete response after {CALL_DEADLINE_S:.0f}s")
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=response.request,
            )
    except httpx.HTTPError as exc:
        raise wrap_transport_error(exc, provider) from exc


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
